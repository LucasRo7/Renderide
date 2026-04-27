//! Small staging-ring state machine for non-blocking GPU readbacks.

use crossbeam_channel as mpsc;

/// Number of staging slots kept alive for asynchronous Hi-Z readback.
pub(crate) const HIZ_STAGING_RING: usize = 3;

/// Receiver produced by a `map_async` readback request.
///
/// `crossbeam_channel::Receiver` is `Send + Sync`, which lets Hi-Z state remain shareable across
/// rayon workers after command submission. `std::sync::mpsc::Receiver` is only `Send`.
pub(crate) type MapRecv = mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>;

/// Creates a fixed-size optional slot array for ring-owned pending work.
pub(crate) const fn pending_none_array<T>() -> [Option<T>; HIZ_STAGING_RING] {
    [None, None, None]
}

/// Creates a cleared fixed-size boolean slot array.
const fn pending_bool_array() -> [bool; HIZ_STAGING_RING] {
    [false; HIZ_STAGING_RING]
}

/// Ownership state for a fixed-size GPU staging readback ring.
pub(crate) struct GpuReadbackRing {
    /// Next slot that an encode pass may write when [`Self::can_claim_next_slot`] returns true.
    write_idx: usize,
    /// Transient submit handoff captured after encode and consumed when the queue callback is built.
    encoded_slot: Option<usize>,
    /// Slots whose copy-to-staging command has been recorded but not confirmed by the driver.
    pending_submit: [bool; HIZ_STAGING_RING],
    /// Slots whose submit callback has fired but whose map request has not been issued yet.
    submit_done: [bool; HIZ_STAGING_RING],
    /// Pending map callbacks for the primary staging buffers.
    primary_pending: [Option<MapRecv>; HIZ_STAGING_RING],
    /// Pending map callbacks for optional secondary staging buffers.
    secondary_pending: Option<[Option<MapRecv>; HIZ_STAGING_RING]>,
}

impl Default for GpuReadbackRing {
    fn default() -> Self {
        Self {
            write_idx: 0,
            encoded_slot: None,
            pending_submit: pending_bool_array(),
            submit_done: pending_bool_array(),
            primary_pending: pending_none_array(),
            secondary_pending: None,
        }
    }
}

impl GpuReadbackRing {
    /// Resets all slot ownership and pending callback state.
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    /// Returns the slot that the next successful encode should target.
    pub(crate) fn next_write_slot(&self) -> usize {
        self.write_idx
    }

    /// Clears a stale encoded-slot handoff before a new encode attempt.
    pub(crate) fn clear_encoded_slot(&mut self) {
        self.encoded_slot = None;
    }

    /// Takes the encoded-slot handoff for queue-submit callback installation.
    pub(crate) fn take_encoded_slot(&mut self) -> Option<usize> {
        self.encoded_slot.take()
    }

    /// Ensures secondary pending-map storage matches the current readback layout.
    pub(crate) fn set_secondary_enabled(&mut self, enabled: bool) {
        match (enabled, self.secondary_pending.is_some()) {
            (true, false) => self.secondary_pending = Some(pending_none_array()),
            (false, true) => self.secondary_pending = None,
            _ => {}
        }
    }

    /// Returns true when the next write slot has no outstanding submit or map work.
    pub(crate) fn can_claim_next_slot(&self, requires_secondary: bool) -> bool {
        let idx = self.write_idx;
        if self.pending_submit[idx] || self.submit_done[idx] || self.primary_pending[idx].is_some()
        {
            return false;
        }

        if requires_secondary {
            if let Some(ref secondary) = self.secondary_pending {
                return secondary[idx].is_none();
            }
        }
        true
    }

    /// Claims the current write slot after encode commands have been recorded.
    pub(crate) fn claim_next_slot(&mut self) -> usize {
        let slot = self.write_idx;
        self.pending_submit[slot] = true;
        self.write_idx = (slot + 1) % HIZ_STAGING_RING;
        self.encoded_slot = Some(slot);
        slot
    }

    /// Marks that the queue submit containing `slot` has completed.
    pub(crate) fn mark_submit_done(&mut self, slot: usize) {
        debug_assert!(slot < HIZ_STAGING_RING);
        self.submit_done[slot] = true;
    }

    /// Returns mutable access to the primary pending map receiver for `slot`.
    pub(crate) fn primary_pending_mut(&mut self, slot: usize) -> &mut Option<MapRecv> {
        debug_assert!(slot < HIZ_STAGING_RING);
        &mut self.primary_pending[slot]
    }

    /// Returns mutable access to the secondary pending map receiver for `slot`, if enabled.
    pub(crate) fn secondary_pending_mut(&mut self, slot: usize) -> Option<&mut Option<MapRecv>> {
        debug_assert!(slot < HIZ_STAGING_RING);
        self.secondary_pending
            .as_mut()
            .map(|pending| &mut pending[slot])
    }

    /// Issues `map_async` for every slot whose submit callback has completed.
    pub(crate) fn start_ready_maps(
        &mut self,
        primary_staging: Option<&[wgpu::Buffer; HIZ_STAGING_RING]>,
        secondary_staging: Option<&[wgpu::Buffer; HIZ_STAGING_RING]>,
    ) {
        let Some(primary_staging) = primary_staging else {
            self.clear_submit_state();
            return;
        };

        self.set_secondary_enabled(secondary_staging.is_some());
        for slot in 0..HIZ_STAGING_RING {
            if !self.submit_done[slot] {
                continue;
            }
            self.submit_done[slot] = false;
            self.pending_submit[slot] = false;

            if self.primary_pending[slot].is_some() {
                continue;
            }

            self.primary_pending[slot] = Some(start_map_read(&primary_staging[slot]));
            if let Some(secondary_staging) = secondary_staging {
                if let Some(secondary_pending) = self.secondary_pending.as_mut() {
                    if secondary_pending[slot].is_none() {
                        secondary_pending[slot] = Some(start_map_read(&secondary_staging[slot]));
                    }
                }
            }
        }
    }

    /// Clears submit ownership flags when the staging buffers backing them no longer exist.
    fn clear_submit_state(&mut self) {
        self.pending_submit = pending_bool_array();
        self.submit_done = pending_bool_array();
    }
}

/// Starts a `map_async` read on `buffer` and returns the callback receiver.
fn start_map_read(buffer: &wgpu::Buffer) -> MapRecv {
    let slice = buffer.slice(..);
    let (tx, rx) = mpsc::bounded::<Result<(), wgpu::BufferAsyncError>>(1);
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::GpuReadbackRing;

    #[test]
    fn default_ring_can_claim_first_primary_slot() {
        let ring = GpuReadbackRing::default();

        assert_eq!(ring.next_write_slot(), 0);
        assert!(ring.can_claim_next_slot(false));
    }

    #[test]
    fn claimed_slot_blocks_until_submit_is_mapped_or_cleared() {
        let mut ring = GpuReadbackRing::default();

        assert_eq!(ring.claim_next_slot(), 0);
        assert_eq!(ring.next_write_slot(), 1);
        assert!(ring.can_claim_next_slot(false));

        assert_eq!(ring.claim_next_slot(), 1);
        assert!(ring.can_claim_next_slot(false));

        assert_eq!(ring.claim_next_slot(), 2);
        assert!(!ring.can_claim_next_slot(false));

        ring.reset();
        assert!(ring.can_claim_next_slot(false));
        assert_eq!(ring.next_write_slot(), 0);
    }

    #[test]
    fn encoded_slot_is_taken_once() {
        let mut ring = GpuReadbackRing::default();

        let slot = ring.claim_next_slot();

        assert_eq!(ring.take_encoded_slot(), Some(slot));
        assert_eq!(ring.take_encoded_slot(), None);
    }
}
