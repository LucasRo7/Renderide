//! Shared admission helpers for texture-family upload IPC handlers.

use std::sync::Arc;

use super::super::AssetTransferQueue;

/// Immutable facts used to classify one texture-family data upload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TextureUploadAdmissionFacts {
    /// Shared-memory descriptor payload length from the host command.
    pub(super) payload_len: i32,
    /// Whether the matching format command has been received.
    pub(super) has_format: bool,
    /// Number of commands currently deferred for this texture family.
    pub(super) pending_len: usize,
    /// Maximum deferred command count for this texture family.
    pub(super) max_pending: usize,
    /// Whether the GPU device and queue are attached.
    pub(super) gpu_attached: bool,
    /// Whether the resident GPU texture already exists.
    pub(super) has_resident: bool,
}

/// Pure admission decision before any allocation side effects are attempted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TextureUploadAdmissionDecision {
    /// Empty data commands are ignored.
    IgnoreEmptyPayload,
    /// Data arrived before the format row and cannot be retained safely.
    RejectMissingFormat,
    /// Retain the command until the GPU device/queue is attached.
    DeferUntilGpuAttached,
    /// The pre-GPU pending queue is full, so the command is dropped.
    DropPendingFull,
    /// A resident texture may be allocatable from the stored format row.
    TryAllocateResident,
    /// All prerequisites are ready and the upload can be enqueued.
    Ready,
}

/// Pure decision after a missing resident texture has had one allocation attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TextureUploadPostAllocationDecision {
    /// The allocation attempt created the resident texture.
    Ready,
    /// Retain the command until a resident texture can be created.
    DeferMissingResident,
    /// The command cannot be retained because the pending queue is full.
    DropMissingResidentPendingFull,
}

/// Classifies the first admission phase without mutating queues or pools.
pub(super) fn plan_texture_upload_admission(
    facts: TextureUploadAdmissionFacts,
) -> TextureUploadAdmissionDecision {
    if facts.payload_len <= 0 {
        return TextureUploadAdmissionDecision::IgnoreEmptyPayload;
    }
    if !facts.has_format {
        return TextureUploadAdmissionDecision::RejectMissingFormat;
    }
    if !facts.gpu_attached {
        if facts.pending_len >= facts.max_pending {
            return TextureUploadAdmissionDecision::DropPendingFull;
        }
        return TextureUploadAdmissionDecision::DeferUntilGpuAttached;
    }
    if !facts.has_resident {
        return TextureUploadAdmissionDecision::TryAllocateResident;
    }
    TextureUploadAdmissionDecision::Ready
}

/// Classifies the post-allocation phase without mutating queues or pools.
pub(super) fn plan_texture_post_allocation(
    pending_len: usize,
    max_pending: usize,
    has_resident_after_allocation: bool,
) -> TextureUploadPostAllocationDecision {
    if has_resident_after_allocation {
        return TextureUploadPostAllocationDecision::Ready;
    }
    if pending_len >= max_pending {
        return TextureUploadPostAllocationDecision::DropMissingResidentPendingFull;
    }
    TextureUploadPostAllocationDecision::DeferMissingResident
}

/// Configuration for [`admit_texture_upload_data`].
pub(super) struct TextureUploadAdmission<
    'a,
    D,
    HasFormat,
    PendingLen,
    PushPending,
    HasResident,
    Flush,
> where
    HasFormat: Fn(&AssetTransferQueue, i32) -> bool,
    PendingLen: Fn(&AssetTransferQueue) -> usize,
    PushPending: Fn(&mut AssetTransferQueue, D),
    HasResident: Fn(&AssetTransferQueue, i32) -> bool,
    Flush: Fn(&mut AssetTransferQueue, &Arc<wgpu::Device>),
{
    /// Asset queue receiving the upload or deferral.
    pub(super) queue: &'a mut AssetTransferQueue,
    /// Upload command being admitted.
    pub(super) data: D,
    /// Host asset id from the upload command.
    pub(super) asset_id: i32,
    /// Payload length from the upload command's shared-memory descriptor.
    pub(super) payload_len: i32,
    /// Diagnostic asset family label.
    pub(super) kind: &'static str,
    /// Name of the format command expected before data arrives.
    pub(super) format_command: &'static str,
    /// Maximum number of deferred upload commands for this family.
    pub(super) max_pending: usize,
    /// Whether a format row is known for `asset_id`.
    pub(super) has_format: HasFormat,
    /// Current deferred upload queue length.
    pub(super) pending_len: PendingLen,
    /// Pushes `data` into the deferred upload queue.
    pub(super) push_pending: PushPending,
    /// Whether the resident GPU texture already exists.
    pub(super) has_resident: HasResident,
    /// Attempts to allocate missing textures from pending format rows.
    pub(super) flush_allocations: Flush,
}

/// Returns `Some(data)` when the texture upload can be enqueued immediately.
///
/// Empty payloads are ignored, missing formats are dropped with a warning, and uploads are
/// deferred when the GPU device/queue or resident texture is not ready yet.
pub(super) fn admit_texture_upload_data<D, HasFormat, PendingLen, PushPending, HasResident, Flush>(
    admission: TextureUploadAdmission<
        '_,
        D,
        HasFormat,
        PendingLen,
        PushPending,
        HasResident,
        Flush,
    >,
) -> Option<D>
where
    HasFormat: Fn(&AssetTransferQueue, i32) -> bool,
    PendingLen: Fn(&AssetTransferQueue) -> usize,
    PushPending: Fn(&mut AssetTransferQueue, D),
    HasResident: Fn(&AssetTransferQueue, i32) -> bool,
    Flush: Fn(&mut AssetTransferQueue, &Arc<wgpu::Device>),
{
    let TextureUploadAdmission {
        queue,
        data,
        asset_id,
        payload_len,
        kind,
        format_command,
        max_pending,
        has_format,
        pending_len,
        push_pending,
        has_resident,
        flush_allocations,
    } = admission;

    match plan_texture_upload_admission(TextureUploadAdmissionFacts {
        payload_len,
        has_format: has_format(queue, asset_id),
        pending_len: pending_len(queue),
        max_pending,
        gpu_attached: queue.gpu.is_attached(),
        has_resident: has_resident(queue, asset_id),
    }) {
        TextureUploadAdmissionDecision::IgnoreEmptyPayload => None,
        TextureUploadAdmissionDecision::RejectMissingFormat => {
            logger::warn!("{kind} {asset_id}: {format_command} before format; ignored");
            None
        }
        TextureUploadAdmissionDecision::DropPendingFull => {
            logger::warn!("{kind} {asset_id}: pending upload queue full; dropping");
            None
        }
        TextureUploadAdmissionDecision::DeferUntilGpuAttached => {
            push_pending(queue, data);
            None
        }
        TextureUploadAdmissionDecision::Ready => Some(data),
        TextureUploadAdmissionDecision::TryAllocateResident => {
            let device = queue.gpu.gpu_device.clone()?;
            flush_allocations(queue, &device);
            match plan_texture_post_allocation(
                pending_len(queue),
                max_pending,
                has_resident(queue, asset_id),
            ) {
                TextureUploadPostAllocationDecision::Ready => Some(data),
                TextureUploadPostAllocationDecision::DeferMissingResident => {
                    push_pending(queue, data);
                    None
                }
                TextureUploadPostAllocationDecision::DropMissingResidentPendingFull => {
                    logger::warn!(
                        "{kind} {asset_id}: no GPU texture and pending full; dropping data"
                    );
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        TextureUploadAdmissionDecision, TextureUploadAdmissionFacts,
        TextureUploadPostAllocationDecision, plan_texture_post_allocation,
        plan_texture_upload_admission,
    };

    fn facts() -> TextureUploadAdmissionFacts {
        TextureUploadAdmissionFacts {
            payload_len: 16,
            has_format: true,
            pending_len: 0,
            max_pending: 2,
            gpu_attached: true,
            has_resident: true,
        }
    }

    #[test]
    fn empty_payload_is_ignored() {
        let decision = plan_texture_upload_admission(TextureUploadAdmissionFacts {
            payload_len: 0,
            ..facts()
        });

        assert_eq!(decision, TextureUploadAdmissionDecision::IgnoreEmptyPayload);
    }

    #[test]
    fn missing_format_is_rejected_before_gpu_state() {
        let decision = plan_texture_upload_admission(TextureUploadAdmissionFacts {
            has_format: false,
            gpu_attached: false,
            ..facts()
        });

        assert_eq!(
            decision,
            TextureUploadAdmissionDecision::RejectMissingFormat
        );
    }

    #[test]
    fn gpu_unavailable_upload_is_deferred_when_queue_has_room() {
        let decision = plan_texture_upload_admission(TextureUploadAdmissionFacts {
            gpu_attached: false,
            pending_len: 1,
            max_pending: 2,
            ..facts()
        });

        assert_eq!(
            decision,
            TextureUploadAdmissionDecision::DeferUntilGpuAttached
        );
    }

    #[test]
    fn gpu_unavailable_upload_is_dropped_when_pending_queue_is_full() {
        let decision = plan_texture_upload_admission(TextureUploadAdmissionFacts {
            gpu_attached: false,
            pending_len: 2,
            max_pending: 2,
            ..facts()
        });

        assert_eq!(decision, TextureUploadAdmissionDecision::DropPendingFull);
    }

    #[test]
    fn missing_resident_requests_allocation_attempt() {
        let decision = plan_texture_upload_admission(TextureUploadAdmissionFacts {
            has_resident: false,
            ..facts()
        });

        assert_eq!(
            decision,
            TextureUploadAdmissionDecision::TryAllocateResident
        );
    }

    #[test]
    fn post_allocation_defers_or_drops_by_pending_capacity() {
        assert_eq!(
            plan_texture_post_allocation(1, 2, false),
            TextureUploadPostAllocationDecision::DeferMissingResident
        );
        assert_eq!(
            plan_texture_post_allocation(2, 2, false),
            TextureUploadPostAllocationDecision::DropMissingResidentPendingFull
        );
        assert_eq!(
            plan_texture_post_allocation(2, 2, true),
            TextureUploadPostAllocationDecision::Ready
        );
    }
}
