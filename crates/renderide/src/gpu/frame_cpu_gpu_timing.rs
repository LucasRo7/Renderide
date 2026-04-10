//! Wall-clock frame timing for the debug HUD: CPU time until the last queue submit, GPU time until
//! that submission completes ([`wgpu::Queue::on_submitted_work_done`]).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Per-tick state for AAA-style CPU/GPU frame metrics (see [`GpuContext`](super::GpuContext)).
#[derive(Debug, Default)]
pub struct FrameCpuGpuTiming {
    /// Monotonic id; callbacks ignore stale generations after a new [`Self::begin_frame`].
    generation: u64,
    /// Start of the winit frame tick ([`crate::app::RenderideApp::tick_frame`]).
    frame_start: Option<Instant>,
    /// Number of tracked submits this tick (1-based).
    submit_seq: u32,
    /// Last submit instant in this tick (for CPU time).
    last_submit_at: Option<Instant>,
    /// Set in [`Self::end_frame`] to the last submit index for this tick.
    finalized_seq: Option<u32>,
    /// GPU duration (ms) per submit index, filled asynchronously by wgpu callbacks.
    pending_gpu_by_seq: HashMap<u32, f64>,
    /// Milliseconds from the winit tick start to the last tracked submit ([`Self::end_frame`]).
    pub(crate) cpu_until_submit_ms: Option<f64>,
    /// Milliseconds from that last submit until GPU completion for that batch (async callback).
    pub(crate) gpu_after_submit_ms: Option<f64>,
}

impl FrameCpuGpuTiming {
    /// Starts tracking for a new winit tick; clears prior tick metrics.
    pub fn begin_frame(&mut self, frame_start: Instant) {
        self.generation = self.generation.wrapping_add(1);
        self.frame_start = Some(frame_start);
        self.submit_seq = 0;
        self.last_submit_at = None;
        self.finalized_seq = None;
        self.pending_gpu_by_seq.clear();
        self.cpu_until_submit_ms = None;
        self.gpu_after_submit_ms = None;
    }

    /// Call after all render graph submits for this tick (last submit index is known).
    pub fn end_frame(&mut self) {
        if self.frame_start.is_none() {
            return;
        }
        self.finalized_seq = Some(self.submit_seq);
        self.cpu_until_submit_ms = match (self.frame_start, self.last_submit_at) {
            (Some(s), Some(t)) => Some(t.duration_since(s).as_secs_f64() * 1000.0),
            _ => None,
        };
        if self.submit_seq > 0 {
            if let Some(ms) = self.pending_gpu_by_seq.remove(&self.submit_seq) {
                self.gpu_after_submit_ms = Some(ms);
            }
        }
    }

    fn record_gpu_done(&mut self, gen: u64, seq: u32, gpu_ms: f64) {
        if gen != self.generation {
            return;
        }
        self.pending_gpu_by_seq.insert(seq, gpu_ms);
        if self.finalized_seq == Some(seq) {
            self.gpu_after_submit_ms = Some(gpu_ms);
        }
    }

    /// Records a tracked submit: returns `(generation, submit_seq)` if tracking is active.
    pub fn on_before_tracked_submit(&mut self) -> Option<(u64, u32)> {
        self.frame_start?;
        self.submit_seq = self.submit_seq.saturating_add(1);
        let seq = self.submit_seq;
        self.last_submit_at = Some(Instant::now());
        Some((self.generation, seq))
    }
}

/// Shared timing state held by [`super::GpuContext`].
pub type FrameCpuGpuTimingHandle = Arc<Mutex<FrameCpuGpuTiming>>;

/// Builds a callback that records GPU duration for `seq` when the queue finishes that submission.
pub fn make_gpu_done_callback(
    handle: FrameCpuGpuTimingHandle,
    generation: u64,
    seq: u32,
    submit_at: Instant,
) -> impl FnOnce() + Send + 'static {
    move || {
        let gpu_ms = submit_at.elapsed().as_secs_f64() * 1000.0;
        let mut g = handle.lock().unwrap_or_else(|e| e.into_inner());
        g.record_gpu_done(generation, seq, gpu_ms);
    }
}
