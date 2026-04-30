//! Builds the [`crate::shared::PerformanceState`] payload carried on every
//! [`crate::shared::FrameStartData`] sent to the host.
//!
//! Contract (matches Renderite.Unity, consumed by `FrooxEngine.PerformanceMetrics`):
//! - `immediate_fps` — instantaneous, derived from the current tick's wall-clock interval
//!   ([`crate::frontend::RendererFrontend::on_tick_frame_wall_clock`]). No smoothing.
//! - `fps` — exponential moving average of `immediate_fps`. Updated and emitted every frame.
//! - `render_time` — most recently completed GPU submit→idle wall-clock duration in seconds
//!   ([`crate::gpu::GpuContext::last_completed_gpu_render_time_seconds`]); excludes the post-submit
//!   present/vsync block. Reports `-1.0` when no GPU completion callback has fired yet, mirroring the
//!   Renderite.Unity `XRStats.TryGetGPUTimeLastFrame` sentinel.
//! - `rendered_frames_since_last` — number of completed renderer ticks since the previous
//!   `FrameStartData` send. `1` in lockstep, `> 1` when the renderer ticked multiple times per
//!   host submit (i.e. host is slow and the renderer kept rendering). Drives
//!   `FrooxEngine.PerformanceStats.RenderedFramesSinceLastTick`.
//!
//! A new [`PerformanceState`] is built on every tick where `wall_interval_us > 0` (i.e. starting
//! from the second tick); the host treats a non-null `FrameStartData.performance` as the latest
//! sample, so emitting every frame keeps `ImmediateFPS` and `RenderTime` in lock-step with the
//! actual frame loop. This is **not** GPU instrumentation; for that, see
//! [`crate::gpu::frame_cpu_gpu_timing`].

use std::time::Instant;

use crate::shared::PerformanceState;

/// Exponential moving average blend factor for the `fps` field (`fps` blends toward `immediate_fps`).
pub(crate) const FPS_EMA_ALPHA: f32 = 0.1;

/// Sentinel reported in `render_time` until the first GPU completion callback has fired, matching
/// the Renderite.Unity behavior of `state.renderTime = -1` when `XRStats.TryGetGPUTimeLastFrame`
/// has no sample yet.
pub(crate) const RENDER_TIME_UNAVAILABLE: f32 = -1.0;

/// Mutable performance accumulator that feeds outgoing frame-start payloads.
pub(crate) struct FrameStartPerformanceState {
    last_tick_wall_start: Option<Instant>,
    wall_interval_us_for_perf: u64,
    last_render_time_seconds: f32,
    smoothed_fps: Option<f32>,
    rendered_frames_since_last: i32,
}

impl Default for FrameStartPerformanceState {
    fn default() -> Self {
        Self {
            last_tick_wall_start: None,
            wall_interval_us_for_perf: 0,
            last_render_time_seconds: RENDER_TIME_UNAVAILABLE,
            smoothed_fps: None,
            rendered_frames_since_last: 0,
        }
    }
}

impl FrameStartPerformanceState {
    /// Records wall-clock spacing between app-driver frame ticks.
    pub(crate) fn on_tick_frame_wall_clock(&mut self, now: Instant) {
        self.wall_interval_us_for_perf = self
            .last_tick_wall_start
            .map_or(0, |t| now.duration_since(t).as_micros() as u64);
        self.last_tick_wall_start = Some(now);
    }

    /// Stores the most recently completed GPU submit-to-idle interval.
    pub(crate) fn set_last_render_time_seconds(&mut self, render_time_seconds: Option<f32>) {
        self.last_render_time_seconds = render_time_seconds.unwrap_or(RENDER_TIME_UNAVAILABLE);
    }

    /// Increments the renderer-tick counter captured by the next frame-start send.
    pub(crate) fn note_render_tick_complete(&mut self) {
        self.rendered_frames_since_last = self.rendered_frames_since_last.saturating_add(1);
    }

    /// Captures and resets the rendered-frame counter while producing the next performance sample.
    pub(crate) fn step_for_frame_start(&mut self) -> Option<PerformanceState> {
        let rendered_frames_since_last = std::mem::replace(&mut self.rendered_frames_since_last, 0);
        step_frame_performance(
            self.wall_interval_us_for_perf,
            self.last_render_time_seconds,
            &mut self.smoothed_fps,
            rendered_frames_since_last,
        )
    }
}

/// Updates the smoothed FPS estimate from the current instantaneous FPS.
pub(crate) fn next_smoothed_fps(prev: Option<f32>, instant_fps: f32, alpha: f32) -> f32 {
    match prev {
        None => instant_fps,
        Some(s) => alpha.mul_add(instant_fps, (1.0 - alpha) * s),
    }
}

/// Builds a [`PerformanceState`] for this frame.
///
/// Returns [`None`] only on the very first tick (`wall_interval_us == 0`), when no
/// frame-to-frame interval has been measured yet and `immediate_fps` has no defined value.
/// All subsequent ticks return [`Some`], so the host-side `PerformanceMetrics` updates every frame.
///
/// `last_frame_render_time_seconds` should be the value returned by
/// [`crate::gpu::GpuContext::last_completed_gpu_render_time_seconds`] mapped through
/// `unwrap_or(`[`RENDER_TIME_UNAVAILABLE`]`)`.
///
/// `rendered_frames_since_last` is the renderer-tick count since the previous `FrameStartData`
/// send (the caller should snapshot then reset its counter for the new send window).
pub(crate) fn step_frame_performance(
    wall_interval_us: u64,
    last_frame_render_time_seconds: f32,
    smoothed_fps: &mut Option<f32>,
    rendered_frames_since_last: i32,
) -> Option<PerformanceState> {
    if wall_interval_us == 0 {
        return None;
    }
    let instant_fps = 1_000_000.0 / wall_interval_us as f32;
    let next_smoothed = next_smoothed_fps(*smoothed_fps, instant_fps, FPS_EMA_ALPHA);
    *smoothed_fps = Some(next_smoothed);
    Some(PerformanceState {
        fps: next_smoothed,
        immediate_fps: instant_fps,
        render_time: last_frame_render_time_seconds,
        rendered_frames_since_last,
        ..PerformanceState::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_smoothed_fps_starts_at_instant() {
        assert!((next_smoothed_fps(None, 60.0, 0.1) - 60.0).abs() < f32::EPSILON);
    }

    #[test]
    fn next_smoothed_fps_ema_blends() {
        let blended = next_smoothed_fps(Some(60.0), 120.0, 0.1);
        assert!((blended - 66.0).abs() < 0.01);
    }

    #[test]
    fn step_frame_performance_first_tick_with_zero_interval_returns_none() {
        let mut smoothed = None;
        let p = step_frame_performance(0, 0.005, &mut smoothed, 0);
        assert!(p.is_none());
        assert!(smoothed.is_none());
    }

    #[test]
    fn step_frame_performance_emits_immediate_smoothed_and_render_time() {
        let mut smoothed = None;
        let p = step_frame_performance(16_666, 0.005, &mut smoothed, 1)
            .expect("payload built when wall_interval_us > 0");
        assert!((p.immediate_fps - 60.0).abs() < 1.0);
        assert!((p.fps - p.immediate_fps).abs() < f32::EPSILON);
        assert!((p.render_time - 0.005).abs() < f32::EPSILON);
        assert!(smoothed.is_some());
    }

    #[test]
    fn step_frame_performance_emits_every_consecutive_call() {
        let mut smoothed = None;
        let a = step_frame_performance(16_666, 0.005, &mut smoothed, 1);
        let b = step_frame_performance(16_666, 0.005, &mut smoothed, 1);
        assert!(a.is_some(), "first non-zero interval must emit");
        assert!(b.is_some(), "subsequent ticks must emit (no throttle)");
    }

    #[test]
    fn step_frame_performance_propagates_render_time_unavailable_sentinel() {
        let mut smoothed = None;
        let p = step_frame_performance(16_666, RENDER_TIME_UNAVAILABLE, &mut smoothed, 0)
            .expect("payload built");
        assert_eq!(p.render_time, RENDER_TIME_UNAVAILABLE);
    }

    #[test]
    fn step_frame_performance_propagates_rendered_frames_since_last() {
        let mut smoothed = None;
        let lockstep = step_frame_performance(16_666, 0.005, &mut smoothed, 1)
            .expect("lockstep payload built");
        assert_eq!(lockstep.rendered_frames_since_last, 1);
        let decoupled = step_frame_performance(16_666, 0.005, &mut smoothed, 7)
            .expect("decoupled payload built");
        assert_eq!(decoupled.rendered_frames_since_last, 7);
    }
}
