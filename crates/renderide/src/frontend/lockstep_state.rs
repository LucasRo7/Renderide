//! Lock-step frame cadence state and pending begin-frame payload queues.

use crate::shared::{
    FrameStartData, InputState, PerformanceState, ReflectionProbeChangeRenderResult,
    VideoTextureClockErrorState,
};

use super::begin_frame::{
    BeginFrameBuildInput, BeginFrameCommit, BeginFrameDecision, BeginFrameGateInput,
    build_frame_start, decide_begin_frame,
};

/// Lock-step state that determines when the renderer may ask the host for another frame.
pub(crate) struct LockstepState {
    last_frame_data_processed: bool,
    last_frame_index: i32,
    sent_bootstrap_frame_start: bool,
    pending_rendered_reflection_probes: Vec<ReflectionProbeChangeRenderResult>,
    pending_video_clock_errors: Vec<VideoTextureClockErrorState>,
}

impl LockstepState {
    /// Builds lock-step state for either standalone or host-connected mode.
    pub(crate) fn new(standalone: bool) -> Self {
        Self {
            last_frame_data_processed: standalone,
            last_frame_index: -1,
            sent_bootstrap_frame_start: false,
            pending_rendered_reflection_probes: Vec::new(),
            pending_video_clock_errors: Vec::new(),
        }
    }

    /// Current host frame index echoed in outgoing frame-start data.
    pub(crate) fn last_frame_index(&self) -> i32 {
        self.last_frame_index
    }

    /// Whether a host frame submit was applied since the last begin-frame send.
    pub(crate) fn last_frame_data_processed(&self) -> bool {
        self.last_frame_data_processed
    }

    /// Whether a frame submit is currently awaited.
    pub(crate) fn awaiting_submit(&self) -> bool {
        !self.last_frame_data_processed
    }

    /// Marks that init data arrived and the initial begin-frame may be sent after finalization.
    pub(crate) fn mark_init_received(&mut self) {
        self.last_frame_data_processed = true;
    }

    /// Updates lock-step state after applying a frame submit.
    pub(crate) fn note_frame_submit_processed(&mut self, frame_index: i32) {
        self.last_frame_index = frame_index;
        self.last_frame_data_processed = true;
    }

    /// Appends reflection-probe render completions for the next outgoing frame-start.
    pub(crate) fn enqueue_rendered_reflection_probes(
        &mut self,
        probes: impl IntoIterator<Item = ReflectionProbeChangeRenderResult>,
    ) {
        self.pending_rendered_reflection_probes.extend(probes);
    }

    /// Appends video texture clock-error samples for the next outgoing frame-start.
    pub(crate) fn enqueue_video_clock_errors(
        &mut self,
        errors: impl IntoIterator<Item = VideoTextureClockErrorState>,
    ) {
        self.pending_video_clock_errors.extend(errors);
    }

    /// Computes whether a begin-frame send is allowed this tick.
    pub(crate) fn begin_frame_decision(
        &self,
        init_finalized: bool,
        fatal_error: bool,
        ipc_connected: bool,
    ) -> BeginFrameDecision {
        decide_begin_frame(BeginFrameGateInput {
            init_finalized,
            fatal_error,
            ipc_connected,
            last_frame_data_processed: self.last_frame_data_processed,
            last_frame_index: self.last_frame_index,
            sent_bootstrap_frame_start: self.sent_bootstrap_frame_start,
        })
    }

    /// Builds outgoing frame-start data plus the commit that should be applied after a successful send.
    pub(crate) fn build_frame_start(
        &self,
        inputs: InputState,
        performance: Option<PerformanceState>,
    ) -> (FrameStartData, BeginFrameCommit) {
        build_frame_start(BeginFrameBuildInput {
            last_frame_index: self.last_frame_index,
            sent_bootstrap_frame_start: self.sent_bootstrap_frame_start,
            performance,
            inputs,
            rendered_reflection_probes: self.pending_rendered_reflection_probes.clone(),
            video_clock_errors: self.pending_video_clock_errors.clone(),
        })
    }

    /// Applies the lock-step commit after a frame-start send has succeeded.
    pub(crate) fn commit_begin_frame_sent(&mut self, commit: BeginFrameCommit) {
        self.pending_rendered_reflection_probes.clear();
        self.pending_video_clock_errors.clear();
        self.last_frame_data_processed = false;
        if commit.mark_bootstrap_sent {
            self.sent_bootstrap_frame_start = true;
        }
    }
}
