//! Lock-step between host [`crate::shared::FrameSubmitData`] and outgoing [`crate::shared::FrameStartData`].
//!
//! The canonical `last_frame_index` echoed to the host lives on [`crate::frontend::RendererFrontend`]
//! and is updated from [`crate::runtime::frame_submit::process_frame_submit`] via
//! [`RendererFrontend::note_frame_submit_processed`](crate::frontend::RendererFrontend::note_frame_submit_processed).

/// Optional trace when the submitted frame index repeats (may be benign during stress tests).
pub(crate) fn trace_duplicate_frame_index_if_interesting(
    submitted: i32,
    previous_host_camera_frame_index: i32,
) {
    if submitted == previous_host_camera_frame_index && submitted >= 0 {
        logger::trace!(
            "lockstep: frame_submit frame_index={submitted} matches previous host_camera.frame_index (duplicate submit?)"
        );
    }
}
