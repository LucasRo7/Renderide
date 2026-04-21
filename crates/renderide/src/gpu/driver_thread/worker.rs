//! Driver thread loop: drains [`super::ring::BoundedRing`] and runs one GPU frame per message.
//!
//! The loop is the only place in the renderer that calls [`wgpu::Queue::submit`] or
//! [`wgpu::SurfaceTexture::present`] for the main render-graph path. Errors are captured
//! into [`super::DriverErrorState`] and surfaced to the main thread at the next
//! [`super::DriverThread::take_pending_error`] call.

use std::sync::Arc;

use super::error::DriverErrorState;
use super::ring::BoundedRing;
use super::submit_batch::{DriverMessage, SubmitBatch};

/// Thread entry point spawned from [`super::DriverThread::new`].
///
/// Registers itself as `"renderer-driver"` in the active profiler so Tracy groups its
/// spans on a single thread row. Exits on the [`DriverMessage::Shutdown`] sentinel.
pub(super) fn driver_loop(
    ring: Arc<BoundedRing<DriverMessage>>,
    queue: Arc<wgpu::Queue>,
    errors: Arc<DriverErrorState>,
) {
    profiling::register_thread!("renderer-driver");

    while let DriverMessage::Submit(batch) = ring.pop() {
        process_batch(queue.as_ref(), &errors, batch);
    }
    // A `DriverMessage::Shutdown` value breaks the `while let` above; nothing further to do.
}

/// Handles one batch end-to-end: submit, install frame-timing callback, present, signal
/// the oneshot. Each step is instrumented for Tracy.
fn process_batch(queue: &wgpu::Queue, errors: &DriverErrorState, batch: SubmitBatch) {
    profiling::scope!("driver::frame");
    let SubmitBatch {
        command_buffers,
        surface_texture,
        on_submitted_work_done,
        wait,
        frame_seq,
    } = batch;

    {
        profiling::scope!("driver::submit");
        queue.submit(command_buffers);
    }

    if let Some(cb) = on_submitted_work_done {
        queue.on_submitted_work_done(cb);
    }

    if let Some(tex) = surface_texture {
        profiling::scope!("driver::present");
        // `SurfaceTexture::present` is infallible in the current wgpu API; if that
        // changes, route the error into `errors` with `DriverErrorKind::Present`.
        tex.present();
    }

    if let Some(wait) = wait {
        wait.signal();
    }

    // `frame_seq` is carried for future error-context enrichment; reference it so the
    // compiler does not warn about unused fields while we grow the error path.
    let _ = frame_seq;
    let _ = errors; // `errors` will fill in once wgpu surfaces fallible submit/present.
}
