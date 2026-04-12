//! Desktop vs OpenXR frame submission helpers for [`super::RenderideApp`].
//!
//! Keeps [`super::RenderideApp::tick_frame`] readable while preserving ordering: OpenXR
//! `wait_frame` / `locate_views` before lock-step [`crate::runtime::RendererRuntime::pre_frame`].

use winit::window::Window;

use crate::gpu::GpuContext;
use crate::render_graph::GraphExecuteError;
use crate::runtime::RendererRuntime;
use crate::xr::{OpenxrFrameTick, XrHostCameraSync, XrStereoSwapchain, XrWgpuHandles};

/// Runs OpenXR `wait_frame` + view pose for stereo uniforms and IPC head tracking.
pub(crate) fn begin_openxr_frame_tick(
    handles: &mut XrWgpuHandles,
    runtime: &mut RendererRuntime,
) -> Option<OpenxrFrameTick> {
    crate::xr::openxr_begin_frame_tick(handles, runtime)
}

/// Renders to the HMD multiview swapchain when VR is active; returns whether a projection layer was submitted.
pub(crate) fn try_hmd_multiview_submit(
    gpu: &mut GpuContext,
    handles: &mut XrWgpuHandles,
    runtime: &mut RendererRuntime,
    xr_swapchain: &mut Option<XrStereoSwapchain>,
    xr_stereo_depth: &mut Option<(wgpu::Texture, wgpu::TextureView)>,
    window: &Window,
    tick: &OpenxrFrameTick,
) -> bool {
    crate::xr::try_openxr_hmd_multiview_submit(
        gpu,
        handles,
        runtime,
        xr_swapchain,
        xr_stereo_depth,
        window,
        tick,
    )
}

/// After HMD work, mirror window uses a single-view matrix when `vr_active`.
pub(crate) fn apply_vr_mirror_stereo_for_desktop_pass(
    runtime: &mut impl XrHostCameraSync,
    xr_tick: Option<&OpenxrFrameTick>,
) {
    if !runtime.vr_active() {
        return;
    }
    let mirror_vp = xr_tick.and_then(|tick| tick.desktop_mirror_view_proj);
    XrHostCameraSync::set_stereo_view_proj(runtime, mirror_vp.map(|vp| (vp, vp)));
    XrHostCameraSync::set_stereo_views(runtime, None);
}

/// Presents the desktop mirror / compositor path.
pub(crate) fn execute_mirror_frame_graph(
    runtime: &mut RendererRuntime,
    gpu: &mut GpuContext,
    window: &Window,
) -> Result<(), GraphExecuteError> {
    runtime.execute_frame_graph(gpu, window)
}
