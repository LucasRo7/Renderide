//! Narrow traits so OpenXR integration does not depend on the full [`crate::runtime::RendererRuntime`] surface.
//!
//! Implementations live on [`crate::runtime::RendererRuntime`] in [`crate::runtime`].

use glam::{Mat4, Quat, Vec3};

use crate::gpu::GpuContext;
use crate::render_graph::{ExternalFrameTargets, GraphExecuteError};
use crate::shared::HeadOutputDevice;
use winit::window::Window;

/// Read/write hooks for per-eye matrices and head-output positioning used by OpenXR frame ticks.
pub trait XrHostCameraSync {
    fn near_clip(&self) -> f32;
    fn far_clip(&self) -> f32;
    fn output_device(&self) -> HeadOutputDevice;
    fn vr_active(&self) -> bool;
    /// Active main space root scale for [`crate::render_graph::camera::effective_head_output_clip_planes`].
    fn scene_root_scale_for_clip(&self) -> Option<Vec3>;
    /// Same rig alignment as [`crate::xr::tracking_space_to_world_matrix`].
    fn world_from_tracking(&self, center_pose_tracking: Option<(Vec3, Quat)>) -> Mat4;
    fn set_head_output_transform(&mut self, transform: Mat4);
    fn set_stereo_view_proj(&mut self, vp: Option<(Mat4, Mat4)>);
    /// Per-eye **view-only** matrices (world-to-view, handedness-fixed) for stereo clustering.
    fn set_stereo_views(&mut self, views: Option<(Mat4, Mat4)>);
}

/// Multiview submission path that reuses the render graph with external stereo targets.
pub trait XrMultiviewFrameRenderer: XrHostCameraSync {
    /// Renders to OpenXR array color / depth ([`RenderBackend::execute_frame_graph_external_multiview`](crate::backend::RenderBackend::execute_frame_graph_external_multiview)).
    fn execute_frame_graph_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        external: ExternalFrameTargets<'_>,
    ) -> Result<(), GraphExecuteError>;
}
