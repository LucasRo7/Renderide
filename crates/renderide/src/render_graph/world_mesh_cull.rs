//! View–projection parameters for CPU frustum culling of world mesh draws.
//!
//! Values match [`super::passes::world_mesh_forward::WorldMeshForwardPass`] per-space `view` and
//! global projection state (`HostCameraFrame`, viewport aspect, clip planes).

use glam::Mat4;

use crate::scene::SceneCoordinator;

use super::camera::{
    clamp_desktop_fov_degrees, effective_head_output_clip_planes, reverse_z_orthographic,
    reverse_z_perspective,
};
use super::frame_params::HostCameraFrame;

/// Host camera + projection bundle for [`super::world_mesh_draw_prep::collect_and_sort_world_mesh_draws`].
pub struct WorldMeshCullInput<'a> {
    /// Shared reverse-Z projection state for the frame.
    pub proj: WorldMeshCullProjParams,
    /// Per-frame head and clip data (bone palette and overlay projection parity).
    pub host_camera: &'a HostCameraFrame,
}

/// Projection matrices shared by all render spaces for a frame (before multiplying per-space `view`).
#[derive(Clone, Copy, Debug)]
pub struct WorldMeshCullProjParams {
    /// Reverse-Z perspective for the main desktop / non-stereo path.
    pub world_proj: Mat4,
    /// Orthographic overlay projection (same choice as forward pass when overlay draws exist).
    pub overlay_proj: Mat4,
    /// OpenXR per-eye view–projection when VR is active; `None` when not using stereo culling.
    pub vr_stereo: Option<(Mat4, Mat4)>,
}

/// Builds [`WorldMeshCullProjParams`] from viewport size and [`HostCameraFrame`].
pub fn build_world_mesh_cull_proj_params(
    scene: &SceneCoordinator,
    viewport_px: (u32, u32),
    hc: &HostCameraFrame,
) -> WorldMeshCullProjParams {
    let (vw, vh) = viewport_px;
    let aspect = vw as f32 / vh.max(1) as f32;
    let (near, far) = effective_head_output_clip_planes(
        hc.near_clip,
        hc.far_clip,
        hc.output_device,
        scene
            .active_main_space()
            .map(|space| space.root_transform.scale),
    );
    let fov_rad = clamp_desktop_fov_degrees(hc.desktop_fov_degrees).to_radians();
    let world_proj = reverse_z_perspective(aspect, fov_rad, near, far);

    let overlay_proj = if let Some((half_h, on, of)) = hc.primary_ortho_task {
        reverse_z_orthographic(half_h * aspect, half_h, on, of)
    } else {
        reverse_z_orthographic(1.0 * aspect, 1.0, near, far)
    };

    let vr_stereo = match (hc.vr_active, hc.stereo_view_proj) {
        (true, Some(pair)) => Some(pair),
        _ => None,
    };

    WorldMeshCullProjParams {
        world_proj,
        overlay_proj,
        vr_stereo,
    }
}
