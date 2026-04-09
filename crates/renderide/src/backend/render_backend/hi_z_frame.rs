//! Temporal Hi-Z: GPU pyramid, staging readback, and [`HiZTemporalState`] for draw collection.

use std::collections::HashMap;

use std::time::Duration;

use glam::Mat4;

use crate::gpu::hi_z::HiZGpuResources;
use crate::render_graph::hi_z_occlusion::{hi_z_capture_prev_views, hi_z_decode_pyramid_buffer};
use crate::render_graph::{
    build_world_mesh_cull_proj_params, HostCameraFrame, WorldMeshCullProjParams,
};
use crate::scene::SceneCoordinator;

use super::RenderBackend;

/// `RENDERIDE_HI_Z=0` disables temporal Hi-Z occlusion (frustum-only culling).
pub fn hi_z_culling_enabled() -> bool {
    std::env::var("RENDERIDE_HI_Z")
        .map(|v| v != "0")
        .unwrap_or(true)
}

impl RenderBackend {
    /// Polls GPU readback from the previous frame; updates [`crate::render_graph::hi_z_occlusion::HiZTemporalState::snapshot`].
    ///
    /// Call once per frame before mesh draw collection (see [`crate::render_graph::CompiledRenderGraph::execute_inner`]).
    pub fn hi_z_begin_frame(&mut self, device: &wgpu::Device) {
        if !hi_z_culling_enabled() {
            return;
        }
        let Some(gpu) = self.hi_z_gpu.as_ref() else {
            return;
        };
        if !self.hi_z_staging_pending_map {
            return;
        }
        let staging = gpu.staging_buffer();
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        });
        let view = staging.slice(..).get_mapped_range();
        let (bw, bh) = gpu.base_dims();
        let floats: &[f32] = bytemuck::cast_slice(&view);
        let snapshot = hi_z_decode_pyramid_buffer(floats, bw, bh).unwrap_or_default();
        drop(view);
        staging.unmap();

        self.hi_z_staging_pending_map = false;
        let Some(ref mut temporal) = self.hi_z_temporal else {
            return;
        };
        temporal.snapshot = snapshot;
    }

    /// Temporal state for [`crate::render_graph::collect_and_sort_world_mesh_draws`] (previous frame).
    pub fn hi_z_temporal_for_cull(&self) -> Option<&crate::render_graph::HiZTemporalState> {
        self.hi_z_temporal.as_ref()
    }

    /// Initializes GPU resources and temporal state after GPU attach.
    pub(crate) fn attach_gpu_hi_z(&mut self, device: &wgpu::Device) {
        if !hi_z_culling_enabled() {
            return;
        }
        self.hi_z_gpu = Some(HiZGpuResources::new(device));
        self.hi_z_temporal = Some(crate::render_graph::HiZTemporalState {
            snapshot: crate::render_graph::HiZCpuSnapshot::default(),
            prev_cull: WorldMeshCullProjParams {
                world_proj: Mat4::IDENTITY,
                overlay_proj: Mat4::IDENTITY,
                vr_stereo: None,
            },
            prev_view_by_space: HashMap::new(),
            depth_viewport_px: (1, 1),
        });
    }

    /// Encodes Hi-Z build + staging copy and records view–projection for the next frame.
    ///
    /// Skips when multiview stereo depth is active (single-layer Hi-Z not built yet).
    #[allow(clippy::too_many_arguments)]
    pub fn hi_z_encode_end_of_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        depth_extent: (u32, u32),
        scene: &SceneCoordinator,
        viewport_px: (u32, u32),
        host_camera: &HostCameraFrame,
        multiview_stereo: bool,
    ) {
        if !hi_z_culling_enabled() || multiview_stereo {
            return;
        }
        let Some(gpu) = self.hi_z_gpu.as_mut() else {
            return;
        };
        let Some(ref mut temporal) = self.hi_z_temporal else {
            return;
        };

        let (dw, dh) = depth_extent;
        gpu.ensure_pyramid(device, dw, dh);
        temporal.prev_cull = build_world_mesh_cull_proj_params(scene, viewport_px, host_camera);
        temporal.depth_viewport_px = depth_extent;
        hi_z_capture_prev_views(scene, temporal);

        gpu.encode_build(device, encoder, depth_view, queue);
        gpu.encode_copy_to_staging(encoder);
        self.hi_z_staging_pending_map = true;
    }
}
