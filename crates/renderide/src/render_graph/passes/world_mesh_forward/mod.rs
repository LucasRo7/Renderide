//! Main forward pass: clear color + depth, debug normal shading for scene meshes.
//!
//! Draws are collected and **sorted by [`MaterialDrawBatchKey`](crate::render_graph::MaterialDrawBatchKey)**
//! so pipeline and batch key drive pipeline switches. **GPU instancing:** consecutive draws that share the
//! same mesh submesh and batch key (opaque, non-skinned) are merged into one indexed draw with
//! `instance_index` sampling [`crate::backend::PerDrawResources`] (`@group(2)` storage). Embedded `@group(1)`
//! skips redundant [`wgpu::RenderPass::set_bind_group`] when [`MaterialBindCacheKey`](crate::backend::MaterialBindCacheKey) matches
//! the previous draw (uniform updates still run each time via [`EmbeddedMaterialBindResources`](crate::backend::EmbeddedMaterialBindResources)).
//! Per-slot [`MaterialPropertyLookupIds`](crate::assets::material::MaterialPropertyLookupIds) are carried on each
//! [`WorldMeshDrawItem`](crate::render_graph::WorldMeshDrawItem) for `get_merged` when building `@group(1)` bind
//! groups for [`crate::materials::RasterPipelineKind::EmbeddedStem`] draws (see [`crate::backend::EmbeddedMaterialBindResources`]).
//!
//! Manifest raster binds use the composed WGSL **stem** from [`crate::materials::MaterialRouter::stem_for_shader_asset`]
//! (not a hard-coded Unlit path). Whether UV0 is bound is stored on [`MaterialDrawBatchKey::Embedded_needs_uv0`]
//! (same rule as the embedded raster pipeline and [`crate::materials::embedded_stem_needs_uv0_stream`], computed during draw collection).
//! Intersection tint subpasses use [`MaterialDrawBatchKey::Embedded_requires_intersection_pass`]
//! ([`crate::materials::embedded_stem_requires_intersection_pass`], WGSL reflection of `_IntersectColor`).
//!
//! ## VR stereo world draws
//!
//! OpenXR per-eye view–projection maps **stage** space to clip. For **non-overlay** draws with
//! `stereo_view_proj`, we use **identity** instead of the host `view_transform` world-to-camera so
//! `VP` is not `P·V_hmd·V_host`, which mixed stage with the host rig and caused playspace-relative
//! offsets. Overlays keep `view` for orthographic / UI alignment with the host camera rig.
//! Matrix composition lives in [`vp`].

mod encode;
mod execute_helpers;
mod vp;

use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::{PassResources, ResourceSlot};
use crate::render_graph::{build_world_mesh_cull_proj_params, WorldMeshCullInput};

use execute_helpers::{
    capture_hi_z_temporal_after_collect, compute_view_projections, encode_clear_only_pass,
    encode_world_mesh_forward_draw_passes, maybe_set_world_mesh_draw_stats,
    pack_and_upload_per_draw_slab, resolve_pass_config, take_or_collect_world_mesh_draws,
    write_frame_uniforms_and_cluster,
};

/// Clears the backbuffer and depth, then draws meshes with material-batched raster pipelines.
#[derive(Debug, Default)]
pub struct WorldMeshForwardPass;

impl WorldMeshForwardPass {
    /// Creates a world mesh forward pass instance.
    pub fn new() -> Self {
        Self
    }
}

impl RenderPass for WorldMeshForwardPass {
    fn name(&self) -> &str {
        "WorldMeshForward"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: vec![ResourceSlot::ClusterBuffers, ResourceSlot::LightBuffer],
            writes: vec![ResourceSlot::Backbuffer, ResourceSlot::Depth],
        }
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_>) -> Result<(), RenderPassError> {
        let Some(bb) = ctx.backbuffer else {
            return Err(RenderPassError::MissingBackbuffer {
                pass: self.name().to_string(),
            });
        };
        let Some(depth) = ctx.depth_view else {
            return Err(RenderPassError::MissingDepth {
                pass: self.name().to_string(),
            });
        };
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };

        let supports_base_instance = ctx.gpu_limits.supports_base_instance;
        let hc = frame.host_camera;
        let pipeline = resolve_pass_config(
            hc,
            frame.multiview_stereo,
            frame.surface_format,
            ctx.gpu_limits,
        );
        let use_multiview = pipeline.use_multiview;
        let pass_desc = pipeline.pass_desc;
        let shader_perm = pipeline.shader_perm;

        let culling = if hc.suppress_occlusion_temporal {
            None
        } else {
            let cull_proj = build_world_mesh_cull_proj_params(frame.scene, frame.viewport_px, &hc);
            let depth_mode = frame.output_depth_mode();
            let view_id = frame.occlusion_view;
            let hi_z_temporal = frame.backend.occlusion.hi_z_temporal_snapshot(view_id);
            let hi_z = frame.backend.occlusion.hi_z_cull_data(depth_mode, view_id);
            Some(WorldMeshCullInput {
                proj: cull_proj,
                host_camera: &hc,
                hi_z,
                hi_z_temporal,
            })
        };

        let collection = take_or_collect_world_mesh_draws(frame, culling.as_ref(), shader_perm);
        capture_hi_z_temporal_after_collect(frame, culling.as_ref(), hc);

        maybe_set_world_mesh_draw_stats(
            frame.backend,
            &collection,
            &collection.items,
            supports_base_instance,
        );

        let draws = collection.items;

        let scene = frame.scene;
        let (render_context, world_proj, overlay_proj) =
            compute_view_projections(scene, hc, frame.viewport_px, &draws);

        let queue_guard = ctx
            .queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let queue = &*queue_guard;

        if !pack_and_upload_per_draw_slab(
            ctx.device,
            queue,
            frame.backend,
            scene,
            hc,
            render_context,
            world_proj,
            overlay_proj,
            &draws,
        ) {
            return Ok(());
        }

        write_frame_uniforms_and_cluster(
            ctx.device,
            queue,
            frame.backend,
            hc,
            scene,
            frame.viewport_px,
            use_multiview,
        );

        if draws.is_empty() {
            encode_clear_only_pass(ctx.encoder, bb, depth, use_multiview);
            return Ok(());
        }

        if !encode_world_mesh_forward_draw_passes(
            ctx.encoder,
            ctx.device,
            frame,
            queue,
            &draws,
            &pass_desc,
            shader_perm,
            use_multiview,
            supports_base_instance,
            bb,
            depth,
        ) {
            return Ok(());
        }

        Ok(())
    }
}
