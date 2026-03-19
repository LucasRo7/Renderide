//! Mesh render pass: draws non-overlay meshes from draw batches.
//!
//! Uses `LoadOp::Clear` for color and depth; draws all non-overlay batches (skinned and
//! non-skinned). Batches are pre-sorted (non-overlay first, then overlay) by the session.

use super::mesh_draw::{self, MeshDrawParams, record_non_skinned_draws, record_skinned_draws};
use super::{RenderPass, RenderPassContext, RenderPassError};
use crate::render::batch::SpaceDrawBatch;
use crate::session::Session;

/// Returns the view (camera) position for the PBR scene, using the same space selection as
/// [`super::ClusteredLightPass`]: primary_view_space_id or the first non-overlay batch's space_id,
/// then the batch matching that space. Keeps view_position, cluster culling, and lights aligned.
fn pbr_view_position_for_space(draw_batches: &[SpaceDrawBatch], session: &Session) -> [f32; 3] {
    let space_id = session.primary_view_space_id().or_else(|| {
        draw_batches
            .iter()
            .find(|b| !b.is_overlay)
            .map(|b| b.space_id)
    });
    let batch = space_id
        .and_then(|sid| {
            draw_batches
                .iter()
                .find(|b| !b.is_overlay && b.space_id == sid)
        })
        .or_else(|| draw_batches.iter().find(|b| !b.is_overlay));
    batch
        .map(|b| {
            let m = crate::scene::render_transform_to_matrix(&b.view_transform);
            [m.w_axis.x, m.w_axis.y, m.w_axis.z]
        })
        .unwrap_or([0.0, 0.0, 0.0])
}

/// Mesh render pass: draws non-overlay meshes from draw batches.
pub struct MeshRenderPass;

impl MeshRenderPass {
    /// Creates a new mesh render pass.
    pub fn new() -> Self {
        Self
    }
}

impl Default for MeshRenderPass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderPass for MeshRenderPass {
    fn name(&self) -> &str {
        "mesh"
    }

    fn execute(&mut self, ctx: &mut RenderPassContext) -> Result<(), RenderPassError> {
        let (non_overlay_skinned, _, non_overlay_non_skinned, _) =
            ctx.cached_mesh_draws
                .as_ref()
                .ok_or(RenderPassError::MissingCachedMeshDraws)?;

        let use_mrt = ctx.render_target.mrt_position_view.is_some()
            && ctx.render_target.mrt_normal_view.is_some();

        let light_buffer_version = ctx.gpu.light_buffer_cache.version;
        let cluster_buffer_version = ctx.gpu.cluster_buffer_cache.version;

        let cluster_buffers = ctx
            .gpu
            .cluster_buffer_cache
            .get_buffers(ctx.viewport, ctx.gpu.cluster_count_z);
        let light_buffer = ctx
            .gpu
            .light_buffer_cache
            .ensure_buffer(&ctx.gpu.device, ctx.gpu.light_count.max(1) as usize);
        let cluster_counts_ok = ctx.gpu.cluster_count_x > 0
            && ctx.gpu.cluster_count_y > 0
            && ctx.gpu.cluster_count_z > 0;
        let cluster_buffers_is_none = cluster_buffers.is_none();
        let light_buffer_is_none = light_buffer.is_none();

        let pbr_scene = match (cluster_buffers, light_buffer, cluster_counts_ok) {
            (Some(crefs), Some(lb), true) => {
                let view_position = pbr_view_position_for_space(ctx.draw_batches, ctx.session);
                Some(mesh_draw::PbrSceneParams {
                    view_position,
                    cluster_count_x: ctx.gpu.cluster_count_x,
                    cluster_count_y: ctx.gpu.cluster_count_y,
                    cluster_count_z: ctx.gpu.cluster_count_z,
                    near_clip: ctx.session.near_clip().max(0.01),
                    far_clip: ctx.session.far_clip(),
                    light_count: ctx.gpu.light_count,
                    light_buffer: lb,
                    cluster_light_counts: crefs.cluster_light_counts,
                    cluster_light_indices: crefs.cluster_light_indices,
                })
            }
            _ => {
                if ctx.session.render_config().use_pbr {
                    let reason = if !cluster_counts_ok {
                        format!(
                            "cluster_count_* zero (x={} y={} z={}) - clustered_light did not run this frame",
                            ctx.gpu.cluster_count_x,
                            ctx.gpu.cluster_count_y,
                            ctx.gpu.cluster_count_z
                        )
                    } else if cluster_buffers_is_none {
                        "cluster get_buffers returned None (viewport or cluster_count_z mismatch)"
                            .to_string()
                    } else if light_buffer_is_none {
                        "light buffer ensure_buffer returned None".to_string()
                    } else {
                        "unknown".to_string()
                    };
                    logger::debug!("PBR scene disabled (no clustered lighting): {}", reason);
                }
                None
            }
        };

        let mut draw_params = MeshDrawParams {
            pipeline_manager: ctx.pipeline_manager,
            device: &ctx.gpu.device,
            queue: &ctx.gpu.queue,
            config: &ctx.gpu.config,
            frame_index: ctx.frame_index,
            mesh_buffer_cache: &ctx.gpu.mesh_buffer_cache,
            skinned_bind_group_cache: &mut ctx.gpu.skinned_bind_group_cache,
            overlay_orthographic: false,
            use_mrt,
            use_pbr: ctx.session.render_config().use_pbr,
            pbr_scene,
            pbr_scene_bind_group_cache: &mut ctx.gpu.pbr_scene_bind_group_cache,
            last_pbr_scene_cache_light_version: &mut ctx.gpu.last_pbr_scene_cache_light_version,
            last_pbr_scene_cache_cluster_version: &mut ctx.gpu.last_pbr_scene_cache_cluster_version,
            light_buffer_version,
            cluster_buffer_version,
        };

        let timestamp_writes =
            ctx.timestamp_query_set
                .map(|query_set| wgpu::RenderPassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                });

        let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = if use_mrt {
            let pos_view = ctx
                .render_target
                .mrt_position_view
                .ok_or(RenderPassError::MissingMrtViews)?;
            let norm_view = ctx
                .render_target
                .mrt_normal_view
                .ok_or(RenderPassError::MissingMrtViews)?;
            vec![
                Some(wgpu::RenderPassColorAttachment {
                    view: ctx.render_target.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.8,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: pos_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: norm_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ]
        } else {
            vec![Some(wgpu::RenderPassColorAttachment {
                view: ctx.render_target.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.8,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })]
        };

        // Non-overlay pass: Clear framebuffer, draw all non-overlay batches.
        {
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mesh pass (non-overlay)"),
                timestamp_writes: timestamp_writes.clone(),
                color_attachments: &color_attachments,
                depth_stencil_attachment: ctx.render_target.depth_view.map(|dv| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view: dv,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0),
                            store: wgpu::StoreOp::Store,
                        }),
                    }
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let debug_blendshapes = ctx.session.render_config().debug_blendshapes;
            record_skinned_draws(
                &mut pass,
                &mut draw_params,
                non_overlay_skinned,
                debug_blendshapes,
            );
            record_non_skinned_draws(&mut pass, &mut draw_params, non_overlay_non_skinned);
        }

        Ok(())
    }
}
