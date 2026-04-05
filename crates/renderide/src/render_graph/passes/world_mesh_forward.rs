//! Main forward pass: clear color + depth, debug normal shading for scene meshes.
//!
//! Draws are collected and **sorted by [`MaterialDrawBatchKey`](crate::render_graph::MaterialDrawBatchKey)**
//! so pipeline and bind-group changes happen only when material / property-block / skinned path changes.
//! Per-slot [`MaterialPropertyLookupIds`](crate::assets::material::MaterialPropertyLookupIds) are carried on each
//! [`WorldMeshDrawItem`](crate::render_graph::WorldMeshDrawItem) for upcoming per-material bind groups (`get_merged`).

use glam::Mat4;

use crate::assets::material::MaterialDictionary;
use crate::materials::MaterialPipelineDesc;
use crate::pipelines::ShaderPermutation;
use crate::present::SWAPCHAIN_CLEAR_COLOR;
use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::{PassResources, ResourceSlot};
use crate::render_graph::{
    collect_and_sort_world_mesh_draws, MaterialDrawBatchKey, WorldMeshDrawItem,
};
use crate::scene::SceneCoordinator;
use crate::scene::{render_transform_to_matrix, RenderSpaceId};

/// Clears the backbuffer and depth, then draws meshes with material-batched raster pipelines.
#[derive(Debug, Default)]
pub struct WorldMeshForwardPass;

impl WorldMeshForwardPass {
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
            reads: Vec::new(),
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

        let desc = MaterialPipelineDesc {
            surface_format: frame.surface_format,
            depth_stencil_format: Some(wgpu::TextureFormat::Depth32Float),
            sample_count: 1,
        };
        let backend = &mut frame.backend;
        let store_ref = backend.material_property_store();
        let dict = MaterialDictionary::new(store_ref);
        let mesh_pool = &backend.mesh_pool;
        let draws = collect_and_sort_world_mesh_draws(frame.scene, mesh_pool, &dict);
        if draws.is_empty() {
            return Ok(());
        }
        let Some(reg) = backend.material_registry.as_mut() else {
            return Ok(());
        };
        let Some(dbg) = backend.debug_draw.as_ref() else {
            return Ok(());
        };
        let debug_bind_group = &dbg.bind_group;
        let globals_buf = &dbg.globals_buffer;
        let model_buf = &dbg.model_buffer;

        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("world-mesh-forward"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: bb,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(SWAPCHAIN_CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        let queue = match ctx.queue.lock() {
            Ok(q) => q,
            Err(poisoned) => poisoned.into_inner(),
        };

        let (vw, vh) = frame.viewport_px;
        let aspect = vw as f32 / vh.max(1) as f32;
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.05, 10_000.0);

        let mut last_space: Option<RenderSpaceId> = None;
        let mut last_batch_key: Option<MaterialDrawBatchKey> = None;
        let mut pipeline_ok = false;

        for item in &draws {
            if last_space != Some(item.space_id) {
                Self::write_view_proj_for_space(
                    frame.scene,
                    item.space_id,
                    &proj,
                    globals_buf,
                    &queue,
                );
                last_space = Some(item.space_id);
            }

            if last_batch_key.as_ref() != Some(&item.batch_key) {
                last_batch_key = Some(item.batch_key);
                match reg.pipeline_for_family(item.batch_key.family_id, &desc, ShaderPermutation(0))
                {
                    Some(pipeline) => {
                        rpass.set_pipeline(pipeline);
                        rpass.set_bind_group(0, debug_bind_group, &[]);
                        pipeline_ok = true;
                    }
                    None => {
                        logger::trace!(
                            "WorldMeshForward: no pipeline for family {:?}, skipping draws until registered",
                            item.batch_key.family_id
                        );
                        pipeline_ok = false;
                    }
                }
            }

            if !pipeline_ok {
                continue;
            }

            // `lookup_ids` feeds future per-material bind groups (`MaterialDictionary::get_merged`).
            let _ = &item.lookup_ids;

            draw_mesh_submesh(&mut rpass, &queue, model_buf, frame.scene, item, mesh_pool);
        }

        Ok(())
    }
}

impl WorldMeshForwardPass {
    fn write_view_proj_for_space(
        scene: &SceneCoordinator,
        space_id: RenderSpaceId,
        proj: &Mat4,
        globals_buf: &wgpu::Buffer,
        queue: &wgpu::Queue,
    ) {
        let Some(space) = scene.space(space_id) else {
            return;
        };
        let cam = render_transform_to_matrix(&space.view_transform);
        let view = cam.inverse();
        let vp = proj * view;
        let vp_bytes: [f32; 16] = vp.to_cols_array();
        queue.write_buffer(globals_buf, 0, bytemuck::cast_slice(&vp_bytes));
    }
}

fn draw_mesh_submesh(
    rpass: &mut wgpu::RenderPass<'_>,
    queue: &wgpu::Queue,
    model_buf: &wgpu::Buffer,
    scene: &SceneCoordinator,
    item: &WorldMeshDrawItem,
    mesh_pool: &crate::resources::MeshPool,
) {
    if item.mesh_asset_id < 0 || item.node_id < 0 || item.index_count == 0 {
        return;
    }
    let Some(mesh) = mesh_pool.get_mesh(item.mesh_asset_id) else {
        return;
    };
    if !mesh.debug_streams_ready() {
        return;
    }
    let Some(normals) = mesh.normals_buffer.as_deref() else {
        return;
    };

    let use_deformed = item.skinned && mesh.has_skeleton;
    let use_blend_only = mesh.num_blendshapes > 0;

    let pos_buf = if use_deformed {
        mesh.deformed_positions_buffer.as_deref()
    } else if use_blend_only {
        mesh.deform_temp_buffer.as_deref()
    } else {
        mesh.positions_buffer.as_deref()
    };
    let Some(pos) = pos_buf else {
        return;
    };

    let node_u = item.node_id as usize;
    let model = scene
        .world_matrix_with_root(item.space_id, node_u)
        .unwrap_or(Mat4::IDENTITY);
    let m_bytes: [f32; 16] = model.to_cols_array();
    queue.write_buffer(model_buf, 0, bytemuck::cast_slice(&m_bytes));

    rpass.set_vertex_buffer(0, pos.slice(..));
    rpass.set_vertex_buffer(1, normals.slice(..));
    rpass.set_index_buffer(mesh.index_buffer.slice(..), mesh.index_format);

    let first = item.first_index;
    let end = first.saturating_add(item.index_count);
    rpass.draw_indexed(first..end, 0, 0..1);
}
