//! Command encoding for blendshape and skinning compute dispatches.

use std::num::NonZeroU64;

use glam::Mat4;

use crate::assets::mesh::BLENDSHAPE_OFFSET_GPU_STRIDE;
use crate::backend::advance_slab_cursor;
use crate::backend::mesh_deform::plan_blendshape_bind_chunks;
use crate::gpu::GpuLimits;
use crate::render_graph::skinning_palette::build_skinning_palette;
use crate::scene::RenderSpaceId;

use super::snapshot::{
    deform_needs_blend_snapshot, deform_needs_skin_snapshot, MeshDeformSnapshot,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn record_mesh_deform(
    queue: &wgpu::Queue,
    device: &wgpu::Device,
    gpu_limits: &GpuLimits,
    encoder: &mut wgpu::CommandEncoder,
    pre: &crate::backend::mesh_deform::MeshPreprocessPipelines,
    scratch: &mut crate::backend::MeshDeformScratch,
    scene: &crate::scene::SceneCoordinator,
    space_id: RenderSpaceId,
    mesh: &MeshDeformSnapshot,
    bone_transform_indices: Option<&[i32]>,
    smr_node_id: i32,
    render_context: crate::shared::RenderingContext,
    head_output_transform: Mat4,
    blend_weights: &[f32],
    bone_cursor: &mut u64,
    blend_weight_cursor: &mut u64,
) {
    let Some(ref positions) = mesh.positions_buffer else {
        return;
    };
    let vc = mesh.vertex_count;
    if vc == 0 {
        return;
    }
    let needs_blend = deform_needs_blend_snapshot(mesh);
    let needs_skin = deform_needs_skin_snapshot(mesh, bone_transform_indices);

    if !needs_blend && !needs_skin {
        return;
    }

    let wg = workgroup_count(vc);
    if !gpu_limits.compute_dispatch_fits(wg, 1, 1) {
        logger::warn!(
            "mesh deform: compute dispatch {}×1×1 exceeds max_compute_workgroups_per_dimension ({})",
            wg,
            gpu_limits.max_compute_workgroups_per_dimension()
        );
        return;
    }

    if needs_blend {
        let Some(ref temp) = mesh.deform_temp_buffer else {
            return;
        };
        let Some(ref deltas) = mesh.blendshape_buffer else {
            return;
        };
        let shape_count = mesh.num_blendshapes;
        if shape_count == 0 {
            return;
        }
        scratch.ensure_shape_weight_capacity(device, shape_count);
        let mut wbytes = vec![0u8; (shape_count as usize) * 4];
        for s in 0..shape_count as usize {
            let w = blend_weights.get(s).copied().unwrap_or(0.0);
            wbytes[s * 4..s * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }

        let weight_binding_len = wbytes.len() as u64;
        scratch.ensure_blend_weight_byte_capacity(
            device,
            (*blend_weight_cursor).saturating_add(weight_binding_len),
        );
        queue.write_buffer(&scratch.blendshape_weights, *blend_weight_cursor, &wbytes);

        let Some(chunks) = plan_blendshape_bind_chunks(
            shape_count,
            vc,
            gpu_limits.wgpu.max_storage_buffer_binding_size,
            gpu_limits.wgpu.min_storage_buffer_offset_alignment,
        ) else {
            logger::warn!(
                "mesh deform: blendshape bind chunks unavailable (vertex_count={vc} shape_count={shape_count} max_bind={})",
                gpu_limits.wgpu.max_storage_buffer_binding_size
            );
            return;
        };

        let stride = u64::from(vc) * u64::from(BLENDSHAPE_OFFSET_GPU_STRIDE as u32);

        let mut packed_params = Vec::with_capacity(chunks.len() * 32);
        for (chunk_i, (shape_start, chunk_shapes)) in chunks.iter().enumerate() {
            let params = build_blend_params(vc, *chunk_shapes, *shape_start, chunk_i == 0);
            packed_params.extend_from_slice(&params);
        }
        scratch.ensure_blendshape_params_staging(device, packed_params.len() as u64);
        queue.write_buffer(&scratch.blendshape_params_staging, 0, &packed_params);

        for (chunk_i, (shape_start, chunk_shapes)) in chunks.iter().enumerate() {
            let src_off = (chunk_i as u64).saturating_mul(32);
            encoder.copy_buffer_to_buffer(
                &scratch.blendshape_params_staging,
                src_off,
                &scratch.blendshape_params,
                0,
                32,
            );

            let offset = u64::from(*shape_start).saturating_mul(stride);
            let size = u64::from(*chunk_shapes).saturating_mul(stride);
            let Some(size_nz) = NonZeroU64::new(size) else {
                continue;
            };

            let Some(weight_size) = NonZeroU64::new(weight_binding_len) else {
                continue;
            };

            let blend_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blendshape_bg"),
                layout: &pre.blendshape_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: scratch.blendshape_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: positions.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: deltas.as_ref(),
                            offset,
                            size: Some(size_nz),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &scratch.blendshape_weights,
                            offset: *blend_weight_cursor,
                            size: Some(weight_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: temp.as_entire_binding(),
                    },
                ],
            });

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("blendshape"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pre.blendshape_pipeline);
            cpass.set_bind_group(0, &blend_bg, &[]);
            cpass.dispatch_workgroups(wg, 1, 1);
        }

        *blend_weight_cursor = advance_slab_cursor(*blend_weight_cursor, weight_binding_len);
    }

    if needs_skin {
        let Some(ref dst) = mesh.deformed_positions_buffer else {
            return;
        };
        let Some(ref dst_n) = mesh.deformed_normals_buffer else {
            return;
        };
        let Some(ref src_n) = mesh.normals_buffer else {
            return;
        };
        let Some(ref bone_idx) = mesh.bone_indices_buffer else {
            return;
        };
        let Some(ref bone_wt) = mesh.bone_weights_vec4_buffer else {
            return;
        };
        let Some(indices) = bone_transform_indices else {
            return;
        };

        let bone_count_u = mesh.skinning_bind_matrices.len() as u32;
        scratch.ensure_bone_capacity(device, bone_count_u);
        let Some(palette_mats) = build_skinning_palette(
            scene,
            space_id,
            &mesh.skinning_bind_matrices,
            mesh.has_skeleton,
            indices,
            smr_node_id,
            render_context,
            head_output_transform,
        ) else {
            return;
        };
        let mut palette: Vec<u8> = vec![0u8; palette_mats.len() * 64];
        for (bi, pal) in palette_mats.iter().enumerate() {
            let cols = pal.to_cols_array();
            palette[bi * 64..bi * 64 + 64].copy_from_slice(bytemuck::cast_slice(&cols));
        }

        let palette_len = palette.len() as u64;
        scratch.ensure_bone_byte_capacity(device, bone_cursor.saturating_add(palette_len));
        queue.write_buffer(&scratch.bone_matrices, *bone_cursor, &palette);

        let Some(bone_binding_size) = NonZeroU64::new(palette_len) else {
            return;
        };

        let src_for_skin: &wgpu::Buffer = if needs_blend {
            let Some(buf) = mesh.deform_temp_buffer.as_deref() else {
                return;
            };
            buf
        } else {
            positions.as_ref()
        };

        let skin_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinning_bg"),
            layout: &pre.skinning_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &scratch.bone_matrices,
                        offset: *bone_cursor,
                        size: Some(bone_binding_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_for_skin.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bone_idx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bone_wt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: src_n.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dst_n.as_entire_binding(),
                },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("skinning"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pre.skinning_pipeline);
        cpass.set_bind_group(0, &skin_bg, &[]);
        cpass.dispatch_workgroups(wg, 1, 1);
        drop(cpass);

        *bone_cursor = advance_slab_cursor(*bone_cursor, palette_len);
    }
}

fn workgroup_count(vertex_count: u32) -> u32 {
    (vertex_count.saturating_add(63)) / 64
}

/// Encodes `source/compute/mesh_blendshape.wgsl` `Params` (32 bytes).
fn build_blend_params(
    vertex_count: u32,
    chunk_shape_count: u32,
    weight_base: u32,
    first_chunk: bool,
) -> [u8; 32] {
    let mut o = [0u8; 32];
    o[0..4].copy_from_slice(&vertex_count.to_le_bytes());
    o[4..8].copy_from_slice(&chunk_shape_count.to_le_bytes());
    o[8..12].copy_from_slice(&vertex_count.to_le_bytes());
    o[12..16].copy_from_slice(&weight_base.to_le_bytes());
    let fc = u32::from(first_chunk);
    o[16..20].copy_from_slice(&fc.to_le_bytes());
    o
}
