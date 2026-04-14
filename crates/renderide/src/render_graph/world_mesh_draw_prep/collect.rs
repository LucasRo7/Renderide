//! Scene walk that pairs material slots with submeshes and applies optional CPU culling.

use std::collections::HashSet;

use glam::Mat4;

use crate::assets::material::{MaterialDictionary, MaterialPropertyLookupIds};
use crate::assets::mesh::GpuMesh;
use crate::materials::MaterialRouter;
use crate::pipelines::ShaderPermutation;
use crate::resources::MeshPool;
use crate::scene::{RenderSpaceId, SceneCoordinator, SkinnedMeshRenderer, StaticMeshRenderer};
use crate::shared::RenderingContext;

use super::sort::{batch_key_for_slot, sort_world_mesh_draws};
use super::types::{
    resolved_material_slots, CameraTransformDrawFilter, WorldMeshDrawCollection, WorldMeshDrawItem,
};

use super::super::world_mesh_cull_eval::{mesh_draw_passes_cpu_cull, CpuCullFailure};

/// Expands one static mesh renderer into draw items (material slots × submeshes).
#[allow(clippy::too_many_arguments)] // Single fan-out site; grouping would obscure the mesh pass.
fn push_draws_for_renderer(
    out: &mut Vec<WorldMeshDrawItem>,
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    renderer: &StaticMeshRenderer,
    renderable_index: usize,
    skinned: bool,
    skinned_renderer: Option<&SkinnedMeshRenderer>,
    mesh: &GpuMesh,
    submeshes: &[(u32, u32)],
    dict: &MaterialDictionary<'_>,
    router: &MaterialRouter,
    shader_perm: ShaderPermutation,
    context: RenderingContext,
    head_output_transform: Mat4,
    mismatch_warned: &mut HashSet<i32>,
    culling: Option<&super::super::world_mesh_cull::WorldMeshCullInput<'_>>,
    cull_stats: &mut (usize, usize, usize),
    transform_filter: Option<&CameraTransformDrawFilter>,
) {
    if let Some(f) = transform_filter {
        if !f.passes(renderer.node_id) {
            return;
        }
    }
    let slots = resolved_material_slots(renderer);
    if slots.is_empty() {
        return;
    }
    let n_sub = submeshes.len();
    let n_slot = slots.len();
    if n_sub != n_slot && mismatch_warned.insert(renderer.mesh_asset_id) {
        logger::trace!(
            "mesh_asset_id={}: material slot count {} != submesh count {} (using first {} pairings only)",
            renderer.mesh_asset_id,
            n_slot,
            n_sub,
            n_sub.min(n_slot),
        );
    }
    let n = n_sub.min(n_slot);
    if n == 0 {
        return;
    }

    let is_overlay = renderer.layer == crate::shared::LayerType::Overlay;

    for slot_index in 0..n {
        let slot = &slots[slot_index];
        let material_asset_id = scene
            .overridden_material_asset_id(space_id, context, skinned, renderable_index, slot_index)
            .unwrap_or(slot.material_asset_id);
        let (first_index, index_count) = submeshes[slot_index];
        if index_count == 0 {
            continue;
        }
        if material_asset_id < 0 {
            continue;
        }
        let rigid_world_matrix = if skinned {
            None
        } else if let Some(c) = culling {
            cull_stats.0 += 1;
            match mesh_draw_passes_cpu_cull(
                scene,
                space_id,
                mesh,
                is_overlay,
                skinned,
                skinned_renderer,
                renderer.node_id,
                c,
                context,
            ) {
                Err(CpuCullFailure::Frustum) => {
                    cull_stats.1 += 1;
                    continue;
                }
                Err(CpuCullFailure::HiZ) => {
                    cull_stats.2 += 1;
                    continue;
                }
                Ok(m) => m,
            }
        } else {
            scene.world_matrix_for_render_context(
                space_id,
                renderer.node_id as usize,
                context,
                head_output_transform,
            )
        };
        let lookup_ids = MaterialPropertyLookupIds {
            material_asset_id,
            mesh_property_block_slot0: slot.property_block_id,
        };
        let batch_key = batch_key_for_slot(
            material_asset_id,
            slot.property_block_id,
            skinned,
            dict,
            router,
            shader_perm,
        );
        out.push(WorldMeshDrawItem {
            space_id,
            node_id: renderer.node_id,
            mesh_asset_id: renderer.mesh_asset_id,
            slot_index,
            first_index,
            index_count,
            is_overlay,
            sorting_order: renderer.sorting_order,
            skinned,
            collect_order: out.len(),
            camera_distance_sq: 0.0,
            lookup_ids,
            batch_key,
            rigid_world_matrix,
        });
    }
}

/// Collects draws from active spaces, then sorts for batching (material / pipeline boundaries).
///
/// When `culling` is [`Some`], instances outside the frustum (and optional Hi-Z) are dropped (see
/// [`mesh_draw_passes_cpu_cull`](super::super::world_mesh_cull_eval::mesh_draw_passes_cpu_cull)).
#[allow(clippy::too_many_arguments)] // Frame-graph entry mirrors host camera + cull snapshot inputs.
pub fn collect_and_sort_world_mesh_draws(
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    dict: &MaterialDictionary<'_>,
    router: &MaterialRouter,
    shader_perm: ShaderPermutation,
    context: RenderingContext,
    head_output_transform: Mat4,
    culling: Option<&super::super::world_mesh_cull::WorldMeshCullInput<'_>>,
    transform_filter: Option<&CameraTransformDrawFilter>,
) -> WorldMeshDrawCollection {
    let mut mismatch_warned = HashSet::new();
    let mut out = Vec::new();
    let mut cull_stats = (0usize, 0usize, 0usize);

    let mut cap_hint = 0usize;
    for space_id in scene.render_space_ids() {
        let Some(space) = scene.space(space_id) else {
            continue;
        };
        if space.is_active {
            cap_hint = cap_hint
                .saturating_add(space.static_mesh_renderers.len())
                .saturating_add(space.skinned_mesh_renderers.len());
        }
    }
    out.reserve(cap_hint.saturating_mul(8));

    for space_id in scene.render_space_ids() {
        let Some(space) = scene.space(space_id) else {
            continue;
        };
        if !space.is_active {
            continue;
        }

        for (renderable_index, r) in space.static_mesh_renderers.iter().enumerate() {
            if r.mesh_asset_id < 0 || r.node_id < 0 {
                continue;
            }
            let Some(mesh) = mesh_pool.get_mesh(r.mesh_asset_id) else {
                continue;
            };
            if mesh.submeshes.is_empty() {
                continue;
            }
            push_draws_for_renderer(
                &mut out,
                scene,
                space_id,
                r,
                renderable_index,
                false,
                None,
                mesh,
                &mesh.submeshes,
                dict,
                router,
                shader_perm,
                context,
                head_output_transform,
                &mut mismatch_warned,
                culling,
                &mut cull_stats,
                transform_filter,
            );
        }
        for (renderable_index, skinned) in space.skinned_mesh_renderers.iter().enumerate() {
            let r = &skinned.base;
            if r.mesh_asset_id < 0 || r.node_id < 0 {
                continue;
            }
            let Some(mesh) = mesh_pool.get_mesh(r.mesh_asset_id) else {
                continue;
            };
            if mesh.submeshes.is_empty() {
                continue;
            }
            push_draws_for_renderer(
                &mut out,
                scene,
                space_id,
                r,
                renderable_index,
                true,
                Some(skinned),
                mesh,
                &mesh.submeshes,
                dict,
                router,
                shader_perm,
                context,
                head_output_transform,
                &mut mismatch_warned,
                culling,
                &mut cull_stats,
                transform_filter,
            );
        }
    }

    sort_world_mesh_draws(&mut out);
    WorldMeshDrawCollection {
        items: out,
        draws_pre_cull: cull_stats.0,
        draws_culled: cull_stats.1,
        draws_hi_z_culled: cull_stats.2,
    }
}
