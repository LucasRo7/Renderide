//! Scene walks that collect live material/property-block keys for the world-mesh cache.

use crate::scene::{MeshMaterialSlot, RenderSpaceId, SceneCoordinator, StaticMeshRenderer};

/// Walks one render space's renderer lists and collects every referenced
/// `(material_asset_id, property_block_id)` key. Pure, so callers can run it in parallel across
/// spaces before serial cache updates.
pub(super) fn collect_material_keys_into(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    out: &mut Vec<(i32, Option<i32>)>,
) {
    let Some(space) = scene.space(space_id) else {
        return;
    };
    for r in &space.static_mesh_renderers {
        if r.mesh_asset_id >= 0 {
            append_renderer_material_keys(r, out);
        }
    }
    for sk in &space.skinned_mesh_renderers {
        if sk.base.mesh_asset_id >= 0 {
            append_renderer_material_keys(&sk.base, out);
        }
    }
}

/// Owning variant of [`collect_material_keys_into`] used by the single-space steady-state path.
pub(super) fn collect_material_keys_for_space(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
) -> Vec<(i32, Option<i32>)> {
    let mut out = Vec::new();
    collect_material_keys_into(scene, space_id, &mut out);
    out
}

/// Appends one renderer's `(material_asset_id, property_block_id)` slot keys to `out`.
fn append_renderer_material_keys(r: &StaticMeshRenderer, out: &mut Vec<(i32, Option<i32>)>) {
    let fallback_slot;
    let slots: &[MeshMaterialSlot] = if !r.material_slots.is_empty() {
        &r.material_slots
    } else if let Some(mat_id) = r.primary_material_asset_id {
        fallback_slot = MeshMaterialSlot {
            material_asset_id: mat_id,
            property_block_id: r.primary_property_block_id,
        };
        std::slice::from_ref(&fallback_slot)
    } else {
        return;
    };
    for slot in slots {
        if slot.material_asset_id < 0 {
            continue;
        }
        out.push((slot.material_asset_id, slot.property_block_id));
    }
}
