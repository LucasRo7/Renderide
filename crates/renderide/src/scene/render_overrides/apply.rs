//! Shared-memory apply steps for transform and material override updates.

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{
    MaterialOverrideState, RenderMaterialOverrideState, RenderMaterialOverridesUpdate,
    RenderTransformOverrideState, RenderTransformOverridesUpdate,
    RENDER_MATERIAL_OVERRIDE_STATE_HOST_ROW_BYTES, RENDER_TRANSFORM_OVERRIDE_STATE_HOST_ROW_BYTES,
};

use super::super::error::SceneError;
use super::super::render_space::RenderSpaceState;
use super::super::transforms_apply::TransformRemovalEvent;
use super::super::world::fixup_transform_id;
use super::types::{
    decode_packed_mesh_renderer_target, MaterialOverrideBinding, RenderMaterialOverrideEntry,
    RenderTransformOverrideEntry,
};

pub(crate) fn apply_render_transform_overrides_update(
    space: &mut RenderSpaceState,
    shm: &mut SharedMemoryAccessor,
    update: &RenderTransformOverridesUpdate,
    scene_id: i32,
    transform_removals: &[TransformRemovalEvent],
) -> Result<(), SceneError> {
    fixup_transform_override_nodes_for_transform_removals(space, transform_removals);

    if update.removals.length > 0 {
        let ctx = format!("render transform override removals scene_id={scene_id}");
        let removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        apply_dense_removals(&mut space.render_transform_overrides, removals.as_slice());
    }

    if update.additions.length > 0 {
        let ctx = format!("render transform override additions scene_id={scene_id}");
        let additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        for &node_id in additions.iter().take_while(|&&id| id >= 0) {
            space
                .render_transform_overrides
                .push(RenderTransformOverrideEntry {
                    node_id,
                    ..Default::default()
                });
        }
    }

    if update.states.length > 0 {
        let ctx = format!("render transform override states scene_id={scene_id}");
        let states = shm
            .access_copy_memory_packable_rows::<RenderTransformOverrideState>(
                &update.states,
                RENDER_TRANSFORM_OVERRIDE_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        let skinned_indices = if update.skinned_mesh_renderers_indexes.length > 0 {
            let ctx = format!("render transform override skinned mesh indexes scene_id={scene_id}");
            shm.access_copy_diagnostic_with_context::<i32>(
                &update.skinned_mesh_renderers_indexes,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?
        } else {
            Vec::new()
        };
        let mut skinned_cursor = 0usize;
        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            let idx = state.renderable_index as usize;
            let Some(entry) = space.render_transform_overrides.get_mut(idx) else {
                continue;
            };
            entry.context = state.context;
            entry.position_override =
                ((state.override_flags & 0b001) != 0).then_some(state.position_override);
            entry.rotation_override =
                ((state.override_flags & 0b010) != 0).then_some(state.rotation_override);
            entry.scale_override =
                ((state.override_flags & 0b100) != 0).then_some(state.scale_override);
            let count = state.skinned_mesh_renderer_count.max(0) as usize;
            entry.skinned_mesh_renderer_indices.clear();
            if count > 0 {
                let end = skinned_cursor
                    .saturating_add(count)
                    .min(skinned_indices.len());
                entry
                    .skinned_mesh_renderer_indices
                    .extend_from_slice(&skinned_indices[skinned_cursor..end]);
                skinned_cursor = end;
            }
        }
    }

    Ok(())
}

pub(crate) fn apply_render_material_overrides_update(
    space: &mut RenderSpaceState,
    shm: &mut SharedMemoryAccessor,
    update: &RenderMaterialOverridesUpdate,
    scene_id: i32,
    transform_removals: &[TransformRemovalEvent],
) -> Result<(), SceneError> {
    fixup_material_override_nodes_for_transform_removals(space, transform_removals);

    if update.removals.length > 0 {
        let ctx = format!("render material override removals scene_id={scene_id}");
        let removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        apply_dense_removals(&mut space.render_material_overrides, removals.as_slice());
    }

    if update.additions.length > 0 {
        let ctx = format!("render material override additions scene_id={scene_id}");
        let additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        for &node_id in additions.iter().take_while(|&&id| id >= 0) {
            space
                .render_material_overrides
                .push(RenderMaterialOverrideEntry {
                    node_id,
                    ..Default::default()
                });
        }
    }

    if update.states.length > 0 {
        let ctx = format!("render material override states scene_id={scene_id}");
        let states = shm
            .access_copy_memory_packable_rows::<RenderMaterialOverrideState>(
                &update.states,
                RENDER_MATERIAL_OVERRIDE_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        let materials = if update.material_override_states.length > 0 {
            let ctx = format!("render material override rows scene_id={scene_id}");
            shm.access_copy_diagnostic_with_context::<MaterialOverrideState>(
                &update.material_override_states,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?
        } else {
            Vec::new()
        };
        let mut material_cursor = 0usize;
        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            let idx = state.renderable_index as usize;
            let Some(entry) = space.render_material_overrides.get_mut(idx) else {
                continue;
            };
            entry.context = state.context;
            entry.target = decode_packed_mesh_renderer_target(state.packed_mesh_renderer_index);
            let count = state.materrial_override_count.max(0) as usize;
            entry.material_overrides.clear();
            if count > 0 {
                let end = material_cursor.saturating_add(count).min(materials.len());
                entry
                    .material_overrides
                    .extend(materials[material_cursor..end].iter().map(|row| {
                        MaterialOverrideBinding {
                            material_slot_index: row.material_slot_index,
                            material_asset_id: row.material_asset_id,
                        }
                    }));
                material_cursor = end;
            }
        }
    }

    Ok(())
}

fn apply_dense_removals<T>(entries: &mut Vec<T>, removals: &[i32]) {
    for &raw in removals.iter().take_while(|&&idx| idx >= 0) {
        let idx = raw as usize;
        if idx < entries.len() {
            entries.swap_remove(idx);
        }
    }
}

fn fixup_transform_override_nodes_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    for removal in removals {
        for entry in &mut space.render_transform_overrides {
            entry.node_id = fixup_transform_id(
                entry.node_id,
                removal.removed_index,
                removal.last_index_before_swap,
            );
        }
    }
}

fn fixup_material_override_nodes_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    for removal in removals {
        for entry in &mut space.render_material_overrides {
            entry.node_id = fixup_transform_id(
                entry.node_id,
                removal.removed_index,
                removal.last_index_before_swap,
            );
        }
    }
}
