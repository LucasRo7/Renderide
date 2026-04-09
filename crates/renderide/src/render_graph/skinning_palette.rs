//! CPU bone palette matching [`super::passes::mesh_deform`] skinning dispatch for culling parity.

use glam::Mat4;

use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::RenderingContext;

/// Builds the same `world_bone * skinning_bind_matrices[i]` palette as the skinning compute pass.
#[allow(clippy::too_many_arguments)] // Mirrors deform dispatch inputs; grouping would obscure parity.
pub fn build_skinning_palette(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    skinning_bind_matrices: &[Mat4],
    has_skeleton: bool,
    bone_transform_indices: &[i32],
    smr_node_id: i32,
    render_context: RenderingContext,
    head_output_transform: Mat4,
) -> Option<Vec<Mat4>> {
    let bone_count = skinning_bind_matrices.len();
    if bone_count == 0 || !has_skeleton {
        return None;
    }

    let smr_world = (smr_node_id >= 0)
        .then(|| {
            scene.world_matrix_for_render_context(
                space_id,
                smr_node_id as usize,
                render_context,
                head_output_transform,
            )
        })
        .flatten()
        .unwrap_or(Mat4::IDENTITY);

    let mut out = Vec::with_capacity(bone_count);
    for (bi, bind_mat) in skinning_bind_matrices.iter().enumerate() {
        let tid = bone_transform_indices.get(bi).copied().unwrap_or(-1);
        let pal = if tid < 0 {
            smr_world
        } else {
            match scene.world_matrix_for_render_context(
                space_id,
                tid as usize,
                render_context,
                head_output_transform,
            ) {
                Some(world) => world * bind_mat,
                None => smr_world,
            }
        };
        out.push(pal);
    }
    Some(out)
}
