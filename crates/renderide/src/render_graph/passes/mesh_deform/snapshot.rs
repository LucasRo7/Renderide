//! GPU mesh handles and predicates for deform dispatch eligibility.

use std::sync::Arc;

use glam::Mat4;

use crate::assets::mesh::GpuMesh;

/// GPU buffer handles + metadata copied from [`crate::assets::mesh::GpuMesh`] so we can hold
/// deform state without borrowing the mesh pool across preprocess/scratch access.
pub(super) struct MeshDeformSnapshot {
    pub(super) vertex_count: u32,
    pub(super) num_blendshapes: u32,
    pub(super) has_skeleton: bool,
    pub(super) positions_buffer: Option<Arc<wgpu::Buffer>>,
    pub(super) normals_buffer: Option<Arc<wgpu::Buffer>>,
    pub(super) blendshape_buffer: Option<Arc<wgpu::Buffer>>,
    pub(super) deform_temp_buffer: Option<Arc<wgpu::Buffer>>,
    pub(super) deformed_positions_buffer: Option<Arc<wgpu::Buffer>>,
    pub(super) deformed_normals_buffer: Option<Arc<wgpu::Buffer>>,
    pub(super) bone_indices_buffer: Option<Arc<wgpu::Buffer>>,
    pub(super) bone_weights_vec4_buffer: Option<Arc<wgpu::Buffer>>,
    pub(super) skinning_bind_matrices: Vec<Mat4>,
}

impl MeshDeformSnapshot {
    /// Copies GPU resources from `m`. When `clone_skinning_bind_matrices` is `false`, bind matrices
    /// are omitted (blendshape-only path never reads them).
    pub(super) fn from_mesh(m: &GpuMesh, clone_skinning_bind_matrices: bool) -> Self {
        Self {
            vertex_count: m.vertex_count,
            num_blendshapes: m.num_blendshapes,
            has_skeleton: m.has_skeleton,
            positions_buffer: m.positions_buffer.clone(),
            normals_buffer: m.normals_buffer.clone(),
            blendshape_buffer: m.blendshape_buffer.clone(),
            deform_temp_buffer: m.deform_temp_buffer.clone(),
            deformed_positions_buffer: m.deformed_positions_buffer.clone(),
            deformed_normals_buffer: m.deformed_normals_buffer.clone(),
            bone_indices_buffer: m.bone_indices_buffer.clone(),
            bone_weights_vec4_buffer: m.bone_weights_vec4_buffer.clone(),
            skinning_bind_matrices: if clone_skinning_bind_matrices {
                m.skinning_bind_matrices.clone()
            } else {
                Vec::new()
            },
        }
    }
}

/// Returns whether blendshape compute should run for `m` (parity with deform encode).
pub(super) fn deform_needs_blend_mesh(m: &GpuMesh) -> bool {
    m.num_blendshapes > 0 && m.blendshape_buffer.is_some() && m.deform_temp_buffer.is_some()
}

/// Returns whether skinning compute should run for `m` (parity with deform encode).
pub(super) fn deform_needs_skin_mesh(m: &GpuMesh, bone_transform_indices: Option<&[i32]>) -> bool {
    bone_transform_indices.is_some()
        && m.has_skeleton
        && m.deformed_positions_buffer.is_some()
        && m.deformed_normals_buffer.is_some()
        && m.normals_buffer.is_some()
        && m.bone_indices_buffer.is_some()
        && m.bone_weights_vec4_buffer.is_some()
        && !m.skinning_bind_matrices.is_empty()
}

/// Returns `true` when deform encoding would run blend and/or skin dispatches (not an early no-op).
pub(super) fn gpu_mesh_needs_deform_dispatch(
    m: &GpuMesh,
    bone_transform_indices: Option<&[i32]>,
) -> bool {
    if m.positions_buffer.is_none() || m.vertex_count == 0 {
        return false;
    }
    deform_needs_blend_mesh(m) || deform_needs_skin_mesh(m, bone_transform_indices)
}

/// Snapshot variant of [`deform_needs_blend_mesh`].
pub(super) fn deform_needs_blend_snapshot(mesh: &MeshDeformSnapshot) -> bool {
    mesh.num_blendshapes > 0
        && mesh.blendshape_buffer.is_some()
        && mesh.deform_temp_buffer.is_some()
}

/// Snapshot variant of [`deform_needs_skin_mesh`].
pub(super) fn deform_needs_skin_snapshot(
    mesh: &MeshDeformSnapshot,
    bone_transform_indices: Option<&[i32]>,
) -> bool {
    bone_transform_indices.is_some()
        && mesh.has_skeleton
        && mesh.deformed_positions_buffer.is_some()
        && mesh.deformed_normals_buffer.is_some()
        && mesh.normals_buffer.is_some()
        && mesh.bone_indices_buffer.is_some()
        && mesh.bone_weights_vec4_buffer.is_some()
        && !mesh.skinning_bind_matrices.is_empty()
}
