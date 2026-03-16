//! Render configuration types.
//!
//! Engine-agnostic configuration structures used by the renderer framework.
//!
//! Extension point for config, feature flags.

/// Render configuration (clip planes, FOV, display settings).
#[derive(Clone, Debug)]
pub struct RenderConfig {
    /// Near clip plane distance.
    pub near_clip: f32,
    /// Far clip plane distance.
    pub far_clip: f32,
    /// Desktop field of view in degrees.
    pub desktop_fov: f32,
    /// Whether vertical sync is enabled.
    pub vsync: bool,
    /// When true, use UV debug pipeline for meshes that have UVs.
    pub use_debug_uv: bool,
    /// When true, apply the mesh root (drawable's model_matrix) to skinned MVP.
    /// Matches Unity SkinnedMeshRenderer: vertices are in mesh root local space.
    pub skinned_apply_mesh_root_transform: bool,
    /// When true, use root_bone_transform_id from BoneAssignment for root-relative bone matrices.
    /// Enables A/B testing of coordinate alignment. Default false.
    pub skinned_use_root_bone: bool,
    /// When true, log diagnostic info for the first skinned draw each frame.
    pub debug_skinned: bool,
    /// When true, apply an extra Z flip to skinned MVP for handedness correction.
    /// Use when skinned meshes appear mirrored vs non-skinned. Default false.
    pub skinned_flip_handedness: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            near_clip: 0.01,
            far_clip: 1024.0,
            desktop_fov: 75.0,
            vsync: false,
            use_debug_uv: false,
            skinned_apply_mesh_root_transform: true,
            skinned_use_root_bone: false,
            debug_skinned: false,
            skinned_flip_handedness: false,
        }
    }
}
