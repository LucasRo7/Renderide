//! Maps host shader asset ids from `set_shader` to renderer [`super::MaterialFamilyId`].
//!
//! Until [`crate::shared::ShaderUpload`] is wired with name extraction, use an explicit map plus
//! fallback family (see [`MaterialRouter::family_for_shader_asset`]).

use std::collections::HashMap;

use super::MaterialFamilyId;

/// Shader asset id → material family; unknown ids use [`Self::fallback`].
#[derive(Debug)]
pub struct MaterialRouter {
    shader_to_family: HashMap<i32, MaterialFamilyId>,
    /// Default when `shader_to_family` has no entry.
    pub fallback: MaterialFamilyId,
}

impl MaterialRouter {
    /// Builds a router with only a fallback family.
    pub fn new(fallback: MaterialFamilyId) -> Self {
        Self {
            shader_to_family: HashMap::new(),
            fallback,
        }
    }

    /// Inserts or replaces a host shader → family mapping.
    pub fn set_shader_family(&mut self, shader_asset_id: i32, family: MaterialFamilyId) {
        self.shader_to_family.insert(shader_asset_id, family);
    }

    /// Resolves the family for a host shader asset id.
    pub fn family_for_shader_asset(&self, shader_asset_id: i32) -> MaterialFamilyId {
        self.shader_to_family
            .get(&shader_asset_id)
            .copied()
            .unwrap_or(self.fallback)
    }
}
