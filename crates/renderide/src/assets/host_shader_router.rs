//! Canonical routing from host [`ShaderAsset`](super::ShaderAsset) identity to native Resonite material
//! families (PBS metallic, UI Unlit, world Unlit) vs legacy host-unlit pilot draws.
//!
//! Used by [`crate::gpu::ShaderKey::effective_variant`] so [`crate::gpu::PipelineVariant::Material`]
//! is only selected for world-unlit and unknown shaders when the host-unlit pilot is enabled — not
//! for [`NativeMaterialPipelineFamily::PbsMetallic`], which must stay on the global PBR path.

use super::AssetRegistry;

/// Native host shader families that map to in-tree WGSL implementations (strangler target).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NativeMaterialPipelineFamily {
    /// Resonite `Shader "PBSMetallic"` → forward clustered PBR ([`crate::gpu::PipelineVariant::Pbr`] family).
    PbsMetallic,
    /// `Shader "UI/Unlit"` / `UI/Text/Unlit` and stencil variants ([`crate::gpu::PipelineVariant::NativeUiUnlit`]).
    UiUnlit,
    /// Resonite world `Shader "Unlit"` ([`crate::gpu::pipeline::WorldUnlitPipeline`]).
    WorldUnlit,
    /// No recognized native family: host-unlit pilot may use [`crate::gpu::PipelineVariant::Material`].
    LegacyFallback,
}

fn compact_alnum_lower(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Returns true when the Unity ShaderLab logical name denotes PBS metallic (Resonite `PBSMetallic`).
pub fn pbs_metallic_family_from_unity_shader_name(name: &str) -> bool {
    let Some(token) = name.split_whitespace().next() else {
        return false;
    };
    let key = compact_alnum_lower(token);
    let k_pbs = compact_alnum_lower("PBSMetallic");
    key == k_pbs
}

/// Path / label hints for bundled PBS shaders (e.g. `PBSSpecular.shader`).
pub fn pbs_metallic_family_from_shader_path_hint(hint: &str) -> bool {
    let h = hint.to_ascii_lowercase();
    h.contains("pbsmetallic")
        || h.contains("pbs_specular")
        || h.contains("pbsspecular")
        || h.contains("pbs/specular")
}

/// Resolves PBS metallic from stored Unity name or upload path/label.
pub fn resolve_pbs_metallic_shader_family(shader_asset_id: i32, registry: &AssetRegistry) -> bool {
    let Some(s) = registry.get_shader(shader_asset_id) else {
        return false;
    };
    if let Some(name) = s.unity_shader_name.as_deref()
        && pbs_metallic_family_from_unity_shader_name(name)
    {
        return true;
    }
    if let Some(file) = s.wgsl_source.as_deref()
        && pbs_metallic_family_from_shader_path_hint(file)
    {
        return true;
    }
    false
}

/// Classifies a host shader asset for [`NativeMaterialPipelineFamily`] (name/path first, then INI
/// compatibility via [`super::ui_material_contract`] / [`super::world_unlit_material_contract`]).
pub fn native_material_family_for_shader(
    host_shader_asset_id: Option<i32>,
    render_config: &crate::config::RenderConfig,
    registry: &AssetRegistry,
) -> NativeMaterialPipelineFamily {
    let Some(sid) = host_shader_asset_id else {
        return NativeMaterialPipelineFamily::LegacyFallback;
    };
    if super::resolve_native_ui_shader_family(
        sid,
        render_config.native_ui_unlit_shader_id,
        render_config.native_ui_text_unlit_shader_id,
        registry,
    )
    .is_some()
    {
        return NativeMaterialPipelineFamily::UiUnlit;
    }
    if super::resolve_world_unlit_shader_family(
        sid,
        render_config.native_world_unlit_shader_id,
        registry,
    )
    .is_some()
    {
        return NativeMaterialPipelineFamily::WorldUnlit;
    }
    if resolve_pbs_metallic_shader_family(sid, registry) {
        return NativeMaterialPipelineFamily::PbsMetallic;
    }
    NativeMaterialPipelineFamily::LegacyFallback
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::AssetRegistry;
    use crate::config::RenderConfig;
    use crate::shared::ShaderUpload;

    #[test]
    fn pbs_metallic_from_unity_name() {
        assert!(pbs_metallic_family_from_unity_shader_name("PBSMetallic"));
        assert!(!pbs_metallic_family_from_unity_shader_name("Unlit"));
    }

    #[test]
    fn native_family_pbs_from_registry() {
        let mut reg = AssetRegistry::new();
        reg.handle_shader_upload(ShaderUpload {
            asset_id: 9,
            file: Some("PBSMetallic".to_string()),
        });
        let rc = RenderConfig::default();
        assert_eq!(
            native_material_family_for_shader(Some(9), &rc, &reg),
            NativeMaterialPipelineFamily::PbsMetallic
        );
    }

    #[test]
    fn native_family_world_unlit_resolves() {
        let mut reg = AssetRegistry::new();
        reg.handle_shader_upload(ShaderUpload {
            asset_id: 2,
            file: Some("Unlit".to_string()),
        });
        let rc = RenderConfig::default();
        assert_eq!(
            native_material_family_for_shader(Some(2), &rc, &reg),
            NativeMaterialPipelineFamily::WorldUnlit
        );
    }
}
