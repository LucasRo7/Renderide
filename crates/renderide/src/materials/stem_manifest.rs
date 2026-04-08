//! Build-time [`crate::embedded_shaders::SHADER_MANIFEST_JSON`] → Unity logical name → target WGSL stem.
//!
//! Each entry may include [`ShaderManifestMaterialEntry::material_properties`], a host-name → GPU bridge
//! validated against [`crate::materials::reflect_raster_material_wgsl`] (naga cannot encode Unity
//! `MaterialPropertyBlock` names). Optional [`ShaderManifestMaterialEntry::vertex_streams`] drives
//! vertex buffer setup for manifest pipelines (see [`super::manifest_stem::ManifestStemMaterialFamily`]).
//!
//! [`ShaderManifestMaterialEntry::uniform_derived`] selects CPU-side fill rules for uniform fields not
//! sent as explicit host properties (e.g. packed `flags` for Unlit).

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Deserialize;

use crate::assets::util::normalize_unity_shader_lookup_key;

/// How to populate the `flags` (or similar) uniform word when the host does not send a dedicated property.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum UniformDerivedRule {
    /// Texture presence (`_Tex`) and `_Cutoff` alpha-test range → `flags` u32 (matches Unlit shader).
    UnlitFlags,
}

/// Host material property name → binding / uniform field (from per-stem `.meta.json` / `manifest.json`).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum MaterialPropertyBindingSpec {
    /// `vec4<f32>` in the material uniform struct.
    Float4 {
        /// WGSL struct member name (must exist in naga reflection for `@group(1)` uniform block).
        uniform_field: String,
    },
    /// `f32` in the material uniform struct.
    Float { uniform_field: String },
    /// Sampled 2D texture at `@group(1)` `@binding(binding)`.
    Texture { binding: u32 },
}

/// One material entry from `shaders/target/manifest.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct ShaderManifestMaterialEntry {
    /// File stem under `shaders/target/<stem>.wgsl`.
    pub stem: String,
    /// Unity `Shader "…"` keys (normalized at lookup time).
    #[serde(default)]
    pub unity_names: Vec<String>,
    /// Maps Unity-style property names (e.g. `_Color`) to WGSL bindings / uniform fields.
    #[serde(default)]
    pub material_properties: HashMap<String, MaterialPropertyBindingSpec>,
    /// Ordered list of vertex streams for this stem (`position`, `normal`, `uv0`, …).
    #[serde(default)]
    pub vertex_streams: Vec<String>,
    /// Optional rule for filling derived uniform fields after explicit properties are written.
    #[serde(default)]
    pub uniform_derived: Option<UniformDerivedRule>,
}

/// Parsed shader manifest embedded at build time.
#[derive(Debug, Deserialize)]
pub struct ShaderManifest {
    pub materials: Vec<ShaderManifestMaterialEntry>,
    #[serde(default)]
    pub globals_module: Option<String>,
}

impl ShaderManifest {
    /// Parses [`crate::embedded_shaders::SHADER_MANIFEST_JSON`].
    pub fn from_embedded_json() -> Self {
        serde_json::from_str(crate::embedded_shaders::SHADER_MANIFEST_JSON)
            .expect("SHADER_MANIFEST_JSON must match ShaderManifest schema")
    }

    /// Looks up a composed target stem entry (e.g. `world_unlit_default`).
    pub fn entry_for_stem(&self, stem: &str) -> Option<&ShaderManifestMaterialEntry> {
        self.materials.iter().find(|e| e.stem == stem)
    }

    /// Maps normalized Unity shader keys to target stems (first manifest match wins).
    pub fn unity_name_to_stem_map(manifest: &Self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        for entry in &manifest.materials {
            for name in &entry.unity_names {
                let key = normalize_unity_shader_lookup_key(name);
                m.entry(key).or_insert_with(|| entry.stem.clone());
            }
        }
        m
    }
}

/// Resolves a Unity-style logical shader name to a composed WGSL stem (`shaders/target/<stem>.wgsl`).
#[derive(Debug)]
pub struct StemResolver {
    unity_to_stem: HashMap<String, String>,
}

impl StemResolver {
    /// Builds the lookup table from the embedded manifest.
    pub fn from_embedded_manifest() -> Self {
        let manifest = ShaderManifest::from_embedded_json();
        Self {
            unity_to_stem: ShaderManifest::unity_name_to_stem_map(&manifest),
        }
    }

    /// Resolves a host-provided Unity shader name to a target stem, if listed in the manifest.
    pub fn stem_for_unity_name(&self, unity_shader_name: &str) -> Option<&str> {
        let key = normalize_unity_shader_lookup_key(unity_shader_name);
        self.unity_to_stem.get(&key).map(String::as_str)
    }
}

/// Returns the composed WGSL stem for `name` when it appears in the embedded manifest (used for routing).
///
/// Uses a process-wide [`StemResolver`] built from [`ShaderManifest::from_embedded_json`].
pub fn manifest_stem_for_unity_name(name: &str) -> Option<String> {
    static RESOLVER: OnceLock<StemResolver> = OnceLock::new();
    RESOLVER
        .get_or_init(StemResolver::from_embedded_manifest)
        .stem_for_unity_name(name)
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_parses() {
        let m = ShaderManifest::from_embedded_json();
        assert!(!m.materials.is_empty());
    }

    #[test]
    fn manifest_stem_resolves_unlit() {
        let s = super::manifest_stem_for_unity_name("Unlit").expect("manifest lists Unlit");
        assert!(
            s.starts_with("world_unlit"),
            "expected world_unlit stem, got {s}"
        );
    }
}
