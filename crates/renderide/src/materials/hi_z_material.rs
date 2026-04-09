//! Reflection-driven classification: skip temporal Hi-Z occlusion for transparent / non-opaque draws.
//!
//! Uses Unity-style `_Mode`, `_SrcBlend`, `_DstBlend`, and `_ZWrite` when those names appear in the
//! reflected `@group(1)` uniform block—no per-shader-name branching in Rust.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::assets::material::{
    MaterialDictionary, MaterialPropertyLookupIds, MaterialPropertyValue, PropertyIdRegistry,
};
use crate::materials::embedded_raster_pipeline::embedded_composed_stem_for_permutation;
use crate::materials::pipeline_kind::RasterPipelineKind;
use crate::materials::reflect_raster_material_wgsl;
use crate::pipelines::ShaderPermutation;

use super::wgsl_reflect::ReflectedMaterialUniformBlock;

type HiZReflectCacheMap = HashMap<(i32, String, u32), Option<ReflectedMaterialUniformBlock>>;

const UNITY_BLEND_ONE: f32 = 1.0;
const UNITY_BLEND_ZERO: f32 = 0.0;
const EPS: f32 = 0.05;

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() < EPS
}

/// Unity Standard-style `_Mode`: 0 Opaque, 1 Cutout, 2 Fade, 3 Transparent.
#[inline]
fn mode_skips_hi_z(mode: f32) -> bool {
    near(mode, 2.0) || near(mode, 3.0)
}

/// Returns `true` when this draw should **not** be tested against Hi-Z (transparent / unreliable depth).
pub fn material_skips_hi_z_occlusion(
    pipeline: &RasterPipelineKind,
    shader_asset_id: i32,
    dict: &MaterialDictionary<'_>,
    lookup: MaterialPropertyLookupIds,
    registry: &PropertyIdRegistry,
    shader_perm: ShaderPermutation,
) -> bool {
    let Some(block) = reflected_material_block_cached(pipeline, shader_asset_id, shader_perm)
    else {
        return false;
    };

    let has = |name: &str| block.fields.contains_key(name);

    let mode_id = registry.intern("_Mode");
    let zwrite_id = registry.intern("_ZWrite");
    let src_id = registry.intern("_SrcBlend");
    let dst_id = registry.intern("_DstBlend");

    if has("_Mode") {
        if let Some(MaterialPropertyValue::Float(m)) = dict.get_merged(lookup, mode_id) {
            if mode_skips_hi_z(*m) {
                return true;
            }
        }
    }

    if has("_ZWrite") {
        if let Some(MaterialPropertyValue::Float(z)) = dict.get_merged(lookup, zwrite_id) {
            if *z < 0.5 {
                return true;
            }
        }
    }

    let has_src = has("_SrcBlend");
    let has_dst = has("_DstBlend");
    if has_src && has_dst {
        let src = dict.get_merged(lookup, src_id).and_then(|v| match v {
            MaterialPropertyValue::Float(f) => Some(*f),
            _ => None,
        });
        let dst = dict.get_merged(lookup, dst_id).and_then(|v| match v {
            MaterialPropertyValue::Float(f) => Some(*f),
            _ => None,
        });
        match (src, dst) {
            (Some(s), Some(d)) => {
                if !(near(s, UNITY_BLEND_ONE) && near(d, UNITY_BLEND_ZERO)) {
                    return true;
                }
            }
            _ => return true,
        }
    } else if has_src || has_dst {
        return true;
    }

    false
}

fn reflected_material_block_cached(
    pipeline: &RasterPipelineKind,
    shader_asset_id: i32,
    shader_perm: ShaderPermutation,
) -> Option<ReflectedMaterialUniformBlock> {
    let RasterPipelineKind::EmbeddedStem(stem) = pipeline else {
        return None;
    };
    let stem = stem.as_ref();
    let key = (shader_asset_id, stem.to_string(), shader_perm.0);
    let cache = hi_z_reflect_cache();
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(b) = guard.get(&key) {
        return b.clone();
    }
    let composed = embedded_composed_stem_for_permutation(stem, shader_perm);
    let wgsl = crate::embedded_shaders::embedded_target_wgsl(&composed)?;
    let reflected = reflect_raster_material_wgsl(wgsl).ok()?;
    let block = reflected.material_uniform?;
    let cloned = block.clone();
    guard.insert(key, Some(cloned.clone()));
    Some(cloned)
}

fn hi_z_reflect_cache() -> &'static Mutex<HiZReflectCacheMap> {
    static CACHE: OnceLock<Mutex<HiZReflectCacheMap>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::material::MaterialPropertyStore;
    use crate::materials::wgsl_reflect::{ReflectedUniformField, ReflectedUniformScalarKind};
    use std::sync::Arc;

    #[test]
    fn mode_fade_skips() {
        let reg = PropertyIdRegistry::new();
        let mid = reg.intern("_Mode");
        let mut store = MaterialPropertyStore::new();
        store.set_material(1, mid, MaterialPropertyValue::Float(2.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 1,
            mesh_property_block_slot0: None,
        };
        let mut fields = HashMap::new();
        fields.insert(
            "_Mode".to_string(),
            ReflectedUniformField {
                offset: 0,
                size: 4,
                kind: ReflectedUniformScalarKind::F32,
            },
        );
        let block = ReflectedMaterialUniformBlock {
            binding: 0,
            total_size: 16,
            fields,
        };
        let mut guard = hi_z_reflect_cache().lock().unwrap();
        guard.insert((42, "pbsmetallic_default".to_string(), 0), Some(block));
        drop(guard);
        let pipeline = RasterPipelineKind::EmbeddedStem(Arc::from("pbsmetallic_default"));
        assert!(material_skips_hi_z_occlusion(
            &pipeline,
            42,
            &dict,
            lookup,
            &reg,
            ShaderPermutation(0),
        ));
    }

    #[test]
    fn unity_one_zero_blend_allows_hi_z() {
        let reg = PropertyIdRegistry::new();
        let src_id = reg.intern("_SrcBlend");
        let dst_id = reg.intern("_DstBlend");
        let mut store = MaterialPropertyStore::new();
        store.set_material(1, src_id, MaterialPropertyValue::Float(UNITY_BLEND_ONE));
        store.set_material(1, dst_id, MaterialPropertyValue::Float(UNITY_BLEND_ZERO));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 1,
            mesh_property_block_slot0: None,
        };
        let mut fields = HashMap::new();
        for (name, offset) in [("_SrcBlend", 0), ("_DstBlend", 4)] {
            fields.insert(
                name.to_string(),
                ReflectedUniformField {
                    offset,
                    size: 4,
                    kind: ReflectedUniformScalarKind::F32,
                },
            );
        }
        let block = ReflectedMaterialUniformBlock {
            binding: 0,
            total_size: 16,
            fields,
        };
        let mut guard = hi_z_reflect_cache().lock().unwrap();
        guard.insert((43, "ui_test".to_string(), 0), Some(block));
        drop(guard);
        let pipeline = RasterPipelineKind::EmbeddedStem(Arc::from("ui_test"));
        assert!(!material_skips_hi_z_occlusion(
            &pipeline,
            43,
            &dict,
            lookup,
            &reg,
            ShaderPermutation(0),
        ));
    }
}
