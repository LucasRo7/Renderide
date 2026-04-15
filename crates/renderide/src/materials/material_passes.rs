//! Per-pass pipeline descriptor for multi-pass material shaders.
//!
//! A material stem may declare multiple passes via `//#pass` directives parsed in `build.rs` and
//! embedded alongside the composed WGSL (see [`crate::embedded_shaders::embedded_target_passes`]).
//! Each pass becomes one `wgpu::RenderPipeline`; the forward encode loop dispatches all pipelines
//! for every draw that binds the material, in declared order.
//!
//! Single-pass materials (the majority) do not declare any directives and fall through to
//! [`default_pass`], which preserves the pre-multi-pass behavior exactly.

use crate::assets::material::{
    MaterialDictionary, MaterialPropertyLookupIds, MaterialPropertyValue, PropertyIdRegistry,
};

/// Resonite/Froox material blend mode, or the shader stem's default when no material field is present.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MaterialBlendMode {
    /// No material-level override; use the stem's normal static behavior.
    #[default]
    StemDefault,
    /// `BlendMode.Opaque` (`0`).
    Opaque,
    /// `BlendMode.Cutout` (`1`).
    Cutout,
    /// `BlendMode.Alpha` (`2`).
    Alpha,
    /// `BlendMode.Transparent` (`3`).
    Transparent,
    /// `BlendMode.Additive` (`4`).
    Additive,
    /// `BlendMode.Multiply` (`5`).
    Multiply,
    /// Direct Unity `Blend[src][dst], One One` factors from `_SrcBlend` / `_DstBlend`.
    UnityBlend {
        /// Unity source blend factor enum value.
        src: u8,
        /// Unity destination blend factor enum value.
        dst: u8,
    },
}

impl MaterialBlendMode {
    /// Converts Resonite's `BlendMode` enum value into a pipeline mode.
    pub fn from_resonite_value(v: f32) -> Self {
        match v.round() as i32 {
            0 => Self::Opaque,
            1 => Self::Cutout,
            2 => Self::Alpha,
            3 => Self::Transparent,
            4 => Self::Additive,
            5 => Self::Multiply,
            _ => Self::StemDefault,
        }
    }

    fn unity_blend_factors(self) -> Option<(u8, u8)> {
        match self {
            Self::StemDefault => None,
            Self::Opaque | Self::Cutout => Some((1, 0)),
            Self::Alpha => Some((5, 10)),
            Self::Transparent => Some((1, 10)),
            Self::Additive => Some((1, 1)),
            Self::Multiply => Some((2, 0)),
            Self::UnityBlend { src, dst } => Some((src, dst)),
        }
    }

    fn unity_src_blend_factor(self) -> Option<u8> {
        self.unity_blend_factors().map(|(src, _)| src)
    }

    /// Converts Unity `BlendMode` factor property values (`_SrcBlend`, `_DstBlend`).
    pub fn from_unity_blend_factors(src: f32, dst: f32) -> Self {
        let src = src.round().clamp(0.0, 255.0) as u8;
        let dst = dst.round().clamp(0.0, 255.0) as u8;
        match (src, dst) {
            // UnityEngine.Rendering.BlendMode.One / Zero.
            (1, 0) => Self::Opaque,
            _ if unity_blend_factor(src).is_some() && unity_blend_factor(dst).is_some() => {
                Self::UnityBlend { src, dst }
            }
            _ => Self::StemDefault,
        }
    }

    /// Returns true when the mode must be sorted/drawn as transparent.
    pub fn is_transparent(self) -> bool {
        matches!(
            self,
            Self::Alpha
                | Self::Transparent
                | Self::Additive
                | Self::Multiply
                | Self::UnityBlend { .. }
        )
    }
}

/// Property ids used for material-driven pipeline state.
#[derive(Clone, Copy, Debug)]
pub struct MaterialPipelinePropertyIds {
    blend_mode: [i32; 2],
    src_blend: [i32; 2],
    dst_blend: [i32; 2],
}

impl MaterialPipelinePropertyIds {
    /// Interns property names used by Resonite material components and Unity-style shaders.
    pub fn new(registry: &PropertyIdRegistry) -> Self {
        Self {
            blend_mode: [registry.intern("_BlendMode"), registry.intern("BlendMode")],
            src_blend: [registry.intern("_SrcBlend"), registry.intern("SrcBlend")],
            dst_blend: [registry.intern("_DstBlend"), registry.intern("DstBlend")],
        }
    }
}

fn first_float_by_pids(
    dict: &MaterialDictionary<'_>,
    lookup: MaterialPropertyLookupIds,
    pids: &[i32],
) -> Option<f32> {
    pids.iter()
        .find_map(|&pid| match dict.get_merged(lookup, pid) {
            Some(MaterialPropertyValue::Float(f)) => Some(*f),
            Some(MaterialPropertyValue::Float4(v)) => Some(v[0]),
            _ => None,
        })
}

/// Resolves a material/property-block `BlendMode` override.
pub fn material_blend_mode_for_lookup(
    dict: &MaterialDictionary<'_>,
    lookup: MaterialPropertyLookupIds,
    ids: &MaterialPipelinePropertyIds,
) -> MaterialBlendMode {
    if let Some(v) = first_float_by_pids(dict, lookup, &ids.blend_mode) {
        return MaterialBlendMode::from_resonite_value(v);
    }
    if let (Some(src), Some(dst)) = (
        first_float_by_pids(dict, lookup, &ids.src_blend),
        first_float_by_pids(dict, lookup, &ids.dst_blend),
    ) {
        return MaterialBlendMode::from_unity_blend_factors(src, dst);
    }

    MaterialBlendMode::StemDefault
}

/// How a declared shader pass applies material-driven Unity render state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MaterialPassState {
    /// Use the pass descriptor exactly as authored.
    #[default]
    Static,
    /// Unity ForwardBase: `Blend [_SrcBlend] [_DstBlend]`, `ZWrite [_ZWrite]`.
    UnityForwardBase,
    /// Unity ForwardAdd: `Blend [_SrcBlend] One`, `ZWrite Off`.
    UnityForwardAdd,
}

/// Pipeline state for one pass of a material shader. All fields are `const`-constructible so the
/// build script can emit tables directly into generated Rust.
#[derive(Debug, Clone, Copy)]
pub struct MaterialPassDesc {
    /// Debug label for logs / pipeline names.
    pub name: &'static str,
    /// Vertex shader entry point.
    pub vertex_entry: &'static str,
    /// Fragment shader entry point.
    pub fragment_entry: &'static str,
    /// Depth comparison under reverse-Z. Unity `LEqual` maps to `GreaterEqual`; Unity `Greater` maps to `Less`.
    pub depth_compare: wgpu::CompareFunction,
    /// Whether this pass writes to the depth buffer.
    pub depth_write: bool,
    /// Backface culling mode (`None` = disabled).
    pub cull_mode: Option<wgpu::Face>,
    /// Color + alpha blend state, or `None` for no blending.
    pub blend: Option<wgpu::BlendState>,
    /// Color attachment write mask.
    pub write_mask: wgpu::ColorWrites,
    /// Slope-scaled depth bias.
    pub depth_bias_slope_scale: f32,
    /// Constant depth bias.
    pub depth_bias_constant: i32,
    /// Optional material-driven Unity pass-state override.
    pub material_state: MaterialPassState,
}

/// Default single-pass descriptor matching the pre-multi-pass pipeline builder.
///
/// `use_alpha_blending` picks between opaque (`ColorWrites::COLOR`, `blend: None`) and transparent
/// (`ColorWrites::ALL`, `ALPHA_BLENDING`). `depth_write` mirrors the old `depth_write_enabled` arg.
pub const fn default_pass(use_alpha_blending: bool, depth_write: bool) -> MaterialPassDesc {
    let (blend, write_mask) = if use_alpha_blending {
        (
            Some(wgpu::BlendState::ALPHA_BLENDING),
            wgpu::ColorWrites::ALL,
        )
    } else {
        (None, wgpu::ColorWrites::COLOR)
    };
    MaterialPassDesc {
        name: "main",
        vertex_entry: "vs_main",
        fragment_entry: "fs_main",
        depth_compare: crate::render_graph::MAIN_FORWARD_DEPTH_COMPARE,
        depth_write,
        cull_mode: None,
        blend,
        write_mask,
        depth_bias_slope_scale: 0.0,
        depth_bias_constant: 0,
        material_state: MaterialPassState::Static,
    }
}

fn unity_blend_factor(value: u8) -> Option<wgpu::BlendFactor> {
    match value {
        // UnityEngine.Rendering.BlendMode enum values.
        0 => Some(wgpu::BlendFactor::Zero),
        1 => Some(wgpu::BlendFactor::One),
        2 => Some(wgpu::BlendFactor::Dst),
        3 => Some(wgpu::BlendFactor::Src),
        4 => Some(wgpu::BlendFactor::OneMinusDst),
        5 => Some(wgpu::BlendFactor::SrcAlpha),
        6 => Some(wgpu::BlendFactor::OneMinusSrc),
        7 => Some(wgpu::BlendFactor::DstAlpha),
        8 => Some(wgpu::BlendFactor::OneMinusDstAlpha),
        9 => Some(wgpu::BlendFactor::SrcAlphaSaturated),
        10 => Some(wgpu::BlendFactor::OneMinusSrcAlpha),
        _ => None,
    }
}

fn unity_blend_state(src: u8, dst: u8) -> Option<wgpu::BlendState> {
    if src == 1 && dst == 0 {
        return None;
    }
    Some(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: unity_blend_factor(src)?,
            dst_factor: unity_blend_factor(dst)?,
            operation: wgpu::BlendOperation::Add,
        },
        // Matches Unity shader syntax: `Blend[src][dst], One One` + `BlendOp Add, Max`.
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Max,
        },
    })
}

fn unity_single_blend_state(src: u8, dst: u8) -> Option<wgpu::BlendState> {
    if src == 1 && dst == 0 {
        return None;
    }
    let src_factor = unity_blend_factor(src)?;
    let dst_factor = unity_blend_factor(dst)?;
    Some(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor,
            dst_factor,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor,
            dst_factor,
            operation: wgpu::BlendOperation::Add,
        },
    })
}

fn unity_blend_pass(name: &'static str, src: u8, dst: u8, depth_write: bool) -> MaterialPassDesc {
    MaterialPassDesc {
        name,
        vertex_entry: "vs_main",
        fragment_entry: "fs_main",
        depth_compare: crate::render_graph::MAIN_FORWARD_DEPTH_COMPARE,
        depth_write,
        cull_mode: None,
        blend: unity_blend_state(src, dst),
        write_mask: wgpu::ColorWrites::ALL,
        depth_bias_slope_scale: 0.0,
        depth_bias_constant: 0,
        material_state: MaterialPassState::Static,
    }
}

/// Applies runtime material blend state to a declared pass descriptor.
pub fn materialized_pass_for_blend_mode(
    pass: &MaterialPassDesc,
    blend_mode: MaterialBlendMode,
) -> MaterialPassDesc {
    match pass.material_state {
        MaterialPassState::Static => *pass,
        MaterialPassState::UnityForwardBase => {
            let Some((src, dst)) = blend_mode.unity_blend_factors() else {
                return *pass;
            };
            let blend = unity_single_blend_state(src, dst);
            MaterialPassDesc {
                blend,
                write_mask: if blend.is_some() {
                    wgpu::ColorWrites::ALL
                } else {
                    wgpu::ColorWrites::COLOR
                },
                depth_write: src == 1 && dst == 0,
                ..*pass
            }
        }
        MaterialPassState::UnityForwardAdd => {
            let src = blend_mode.unity_src_blend_factor().unwrap_or(1);
            MaterialPassDesc {
                blend: unity_single_blend_state(src, 1),
                write_mask: wgpu::ColorWrites::ALL,
                depth_write: false,
                ..*pass
            }
        }
    }
}

/// Default single-pass descriptor after applying a material `BlendMode` override.
pub fn default_pass_for_blend_mode(
    stem_uses_alpha_blending: bool,
    blend_mode: MaterialBlendMode,
) -> MaterialPassDesc {
    match blend_mode {
        MaterialBlendMode::StemDefault => {
            default_pass(stem_uses_alpha_blending, !stem_uses_alpha_blending)
        }
        MaterialBlendMode::Opaque | MaterialBlendMode::Cutout => default_pass(false, true),
        MaterialBlendMode::Alpha => unity_blend_pass("alpha", 5, 10, false),
        // Resonite's Transparent mode is premultiplied-alpha style.
        MaterialBlendMode::Transparent => unity_blend_pass("transparent", 1, 10, false),
        MaterialBlendMode::Additive => unity_blend_pass("additive", 1, 1, false),
        MaterialBlendMode::Multiply => unity_blend_pass("multiply", 2, 0, false),
        MaterialBlendMode::UnityBlend { src, dst } => {
            unity_blend_pass("unity_blend", src, dst, src == 1 && dst == 0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::material::{MaterialPropertyStore, PropertyIdRegistry};

    #[test]
    fn resolves_resonite_blend_mode_property() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let pid = reg.intern("BlendMode");
        store.set_material(42, pid, MaterialPropertyValue::Float(4.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 42,
            mesh_property_block_slot0: None,
        };
        assert_eq!(
            material_blend_mode_for_lookup(&dict, lookup, &ids),
            MaterialBlendMode::Additive
        );
    }

    #[test]
    fn resolves_unity_src_dst_blend_properties() {
        let reg = PropertyIdRegistry::new();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut store = MaterialPropertyStore::new();
        let src = reg.intern("_SrcBlend");
        let dst = reg.intern("_DstBlend");
        store.set_material(43, src, MaterialPropertyValue::Float(1.0));
        store.set_material(43, dst, MaterialPropertyValue::Float(1.0));
        let dict = MaterialDictionary::new(&store);
        let lookup = MaterialPropertyLookupIds {
            material_asset_id: 43,
            mesh_property_block_slot0: None,
        };
        assert_eq!(
            material_blend_mode_for_lookup(&dict, lookup, &ids),
            MaterialBlendMode::UnityBlend { src: 1, dst: 1 }
        );
    }
}
