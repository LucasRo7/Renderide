//! Canvas UI Unlit (`Shader "UI/Unlit"`): sprite texture, tint, optional alpha clip, optional alpha mask.
//!
//! Build emits `ui_unlit_default` / `ui_unlit_multiview` via [`MULTIVIEW`](https://docs.rs/naga_oil).
//! `@group(1)` global names match Unity `UI_Unlit.shader` material property names for host reflection.
//!
//! **Vertex color:** Unity multiplies `vertex_color * _Tint`. The mesh pass provides a dense
//! float4 color stream at `@location(3)` with opaque-white fallback when the host mesh lacks color.
//!
//! Mask-mode caveat: Unity's UI_Unlit shader gates mask handling on
//! `_MASK_TEXTURE_MUL` / `_MASK_TEXTURE_CLIP` multi-compile keywords that FrooxEngine sets
//! through `ShaderKeywords.SetKeyword`, which the renderer never receives. The
//! `_ALPHATEST_ON` / `_ALPHABLEND_ON` keyword fields below are populated by
//! [`crate::backend::embedded::uniform_pack::inferred_keyword_float_f32`] from the on-wire
//! `MaterialRenderType` tag (Cutout enables `_ALPHATEST_ON`; Transparent enables
//! `_ALPHABLEND_ON`); Opaque leaves both at zero so mask and cutoff branches stay inert.
//! The default-white texture fallback keeps each mask branch a no-op when no host mask is
//! bound (`mask.a == 1.0`).
//!
//! Per-draw uniforms (`@group(2)`) use [`renderide::per_draw`].


#import renderide::texture_sampling as ts
#import renderide::globals as rg
#import renderide::alpha_clip_sample as acs
#import renderide::material::alpha as ma
#import renderide::mesh::vertex as mv
#import renderide::uv_utils as uvu

struct UiUnlitMaterial {
    _MainTex_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _MaskTex_ST: vec4<f32>,
    _Tint: vec4<f32>,
    _Cutoff: f32,
    _ALPHATEST_ON: f32,
    _ALPHABLEND_ON: f32,
    _MainTex_LodBias: f32,
    _MaskTex_LodBias: f32,
}

@group(1) @binding(0) var<uniform> mat: UiUnlitMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _MaskTex: texture_2d<f32>;
@group(1) @binding(4) var _MaskTex_sampler: sampler;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
) -> mv::UvColorVertexOutput {
#ifdef MULTIVIEW
    return mv::uv_color_vertex_main(instance_index, view_idx, pos, uv, color * mat._Tint);
#else
    return mv::uv_color_vertex_main(instance_index, 0u, pos, uv, color * mat._Tint);
#endif
}

//#pass forward
@fragment
fn fs_main(in: mv::UvColorVertexOutput) -> @location(0) vec4<f32> {
    let uv_s = uvu::apply_st_for_storage(in.uv, mat._MainTex_ST, mat._MainTex_StorageVInverted);
    let t = ts::sample_tex_2d(_MainTex, _MainTex_sampler, uv_s, mat._MainTex_LodBias);
    var color = in.color * t;

    let alpha_test = uvu::kw_enabled(mat._ALPHATEST_ON);
    let alpha_blend = uvu::kw_enabled(mat._ALPHABLEND_ON);

    let uv_mask = uvu::apply_st(in.uv, mat._MaskTex_ST);

    if (alpha_test) {
        let tex_clip_alpha = in.color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_s);
        let mask_clip_alpha = acs::mask_luminance_mul_base_mip(_MaskTex, _MaskTex_sampler, uv_mask);
        if (tex_clip_alpha * mask_clip_alpha <= mat._Cutoff) {
            discard;
        }
    } else if (alpha_blend) {
        let mask_sample = ts::sample_tex_2d(_MaskTex, _MaskTex_sampler, uv_mask, mat._MaskTex_LodBias);
        color = ma::apply_alpha_mask(color, mask_sample);
    }

    return rg::retain_globals_additive(color);
}
