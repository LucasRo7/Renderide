//! Unity `Shader "Invisible"`: vertex collapses to origin and the fragment unconditionally
//! discards. Used as a hit-volume material that contributes nothing to color or depth.


#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::mesh::vertex as mv

struct InvisibleMaterial {
    _pad: vec4<f32>,
}

@group(1) @binding(0) var<uniform> mat: InvisibleMaterial;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) _pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
) -> mv::ClipVertexOutput {
    let d = pd::get_draw(instance_index);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif
    var out: mv::ClipVertexOutput;
    out.clip_pos = vp * vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return out;
}

//#pass forward
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    discard;
    return rg::retain_globals_additive(vec4<f32>(mat._pad.x * 0.0));
}
