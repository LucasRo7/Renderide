//! Unity unlit `Shader "Unlit/NewUnlitShader"`: Shader Forge "diffuse-only" output collapsed to an
//! emissive solid color in this renderer — Unity's `_LightColor0` / per-light path is replaced by a
//! constant `_node_2829` tint. The shader was a placeholder in the Resonite asset bundle and never
//! consumed real light data downstream.


#import renderide::globals as rg
#import renderide::mesh::vertex as mv

struct NewUnlitMaterial {
    _node_2829: vec4<f32>,
}

@group(1) @binding(0) var<uniform> mat: NewUnlitMaterial;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
) -> mv::ClipVertexOutput {
#ifdef MULTIVIEW
    return mv::clip_vertex_main(instance_index, view_idx, pos);
#else
    return mv::clip_vertex_main(instance_index, 0u, pos);
#endif
}

//#pass forward
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return rg::retain_globals_additive(vec4<f32>(mat._node_2829.rgb, 1.0));
}
