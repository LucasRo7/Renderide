//! Fullscreen pass: samples HDR [`scene_color_hdr`] and writes the displayable color target.
//! Multiview stereo (`array_layer_count` 2). Future exposure / tonemap / grading hook.

@group(0) @binding(0) var scene_color_hdr: texture_2d_array<f32>;
@group(0) @binding(1) var scene_color_sampler: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    out.clip_pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VsOut, @builtin(view_index) view: u32) -> @location(0) vec4<f32> {
    return textureSample(scene_color_hdr, scene_color_sampler, in.uv, i32(view));
}
