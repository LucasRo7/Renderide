#import uniform_ring
#import ui_common

/// UI text unlit: RASTER vs SDF vs MSDF, derivative-based AA (`dpdx`/`dpdy` like HLSL `fwidth`),
/// median for MSDF, signed distance in alpha for SDF.

struct NativeUiOverlayUnproject {
    inv_scene_proj: mat4x4f,
    inv_ui_proj: mat4x4f,
}

fn overlay_ndc_xy(frag_xy: vec2f, dims: vec2u) -> vec2f {
    let w = max(f32(dims.x), 1.0);
    let h = max(f32(dims.y), 1.0);
    let x = (frag_xy.x + 0.5) / w * 2.0 - 1.0;
    let y = 1.0 - (frag_xy.y + 0.5) / h * 2.0;
    return vec2f(x, y);
}

fn overlay_view_z_from_depth(d: f32, ndc_xy: vec2f, inv_proj: mat4x4f) -> f32 {
    let ndc_z = d * 2.0 - 1.0;
    let h = inv_proj * vec4f(ndc_xy.x, ndc_xy.y, ndc_z, 1.0);
    let v = h.xyz / max(h.w, 1e-8);
    return v.z;
}

fn overlay_linear_eye_dist(view_z: f32) -> f32 {
    return -view_z;
}

/// MSDF channel combine (median of three channels).
fn median3(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

/// HLSL `fwidth(x) = abs(ddx(x)) + abs(ddy(x))` per scalar component of `uv`.
fn fwidth_uv(uv: vec2f) -> vec2f {
    let fwx = abs(dpdx(uv.x)) + abs(dpdy(uv.x));
    let fwy = abs(dpdx(uv.y)) + abs(dpdy(uv.y));
    return vec2f(fwx, fwy);
}

struct UiTextUnlitMaterialUniform {
    tint_color: vec4f,
    overlay_tint: vec4f,
    outline_color: vec4f,
    background_color: vec4f,
    range_xy: vec4f,
    face_dilate: f32,
    face_softness: f32,
    outline_size: f32,
    pad_scalar: f32,
    rect: vec4f,
    flags: u32,
    pad_flags: u32,
    pad_tail: vec2u,
}

@group(0) @binding(0) var<uniform> uniforms: array<uniform_ring::UniformsSlot, 64>;
@group(1) @binding(0) var scene_depth: texture_depth_2d;
@group(1) @binding(1) var<uniform> overlay_unproject: NativeUiOverlayUnproject;
@group(2) @binding(0) var<uniform> mat: UiTextUnlitMaterialUniform;
@group(2) @binding(1) var font_atlas: texture_2d<f32>;
@group(2) @binding(2) var font_samp: sampler;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
    @location(2) color: vec4f,
    @location(3) aux: vec4f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
    /// Vertex color only; tint is applied in the fragment stage for SDF/MSDF (raster uses vertex color only).
    @location(1) color: vec4f,
    @location(2) local_xy: vec2f,
    /// Matches Unity `extraData` from the NORMAL slot (per-vertex dilate/outline bias).
    @location(3) extra_xy: vec2f,
}

const MODE_RASTER: u32 = 0u;
const MODE_SDF: u32 = 1u;
const MODE_MSDF: u32 = 2u;
const FLAG_OUTLINE: u32 = 256u;
const FLAG_RECTCLIP: u32 = 512u;
const FLAG_OVERLAY: u32 = 1024u;

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.local_xy = in.position.xy;
    out.extra_xy = in.aux.xy;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let mode = mat.flags & 3u;

    if ((mat.flags & FLAG_RECTCLIP) != 0u) {
        if (!ui_common::inside_rect_clip(in.local_xy, mat.rect)) {
            discard;
        }
    }

    let atlas_color = textureSample(font_atlas, font_samp, in.uv);
    var c: vec4f;

    if (mode == MODE_RASTER) {
        c = atlas_color * in.color;
        if (c.a <= 0.001) {
            discard;
        }
    } else {
        var sig_dist: f32;
        if (mode == MODE_MSDF) {
            sig_dist = median3(atlas_color.r, atlas_color.g, atlas_color.b) - 0.5;
        } else {
            sig_dist = atlas_color.a - 0.5;
        }

        sig_dist += mat.face_dilate + in.extra_xy.x;

        let fw = fwidth_uv(in.uv);
        let inv_fw = vec2f(0.5 / max(fw.x, 1e-8), 0.5 / max(fw.y, 1e-8));
        var anti_aliasing = dot(mat.range_xy.xy, inv_fw);
        anti_aliasing = max(anti_aliasing, 1.0);

        var glyph_lerp = mix(sig_dist * anti_aliasing, sig_dist, mat.face_softness);
        glyph_lerp = clamp(glyph_lerp + 0.5, 0.0, 1.0);

        if (max(glyph_lerp, mat.background_color.a) <= 0.001) {
            discard;
        }

        var fill_color = mat.tint_color * in.color;
        if ((mat.flags & FLAG_OUTLINE) != 0u) {
            let outline_dist = sig_dist - (mat.outline_size + in.extra_xy.y);
            var outline_lerp = mix(outline_dist * anti_aliasing, outline_dist, mat.face_softness);
            outline_lerp = clamp(outline_lerp + 0.5, 0.0, 1.0);
            let outline_src = vec4f(mat.outline_color.rgb, mat.outline_color.a * in.color.a);
            fill_color = mix(outline_src, fill_color, outline_lerp);
        }

        c = mix(mat.background_color * in.color, fill_color, glyph_lerp);
    }

    if ((mat.flags & FLAG_OVERLAY) != 0u) {
        let dims = textureDimensions(scene_depth);
        let px = vec2i(i32(in.clip_position.x), i32(in.clip_position.y));
        let sx = clamp(px.x, 0, i32(dims.x) - 1);
        let sy = clamp(px.y, 0, i32(dims.y) - 1);
        let scene_d = textureLoad(scene_depth, vec2i(sx, sy), 0);
        let part_d = in.clip_position.z;
        let ndc_xy = overlay_ndc_xy(in.clip_position.xy, dims);
        let scene_vz = overlay_view_z_from_depth(scene_d, ndc_xy, overlay_unproject.inv_scene_proj);
        let part_vz = overlay_view_z_from_depth(part_d, ndc_xy, overlay_unproject.inv_ui_proj);
        let scene_eye = overlay_linear_eye_dist(scene_vz);
        let part_eye = overlay_linear_eye_dist(part_vz);
        if (part_eye > scene_eye) {
            c *= mat.overlay_tint;
        }
    }

    return c;
}
