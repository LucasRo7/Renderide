//! Null fallback material: object-space 3D checkerboard.
//!
//! Build emits two targets from this file via [`MULTIVIEW`](https://docs.rs/naga_oil) shader defs:
//! - `null_default.wgsl` — `MULTIVIEW` off (single-view desktop)
//! - `null_multiview.wgsl` — `MULTIVIEW` on (stereo `@builtin(view_index)`)
//!
//! Used when the host shader has no embedded target or pipeline build fails.
//! Object-space projection (not UV-based) so the pattern is visible regardless
//! of mesh UV quality — mirrors `Null.shader` in `Resonite.UnityShaders`.
//!
//! Imports `renderide::globals` so composed targets declare the full `@group(0)`
//! frame bind layout that the renderer enforces in reflection; `retain_globals_additive`
//! keeps each binding referenced after naga-oil import pruning.
//! [`PerDrawUniforms`] lives in [`renderide::per_draw`].

#import renderide::globals as rg
#import renderide::per_draw as pd

/// Vertex-to-fragment payload: clip-space position and object-space position used to derive the checker pattern.
struct VertexOutput {
    /// Clip-space position consumed by the rasterizer.
    @builtin(position) clip_pos: vec4<f32>,
    /// Object-space position of the vertex; the fragment stage projects it into the checker grid.
    @location(0) local_pos: vec3<f32>,
}

/// Cells per object-space unit used to space the checker grid.
const CHECKER_SCALE: f32 = 5.0;
/// Dark cell color (sRGB linear). Not pure black so the surface still has subtle shading in dark scenes.
const COLOR_DARK: vec3<f32> = vec3<f32>(0.02, 0.02, 0.02);
/// Light cell color (sRGB linear). Mid-grey so the fallback is clearly distinct from real materials.
const COLOR_LIGHT: vec3<f32> = vec3<f32>(0.35, 0.35, 0.35);

/// Vertex stage: project to clip space and forward object-space position for the checker projection.
@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = d.view_proj_left;
    } else {
        vp = d.view_proj_right;
    }
#else
    let vp = d.view_proj_left;
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.local_pos = pos.xyz;
    return out;
}

/// Fragment stage: select the dark or light cell color from the parity of the object-space cell index.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let cell = floor(in.local_pos * CHECKER_SCALE);
    let parity = (i32(cell.x) + i32(cell.y) + i32(cell.z)) & 1;
    let c = select(COLOR_DARK, COLOR_LIGHT, parity == 0);
    return rg::retain_globals_additive(vec4<f32>(c, 1.0));
}
