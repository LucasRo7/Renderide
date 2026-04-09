//! Builds a reverse-Z Hi-Z pyramid (max reduction) from a depth attachment into a linear `f32` buffer.
//!
//! Level 0 is reduced from full-resolution depth to a base size (≤256 on the long edge). Each
//! subsequent mip stores the max of a 2×2 neighborhood (closest surface in reverse-Z).

struct DepthToBaseParams {
    depth_width: u32,
    depth_height: u32,
    base_width: u32,
    base_height: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var<uniform> depth_params: DepthToBaseParams;
@group(0) @binding(2) var<storage, read_write> pyramid: array<f32>;

@compute @workgroup_size(8, 8)
fn depth_to_base(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let bw = depth_params.base_width;
    let bh = depth_params.base_height;
    if (ox >= bw || oy >= bh) {
        return;
    }
    let dw = depth_params.depth_width;
    let dh = depth_params.depth_height;
    let x0 = u32(i32(ox) * i32(dw) / i32(bw));
    let y0 = u32(i32(oy) * i32(dh) / i32(bh));
    let x1 = u32((i32(ox) + 1) * i32(dw) / i32(bw));
    let y1 = u32((i32(oy) + 1) * i32(dh) / i32(bh));
    var best = 0.0;
    for (var ty = y0; ty < y1; ty = ty + 1u) {
        for (var tx = x0; tx < x1; tx = tx + 1u) {
            let d = textureLoad(depth_tex, vec2<i32>(i32(tx), i32(ty)), 0);
            best = max(best, d);
        }
    }
    let idx = oy * bw + ox;
    pyramid[idx] = best;
}

struct DownsampleParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    src_offset: u32,
    dst_offset: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> ds_params: DownsampleParams;
@group(0) @binding(1) var<storage, read_write> pyramid_rw: array<f32>;

@compute @workgroup_size(8, 8)
fn downsample_mip(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let dw = ds_params.dst_width;
    let dh = ds_params.dst_height;
    if (ox >= dw || oy >= dh) {
        return;
    }
    let sw = ds_params.src_width;
    let sh = ds_params.src_height;
    let sx0 = ox * 2u;
    let sy0 = oy * 2u;
    let sx1 = min(sx0 + 1u, sw - 1u);
    let sy1 = min(sy0 + 1u, sh - 1u);
    let off = ds_params.src_offset;
    let a = pyramid_rw[off + sy0 * sw + sx0];
    let b = pyramid_rw[off + sy0 * sw + sx1];
    let c = pyramid_rw[off + sy1 * sw + sx0];
    let d = pyramid_rw[off + sy1 * sw + sx1];
    let best = max(max(a, b), max(c, d));
    let dst = ds_params.dst_offset + oy * dw + ox;
    pyramid_rw[dst] = best;
}
