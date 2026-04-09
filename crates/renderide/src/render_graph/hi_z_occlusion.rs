//! CPU-side Hi-Z sampling for temporal occlusion tests (reverse-Z depth).
//!
//! The GPU pyramid stores **maximum** depth per tile (closest surface in reverse-Z). An object is
//! conservatively occluded when its closest point on the AABB in **previous** clip space is farther
//! than the Hi-Z value (smaller depth value in reverse-Z).

use std::collections::HashMap;

use glam::{Mat4, Vec3, Vec4};

use crate::scene::RenderSpaceId;

use super::view_matrix_from_render_transform;
use super::world_mesh_cull::WorldMeshCullProjParams;

/// One mip level of the Hi-Z pyramid (row-major, width × height).
#[derive(Clone, Debug, Default)]
pub struct HiZCpuSnapshot {
    /// `mips[0]` is the coarsest base level; each following level is half-sized (rounded up).
    pub mips: Vec<Vec<f32>>,
    pub base_width: u32,
    pub base_height: u32,
    /// `true` when pyramid data matches the previous frame’s depth and culling may use it.
    pub valid: bool,
}

impl HiZCpuSnapshot {
    /// Nearest-texel sample at UV in [0,1]² with **v** = 0 at the top row (matching depth viewport).
    pub fn sample_max_depth(&self, uv: (f32, f32), mip_level: u32) -> Option<f32> {
        if !self.valid || self.mips.is_empty() {
            return None;
        }
        let level = (mip_level as usize).min(self.mips.len().saturating_sub(1));
        let mip = self.mips.get(level)?;
        let w = level_width(self.base_width, level as u32);
        let h = level_height(self.base_height, level as u32);
        if w == 0 || h == 0 {
            return None;
        }
        let u = uv.0.clamp(0.0, 1.0);
        let v = uv.1.clamp(0.0, 1.0);
        let x = ((u * w as f32).floor() as u32).min(w - 1);
        let y = ((v * h as f32).floor() as u32).min(h - 1);
        mip.get((y * w + x) as usize).copied()
    }
}

/// Camera and per-space view state from the **frame that produced** the Hi-Z depth buffer.
#[derive(Clone, Debug)]
pub struct HiZTemporalState {
    pub snapshot: HiZCpuSnapshot,
    pub prev_cull: WorldMeshCullProjParams,
    pub prev_view_by_space: HashMap<RenderSpaceId, Mat4>,
    /// Viewport dimensions of the depth texture used to build Hi-Z (for mip footprint).
    pub depth_viewport_px: (u32, u32),
}

/// Builds base dimensions (long edge ≤ 256) matching [`crate::gpu::hi_z::hi_z_base_dimensions`].
pub fn hi_z_base_dimensions(depth_w: u32, depth_h: u32) -> (u32, u32) {
    let max_dim = depth_w.max(depth_h).max(1);
    let scale = max_dim.div_ceil(256).max(1);
    let bw = depth_w.div_ceil(scale).max(1);
    let bh = depth_h.div_ceil(scale).max(1);
    (bw, bh)
}

fn level_width(base: u32, level: u32) -> u32 {
    let mut w = base;
    for _ in 0..level {
        w = w.div_ceil(2).max(1);
    }
    w.max(1)
}

fn level_height(base: u32, level: u32) -> u32 {
    let mut h = base;
    for _ in 0..level {
        h = h.div_ceil(2).max(1);
    }
    h.max(1)
}

/// Total `f32` count for a full pyramid.
pub fn hi_z_total_floats(base_w: u32, base_h: u32) -> usize {
    let mut total = 0usize;
    let mut cw = base_w.max(1);
    let mut ch = base_h.max(1);
    loop {
        total += (cw as usize) * (ch as usize);
        if cw <= 1 && ch <= 1 {
            break;
        }
        cw = cw.div_ceil(2);
        ch = ch.div_ceil(2);
    }
    total
}

/// Byte offsets (in `f32` elements) for each mip level.
pub fn hi_z_mip_offsets(base_w: u32, base_h: u32) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut o = 0usize;
    let mut cw = base_w.max(1);
    let mut ch = base_h.max(1);
    loop {
        offsets.push(o);
        o += (cw as usize) * (ch as usize);
        if cw <= 1 && ch <= 1 {
            break;
        }
        cw = cw.div_ceil(2);
        ch = ch.div_ceil(2);
    }
    offsets
}

/// Decodes a linear GPU buffer into [`HiZCpuSnapshot`].
pub fn hi_z_decode_pyramid_buffer(
    data: &[f32],
    base_w: u32,
    base_h: u32,
) -> Option<HiZCpuSnapshot> {
    let expected = hi_z_total_floats(base_w, base_h);
    if data.len() < expected {
        return None;
    }
    let offsets = hi_z_mip_offsets(base_w, base_h);
    let mut mips = Vec::with_capacity(offsets.len());
    for (i, &off) in offsets.iter().enumerate() {
        let w = level_width(base_w, i as u32);
        let h = level_height(base_h, i as u32);
        let len = (w as usize) * (h as usize);
        let end = off.checked_add(len)?;
        mips.push(data.get(off..end)?.to_vec());
    }
    Some(HiZCpuSnapshot {
        mips,
        base_width: base_w,
        base_height: base_h,
        valid: true,
    })
}

/// Picks a mip level from an approximate pixel footprint (diameter) on the **base** Hi-Z grid.
pub fn hi_z_mip_for_pixel_extent(extent_px: f32) -> u32 {
    if !extent_px.is_finite() || extent_px <= 1.0 {
        return 0;
    }
    let mut e = extent_px.max(1.0);
    let mut level = 0u32;
    while e > 2.0 && level < 15 {
        e *= 0.5;
        level += 1;
    }
    level
}

/// View–projection matrices for Hi-Z (same rules as frustum culling, using **previous** frame data).
pub fn hi_z_view_proj_matrices(
    prev: &WorldMeshCullProjParams,
    prev_view: Mat4,
    is_overlay: bool,
) -> Vec<Mat4> {
    if let Some((sl, sr)) = prev.vr_stereo {
        if is_overlay {
            return vec![prev.overlay_proj * prev_view];
        }
        return vec![sl, sr];
    }
    let base = if is_overlay {
        prev.overlay_proj
    } else {
        prev.world_proj
    };
    vec![base * prev_view]
}

const HI_Z_BIAS: f32 = 5e-5;

/// Returns `true` if the mesh draw should be **culled** by Hi-Z (fully occluded in **all** relevant VPs).
#[allow(clippy::too_many_arguments)]
pub fn mesh_fully_occluded_hi_z(
    world_mins: Vec3,
    world_maxs: Vec3,
    temporal: &HiZTemporalState,
    space_id: RenderSpaceId,
    is_overlay: bool,
) -> bool {
    if !temporal.snapshot.valid || temporal.snapshot.mips.is_empty() {
        return false;
    }
    let Some(prev_view) = temporal.prev_view_by_space.get(&space_id).copied() else {
        return false;
    };

    let corners = aabb_corners(world_mins, world_maxs);
    let vps = hi_z_view_proj_matrices(&temporal.prev_cull, prev_view, is_overlay);
    if vps.is_empty() {
        return false;
    }

    let vw = temporal.depth_viewport_px.0.max(1) as f32;
    let vh = temporal.depth_viewport_px.1.max(1) as f32;

    for vp in &vps {
        if !aabb_occluded_in_vp(&corners, vp, &temporal.snapshot, vw, vh) {
            return false;
        }
    }
    true
}

fn aabb_occluded_in_vp(
    corners: &[Vec4; 8],
    vp: &Mat4,
    snapshot: &HiZCpuSnapshot,
    vw: f32,
    vh: f32,
) -> bool {
    let mut min_ndc_z = f32::MAX;
    let mut max_ndc_x = f32::MIN;
    let mut min_ndc_x = f32::MAX;
    let mut max_ndc_y = f32::MIN;
    let mut min_ndc_y = f32::MAX;
    let mut any_in_front = false;

    for c in corners {
        let clip = *vp * *c;
        let aw = clip.w.abs();
        if aw < 1e-8 || !aw.is_finite() {
            continue;
        }
        if clip.w <= 0.0 {
            return false;
        }
        any_in_front = true;
        let inv_w = 1.0 / clip.w;
        let ndc_x = clip.x * inv_w;
        let ndc_y = clip.y * inv_w;
        let ndc_z = clip.z * inv_w;
        min_ndc_z = min_ndc_z.min(ndc_z);
        max_ndc_x = max_ndc_x.max(ndc_x);
        min_ndc_x = min_ndc_x.min(ndc_x);
        max_ndc_y = max_ndc_y.max(ndc_y);
        min_ndc_y = min_ndc_y.min(ndc_y);
    }

    if !any_in_front {
        return false;
    }

    let u0 = min_ndc_x * 0.5 + 0.5;
    let u1 = max_ndc_x * 0.5 + 0.5;
    let v0 = 1.0 - (max_ndc_y * 0.5 + 0.5);
    let v1 = 1.0 - (min_ndc_y * 0.5 + 0.5);
    let du = (u1 - u0).abs() * vw;
    let dv = (v1 - v0).abs() * vh;
    let extent = du.max(dv).max(1.0);
    let mip = hi_z_mip_for_pixel_extent(extent).min(snapshot.mips.len().saturating_sub(1) as u32);

    let uc = ((u0 + u1) * 0.5).clamp(0.0, 1.0);
    let vc = ((v0 + v1) * 0.5).clamp(0.0, 1.0);
    let Some(hiz) = snapshot.sample_max_depth((uc, vc), mip) else {
        return false;
    };

    let object_closest = min_ndc_z;
    object_closest + HI_Z_BIAS < hiz
}

fn aabb_corners(min: Vec3, max: Vec3) -> [Vec4; 8] {
    [
        Vec4::new(min.x, min.y, min.z, 1.0),
        Vec4::new(max.x, min.y, min.z, 1.0),
        Vec4::new(min.x, max.y, min.z, 1.0),
        Vec4::new(max.x, max.y, min.z, 1.0),
        Vec4::new(min.x, min.y, max.z, 1.0),
        Vec4::new(max.x, min.y, max.z, 1.0),
        Vec4::new(min.x, max.y, max.z, 1.0),
        Vec4::new(max.x, max.y, max.z, 1.0),
    ]
}

/// Updates [`HiZTemporalState::prev_view_by_space`] from the current scene (call after depth is finalized).
pub fn hi_z_capture_prev_views(scene: &crate::scene::SceneCoordinator, out: &mut HiZTemporalState) {
    out.prev_view_by_space.clear();
    for id in scene.render_space_ids() {
        let Some(space) = scene.space(id) else {
            continue;
        };
        let v = view_matrix_from_render_transform(&space.view_transform);
        out.prev_view_by_space.insert(id, v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hi_z_base_dimensions_scales_down_large_viewports() {
        let (w, h) = hi_z_base_dimensions(512, 256);
        assert_eq!((w, h), (256, 128));
        let (w2, h2) = hi_z_base_dimensions(256, 256);
        assert_eq!((w2, h2), (256, 256));
    }

    #[test]
    fn mip_offsets_sum_matches_total_floats() {
        let bw = 4u32;
        let bh = 4u32;
        let offsets = hi_z_mip_offsets(bw, bh);
        let total = hi_z_total_floats(bw, bh);
        assert_eq!(offsets.len(), 3);
        let last_off = offsets[2];
        let w2 = super::level_width(bw, 2);
        let h2 = super::level_height(bh, 2);
        assert_eq!(last_off + w2 as usize * h2 as usize, total);
    }

    #[test]
    fn decode_roundtrip_preserves_layout() {
        let bw = 2u32;
        let bh = 2u32;
        let total = hi_z_total_floats(bw, bh);
        let mut data = vec![0.0f32; total];
        for i in 0..total {
            data[i] = i as f32 * 0.1 + 0.5;
        }
        let snap = hi_z_decode_pyramid_buffer(&data, bw, bh).expect("decode");
        assert!(snap.valid);
        assert_eq!(snap.base_width, bw);
        assert_eq!(snap.base_height, bh);
        assert_eq!(snap.mips.len(), 2);
        assert_eq!(snap.mips[0].len(), 4);
        assert_eq!(snap.mips[1].len(), 1);
        assert!((snap.sample_max_depth((0.0, 0.0), 0).unwrap() - data[0]).abs() < 1e-6);
    }
}
