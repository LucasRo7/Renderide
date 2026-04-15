//! Reverse-Z projections and host clip-plane helpers for desktop and OpenXR.

use glam::{Mat4, Vec3, Vec4};
use openxr::Fovf;

use crate::shared::HeadOutputDevice;

use super::view::filter_scale_legacy;

/// Minimum desktop vertical FOV in **degrees** after clamping.
///
/// Mirrors a small positive host lower bound so `tan(fov/2)` stays finite and non-zero.
pub const DESKTOP_FOV_DEGREES_MIN: f32 = 1e-4;

/// Maximum desktop vertical FOV in **degrees** after clamping (non-inclusive of 180° degeneracy).
pub const DESKTOP_FOV_DEGREES_MAX: f32 = 179.0;

/// Default fallback when the host sends non-finite FOV (matches [`crate::render_graph::frame_params::HostCameraFrame::default`]).
pub(crate) const DEFAULT_DESKTOP_FOV_DEGREES: f32 = 60.0;

/// Clamps host `desktopFOV` to a sane range before perspective projection.
///
/// [`f32::NAN`] falls back to [`DEFAULT_DESKTOP_FOV_DEGREES`]. Infinities clamp to the min/max
/// bounds like any other out-of-range value.
pub fn clamp_desktop_fov_degrees(degrees: f32) -> f32 {
    if degrees.is_nan() {
        DEFAULT_DESKTOP_FOV_DEGREES
    } else {
        degrees.clamp(DESKTOP_FOV_DEGREES_MIN, DESKTOP_FOV_DEGREES_MAX)
    }
}

/// Clip-plane adjustment derived from head output device and root scale (Unity-style parity).
pub fn effective_head_output_clip_planes(
    near_clip: f32,
    far_clip: f32,
    output_device: HeadOutputDevice,
    root_scale: Option<Vec3>,
) -> (f32, f32) {
    let near_min = if output_device == HeadOutputDevice::Screen360 {
        0.25
    } else {
        0.001
    };
    let filtered_root_scale = filter_scale_legacy(root_scale.unwrap_or(Vec3::ONE));
    (
        near_clip.max(near_min) * filtered_root_scale.x,
        far_clip.max(0.5) * filtered_root_scale.x,
    )
}

/// Reverse-Z perspective projection (column-major [`Mat4`], same coefficients as the historical nalgebra path).
///
/// * `vertical_fov` — vertical field of view in **radians**
/// * `near` / `far` — positive distances (`far > near`)
pub fn reverse_z_perspective(aspect: f32, vertical_fov: f32, near: f32, far: f32) -> Mat4 {
    let vertical_half = vertical_fov / 2.0;
    let tan_vertical_half = vertical_half.tan();
    let horizontal_fov = (tan_vertical_half * aspect)
        .atan()
        .clamp(0.1_f32, std::f32::consts::FRAC_PI_2 - 0.1)
        * 2.0;
    let tan_horizontal_half = (horizontal_fov / 2.0).tan();
    let f_x = 1.0 / tan_horizontal_half;
    let f_y = 1.0 / tan_vertical_half;
    reverse_z_perspective_from_scales(f_x, f_y, 0.0, 0.0, near, far)
}

/// Reverse-Z perspective with optional **off-center** (asymmetric) X/Y skew from OpenXR tangents.
///
/// `skew_x` / `skew_y` are `(tan_right + tan_left) / (tan_right - tan_left)` and
/// `(tan_up + tan_down) / (tan_up - tan_down)` on the **Z basis column** so clip X/Y depend on view-space Z.
fn reverse_z_perspective_from_scales(
    x_scale: f32,
    y_scale: f32,
    skew_x: f32,
    skew_y: f32,
    near: f32,
    far: f32,
) -> Mat4 {
    let z2 = near / (far - near);
    let z3 = (far * near) / (far - near);
    Mat4::from_cols(
        Vec4::new(x_scale, 0.0, 0.0, 0.0),
        Vec4::new(0.0, y_scale, 0.0, 0.0),
        Vec4::new(skew_x, skew_y, z2, -1.0),
        Vec4::new(0.0, 0.0, z3, 0.0),
    )
}

/// Asymmetric reverse-Z projection from OpenXR [`Fovf`] tangents (Khronos `XrMatrix4x4f_CreateProjectionFov` X/Y,
/// with the same reverse-Z depth row as [`reverse_z_perspective`]).
///
/// View space matches the renderer: **right-handed**, **−Z** forward, **+Y** up.
pub fn reverse_z_perspective_openxr_fov(fov: &Fovf, near: f32, far: f32) -> Mat4 {
    let tl = fov.angle_left.tan();
    let tr = fov.angle_right.tan();
    let td = fov.angle_down.tan();
    let tu = fov.angle_up.tan();
    let w = tr - tl;
    let h = tu - td;
    if !(w.is_finite() && h.is_finite()) || w.abs() < 1e-6 || h.abs() < 1e-6 {
        logger::trace!(
            "OpenXR FOV degenerate; using symmetric fallback (16:9, 45° vertical). raw angles rad: left={:.4} right={:.4} down={:.4} up={:.4} w={w} h={h}",
            fov.angle_left,
            fov.angle_right,
            fov.angle_down,
            fov.angle_up
        );
        let aspect = 16.0 / 9.0;
        let vertical_fov = std::f32::consts::FRAC_PI_2 * 0.5;
        return reverse_z_perspective(aspect, vertical_fov, near, far);
    }
    let x_scale = 2.0 / w;
    let y_scale = 2.0 / h;
    let skew_x = (tr + tl) / w;
    let skew_y = (tu + td) / h;
    reverse_z_perspective_from_scales(x_scale, y_scale, skew_x, skew_y, near, far)
}

/// Reverse-Z orthographic projection (`half_width`, `half_height` in view space).
pub fn reverse_z_orthographic(half_width: f32, half_height: f32, near: f32, far: f32) -> Mat4 {
    let range = far - near;
    let z_scale = -2.0 / range;
    let z_offset = (far + near) / range;
    Mat4::from_cols(
        Vec4::new(1.0 / half_width, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0 / half_height, 0.0, 0.0),
        Vec4::new(0.0, 0.0, z_scale, 0.0),
        Vec4::new(0.0, 0.0, z_offset, 1.0),
    )
}
