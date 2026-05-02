//! Tangent-space → world-space normal-mapping primitives shared across all PBS materials.
//!
//! Each PBS material file owns its own `sample_normal_world` wrapper because per-material details
//! (single-UV vs multi-UV vs triplanar, dual-sided front_facing flip, detail-mask blending, etc.)
//! legitimately differ. What is duplicated across ~46 files is the inner math: building an
//! orthonormal basis from the geometric world normal, applying the basis to a decoded tangent
//! normal, and (for dual-sided materials) flipping for back faces. Those primitives live here.
//!
//! Tangent-space normal *decoding* (BC3/BC5 swizzle, scale-after-Z reconstruction) lives in
//! [`renderide::normal_decode`]. This module is strictly the basis-construction step and above.
//!
//! Import with `#import renderide::pbs::normal as pnorm`.

#define_import_path renderide::pbs::normal

/// Construct a 3x3 matrix of the tangent, bitangent and normal vectors
fn orthonormal_tbn(n: vec3<f32>, t: vec3<f32>) -> mat3x3<f32> {
    let bitangent = cross(n, t);
    return mat3x3<f32>(t, bitangent, n);
}

/// Apply a tangent-space normal `ts_n` to a geometric world normal `world_n`, returning the
/// perturbed world-space normal. `world_n` is assumed non-zero; it is normalized internally.
fn tangent_to_world(world_n: vec3<f32>, ts_n: vec3<f32>) -> vec3<f32> {
    let n = normalize(world_n);
    let tbn = orthonormal_tbn(n);
    return normalize(tbn * ts_n);
}

/// Flip a normal for back-facing fragments. Dual-sided materials use this so geometry seen from
/// the back side still receives lighting consistent with its visible orientation.
fn flip_for_backface(n: vec3<f32>, front_facing: bool) -> vec3<f32> {
    return select(-n, n, front_facing);
}
