//! Tangent-space normal map decoding (RGB normal maps, optional white-placeholder handling).
//!
//! Import with `#import renderide::normal_decode as nd` (or another alias; avoid `as uv`, which naga-oil rejects).

#define_import_path renderide::normal_decode

/// Decode a tangent-space normal from an RGB normal map sample (standard path).
fn decode_ts_normal(raw: vec3<f32>, scale: f32) -> vec3<f32> {
    let nm_xy = (raw.xy * 2.0 - 1.0) * scale;
    let z = max(sqrt(max(1.0 - dot(nm_xy, nm_xy), 0.0)), 1e-6);
    return normalize(vec3<f32>(nm_xy, z));
}

/// Same as [`decode_ts_normal`], but treat an all-white sample as flat +Z (renderer placeholder texture).
fn decode_ts_normal_with_placeholder(raw: vec3<f32>, scale: f32) -> vec3<f32> {
    if (all(raw > vec3<f32>(0.99, 0.99, 0.99))) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    let nm_xy = (raw.xy * 2.0 - 1.0) * scale;
    let z = max(sqrt(max(1.0 - dot(nm_xy, nm_xy), 0.0)), 1e-6);
    return normalize(vec3<f32>(nm_xy, z));
}
