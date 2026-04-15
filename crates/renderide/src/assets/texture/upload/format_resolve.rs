//! Maps host [`SetTexture2DFormat`](crate::shared::SetTexture2DFormat) to [`wgpu::TextureFormat`] for new textures.
//!
//! **BC3 / DXT5:** [`TextureFormat::BC3`](crate::shared::TextureFormat::BC3) is included in
//! [`crate::assets::texture::decode::needs_rgba8_decode_before_upload`], so 2D uploads always resolve
//! to an RGBA8 family format and go through CPU [`decode_mip_to_rgba8`](crate::assets::texture::decode::decode_mip_to_rgba8),
//! where **BC3nm** (normal-map) DXT5 packing is unswizzled for correct PBS sampling. If BC3 were ever
//! uploaded as native [`wgpu::TextureFormat::Bc3RgbaUnorm`], that path would bypass the fix.

use crate::shared::{ColorProfile, SetCubemapFormat, SetTexture2DFormat, SetTexture3DFormat};

use super::super::decode::needs_rgba8_decode_before_upload;
use super::super::format::pick_wgpu_storage_format;

/// Decides GPU storage format for a new 2D texture from host [`SetTexture2DFormat`].
///
/// Uses native compressed/uncompressed `wgpu` formats when supported; falls back to RGBA8 when
/// compression features are missing or the host layout needs swizzle ([`needs_rgba8_decode_before_upload`]).
pub fn resolve_texture2d_wgpu_format(
    device: &wgpu::Device,
    fmt: &SetTexture2DFormat,
) -> wgpu::TextureFormat {
    if needs_rgba8_decode_before_upload(fmt.format) {
        return rgba8_fallback_format(fmt.profile);
    }
    if let Some(f) = pick_wgpu_storage_format(device, fmt.format, fmt.profile) {
        return f;
    }
    rgba8_fallback_format(fmt.profile)
}

fn rgba8_fallback_format(profile: ColorProfile) -> wgpu::TextureFormat {
    match profile {
        ColorProfile::SRGB | ColorProfile::SRGBAlpha => wgpu::TextureFormat::Rgba8UnormSrgb,
        ColorProfile::Linear => wgpu::TextureFormat::Rgba8Unorm,
    }
}

/// Decides GPU storage format for a new 3D texture from host [`SetTexture3DFormat`].
pub fn resolve_texture3d_wgpu_format(
    device: &wgpu::Device,
    fmt: &SetTexture3DFormat,
) -> wgpu::TextureFormat {
    if needs_rgba8_decode_before_upload(fmt.format) {
        return rgba8_fallback_format(fmt.profile);
    }
    if let Some(f) = pick_wgpu_storage_format(device, fmt.format, fmt.profile) {
        return f;
    }
    rgba8_fallback_format(fmt.profile)
}

/// Decides GPU storage format for a new cubemap from host [`SetCubemapFormat`].
pub fn resolve_cubemap_wgpu_format(
    device: &wgpu::Device,
    fmt: &SetCubemapFormat,
) -> wgpu::TextureFormat {
    if needs_rgba8_decode_before_upload(fmt.format) {
        return rgba8_fallback_format(fmt.profile);
    }
    if let Some(f) = pick_wgpu_storage_format(device, fmt.format, fmt.profile) {
        return f;
    }
    rgba8_fallback_format(fmt.profile)
}
