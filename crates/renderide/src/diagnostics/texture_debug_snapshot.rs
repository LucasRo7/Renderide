//! Read-only snapshot of GPU texture pool entries for the **Textures** debug HUD.

use crate::resources::TexturePool;
use crate::shared::{ColorProfile, TextureFilterMode, TextureFormat, TextureWrapMode};

/// Per-texture row in the debug HUD: asset id, dimensions, host/GPU formats, mip counts, sampler.
#[derive(Clone, Debug)]
pub struct TextureDebugRow {
    /// Host texture asset id.
    pub asset_id: i32,
    /// Width of mip 0 in texels.
    pub width: u32,
    /// Height of mip 0 in texels.
    pub height: u32,
    /// Total mip levels allocated for the texture.
    pub mip_levels_total: u32,
    /// Mip levels with authored texels uploaded.
    pub mip_levels_resident: u32,
    /// Estimated resident GPU bytes for the allocated texture.
    pub resident_bytes: u64,
    /// Host texture format enum.
    pub host_format: TextureFormat,
    /// Resolved wgpu texture format.
    pub wgpu_format: wgpu::TextureFormat,
    /// Host color profile for linear or sRGB sampling.
    pub color_profile: ColorProfile,
    /// Host sampler filter mode.
    pub filter_mode: TextureFilterMode,
    /// Host anisotropy level.
    pub aniso_level: i32,
    /// U address mode.
    pub wrap_u: TextureWrapMode,
    /// V address mode.
    pub wrap_v: TextureWrapMode,
    /// Host mipmap bias value.
    pub mipmap_bias: f32,
}

/// Per-frame snapshot of every resident texture for the **Textures** ImGui window.
#[derive(Clone, Debug, Default)]
pub struct TextureDebugSnapshot {
    /// Texture rows sorted by host asset id.
    pub rows: Vec<TextureDebugRow>,
    /// Sum of [`TextureDebugRow::resident_bytes`] across all rows.
    pub total_resident_bytes: u64,
}

impl TextureDebugSnapshot {
    /// Collects a row per resident texture, sorted by asset id for stable listing.
    pub fn capture(pool: &TexturePool) -> Self {
        let mut rows: Vec<TextureDebugRow> = pool
            .textures()
            .values()
            .map(|t| TextureDebugRow {
                asset_id: t.asset_id,
                width: t.width,
                height: t.height,
                mip_levels_total: t.mip_levels_total,
                mip_levels_resident: t.mip_levels_resident,
                resident_bytes: t.resident_bytes,
                host_format: t.host_format,
                wgpu_format: t.wgpu_format,
                color_profile: t.color_profile,
                filter_mode: t.sampler.filter_mode,
                aniso_level: t.sampler.aniso_level,
                wrap_u: t.sampler.wrap_u,
                wrap_v: t.sampler.wrap_v,
                mipmap_bias: t.sampler.mipmap_bias,
            })
            .collect();
        rows.sort_by_key(|r| r.asset_id);
        let total_resident_bytes = rows.iter().map(|r| r.resident_bytes).sum();
        Self {
            rows,
            total_resident_bytes,
        }
    }
}
