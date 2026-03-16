//! Texture asset type. Stub for future Texture2D support.
//!
//! Extension point for texture upload, sampling.

use super::Asset;
use super::AssetId;

/// Stored texture data for GPU upload.
/// Stub for future implementation; mirrors Unity's Texture2DAsset.
pub struct TextureAsset {
    /// Unique identifier for this texture.
    pub id: AssetId,
}

impl Asset for TextureAsset {
    fn id(&self) -> AssetId {
        self.id
    }
}
