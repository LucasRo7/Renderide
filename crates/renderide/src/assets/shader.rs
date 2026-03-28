//! Shader asset type for host shader uploads and logical name resolution for pipeline routing.
//!
//! Filled by [`super::AssetRegistry::handle_shader_upload`].

use super::Asset;
use super::AssetId;

/// Stored shader data for pipeline creation.
pub struct ShaderAsset {
    /// Unique identifier for this shader.
    pub id: AssetId,
    /// Raw contents of the host [`crate::shared::ShaderUpload::file`] field: filesystem path, logical stem
    /// label, or legacy inline ShaderLab / WGSL text — not necessarily WGSL source.
    ///
    /// [`crate::assets::host_shader_router`] and [`super::ui_material_contract`] use this as a **path or
    /// label hint** together with [`Self::unity_shader_name`] for routing (no ShaderLab parse required).
    pub wgsl_source: Option<String>,
    /// Unity ShaderLab logical name (`Shader "UI/Unlit"`) from parsed ShaderLab/WGSL text, file contents,
    /// or an optional host hint when your IPC layer supplies one (see [`crate::shared::shader_upload_extras`]).
    pub unity_shader_name: Option<String>,
}

impl ShaderAsset {
    /// Host upload path, label, or inline payload — same as [`Self::wgsl_source`], for callers that
    /// prefer explicit naming over the historical `wgsl_source` field.
    pub fn upload_file_field(&self) -> Option<&str> {
        self.wgsl_source.as_deref()
    }
}

impl Asset for ShaderAsset {
    fn id(&self) -> AssetId {
        self.id
    }
}
