//! Shared in-memory handle for [`super::types::RendererSettings`].
//!
//! The frame loop reads through this handle every tick; the debug HUD writes through it when
//! the user edits values; saves to disk go through [`super::save`]. Wrapping the settings in
//! `Arc<RwLock<…>>` (rather than handing out clones) means the HUD's edits are immediately
//! visible to the next frame without a propagation step.

use std::sync::Arc;

use super::load::ConfigLoadResult;
use super::types::RendererSettings;

/// Shared handle for the process-wide settings store (read by the frame loop, written by the HUD).
pub type RendererSettingsHandle = Arc<std::sync::RwLock<RendererSettings>>;

/// Builds a [`RendererSettingsHandle`] from post-load settings.
pub fn settings_handle_from(load: &ConfigLoadResult) -> RendererSettingsHandle {
    Arc::new(std::sync::RwLock::new(load.settings.clone()))
}
