//! VR desktop mirror: copy one HMD eye into a staging texture, then blit to the window surface.
//!
//! The surface blit uses **cover** (fill) mapping: the window is filled with a uniform scale of the
//! staging texture; aspect mismatch is resolved by cropping the center (no letterboxing).
//!
//! Used instead of a second full world render when OpenXR multiview has already drawn the scene.

mod cover;
mod pipelines;
mod resources;

/// OpenXR `PRIMARY_STEREO` layer index used for the desktop mirror (left eye).
pub const VR_MIRROR_EYE_LAYER: u32 = 0;

pub use resources::VrMirrorBlitResources;
