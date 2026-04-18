//! Transactional allocation of `@group(0)` frame resources, empty `@group(1)`, and `@group(2)` per-draw slab.
//!
//! [`FrameGpuBindings::try_new`] succeeds only when all three are created, avoiding a partially wired
//! frame bind set.

use std::sync::Arc;

use crate::gpu::GpuLimits;
use crate::materials::PipelineBuildError;
use thiserror::Error;

use super::frame_gpu::{EmptyMaterialBindGroup, FrameGpuResources};
use super::frame_gpu_error::FrameGpuInitError;
use super::per_draw_resources::PerDrawResources;

/// Either frame globals failed to allocate, or the per-draw slab/bind group could not be built.
#[derive(Debug, Error)]
pub enum FrameGpuBindingsError {
    /// `@group(0)` frame buffers / cluster bootstrap failed.
    #[error(transparent)]
    FrameGpuInit(#[from] FrameGpuInitError),
    /// `@group(2)` per-draw layout or buffer creation failed.
    #[error(transparent)]
    PipelineBuild(#[from] PipelineBuildError),
}

/// All mesh-forward frame bind resources allocated together ([`FrameGpuBindings::try_new`]).
pub struct FrameGpuBindings {
    /// Camera + lights (`@group(0)`).
    pub frame_gpu: FrameGpuResources,
    /// Fallback material (`@group(1)`).
    pub empty_material: EmptyMaterialBindGroup,
    /// Per-draw instance storage (`@group(2)`).
    pub per_draw: PerDrawResources,
}

impl FrameGpuBindings {
    /// Allocates frame globals, empty material bind group, and per-draw storage in one step.
    ///
    /// On error, nothing is returned; callers must not treat any partial state as attached.
    pub fn try_new(
        device: &wgpu::Device,
        limits: Arc<GpuLimits>,
    ) -> Result<Self, FrameGpuBindingsError> {
        let frame_gpu = FrameGpuResources::new(device, Arc::clone(&limits))?;
        let empty_material = EmptyMaterialBindGroup::new(device);
        let per_draw = PerDrawResources::new(device, limits)?;
        Ok(Self {
            frame_gpu,
            empty_material,
            per_draw,
        })
    }
}
