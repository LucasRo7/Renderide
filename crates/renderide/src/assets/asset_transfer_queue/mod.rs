//! Mesh and Texture2D upload queues, cooperative integration, CPU-side format/property tables, and resident pools.
//!
//! [`AssetTransferQueue`] lives in the [`crate::assets`] module and is owned by
//! [`crate::backend::RenderBackend`]. It handles shared-memory ingestion paths that populate
//! [`crate::resources::MeshPool`] and [`crate::resources::TexturePool`].

mod integrator;
mod mesh_task;
mod texture_task;
mod uploads;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::gpu::GpuLimits;
use crate::resources::{MeshPool, RenderTexturePool, TexturePool};
use crate::shared::{
    MeshUploadData, SetRenderTextureFormat, SetTexture2DData, SetTexture2DFormat,
    SetTexture2DProperties,
};

pub use integrator::{
    drain_asset_tasks, drain_asset_tasks_unbounded, AssetIntegrator, AssetTask, StepResult,
    MAX_ASSET_INTEGRATION_QUEUED,
};
pub use uploads::{
    attach_flush_pending_asset_uploads, on_mesh_unload, on_set_render_texture_format,
    on_set_texture_2d_data, on_set_texture_2d_format, on_set_texture_2d_properties,
    on_unload_render_texture, on_unload_texture_2d, try_process_mesh_upload,
    try_texture_upload_with_device, MAX_PENDING_MESH_UPLOADS, MAX_PENDING_TEXTURE_UPLOADS,
};

/// Pending mesh/texture payloads, CPU texture tables, GPU device/queue, resident pools, and [`AssetIntegrator`].
pub struct AssetTransferQueue {
    /// Resident meshes (upload target).
    pub(crate) mesh_pool: MeshPool,
    /// Resident textures (upload target).
    pub(crate) texture_pool: TexturePool,
    /// Resident host render textures (color + optional depth).
    pub(crate) render_texture_pool: RenderTexturePool,
    /// Latest [`SetRenderTextureFormat`] per asset.
    pub(crate) render_texture_formats: HashMap<i32, SetRenderTextureFormat>,
    /// Latest [`SetTexture2DFormat`] per asset (required before data upload).
    pub(crate) texture_formats: HashMap<i32, SetTexture2DFormat>,
    /// Latest [`SetTexture2DProperties`] per asset (sampler metadata on [`crate::resources::GpuTexture2d`]).
    pub(crate) texture_properties: HashMap<i32, SetTexture2DProperties>,
    /// Bound wgpu device after [`crate::backend::RenderBackend::attach`].
    pub(crate) gpu_device: Option<Arc<wgpu::Device>>,
    /// Submission queue paired with [`Self::gpu_device`].
    pub(crate) gpu_queue: Option<Arc<Mutex<wgpu::Queue>>>,
    /// Effective limits snapshot (set with device on attach).
    pub(crate) gpu_limits: Option<Arc<GpuLimits>>,
    /// Mesh payloads waiting for GPU or shared memory (drained on attach).
    pub(crate) pending_mesh_uploads: VecDeque<MeshUploadData>,
    /// Texture mip payloads waiting for GPU allocation or shared memory.
    pub(crate) pending_texture_uploads: VecDeque<SetTexture2DData>,
    /// Cooperative uploads drained by [`drain_asset_tasks`] / [`drain_asset_tasks_unbounded`].
    pub(crate) integrator: AssetIntegrator,
}

impl AssetTransferQueue {
    pub(crate) fn integrator_mut(&mut self) -> &mut AssetIntegrator {
        &mut self.integrator
    }
}

impl Default for AssetTransferQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetTransferQueue {
    /// Empty pools and tables; no GPU until the backend calls attach.
    pub fn new() -> Self {
        Self {
            mesh_pool: MeshPool::default_pool(),
            texture_pool: TexturePool::default_pool(),
            render_texture_pool: RenderTexturePool::new(),
            render_texture_formats: HashMap::new(),
            texture_formats: HashMap::new(),
            texture_properties: HashMap::new(),
            gpu_device: None,
            gpu_queue: None,
            gpu_limits: None,
            pending_mesh_uploads: VecDeque::new(),
            pending_texture_uploads: VecDeque::new(),
            integrator: AssetIntegrator::default(),
        }
    }
}
