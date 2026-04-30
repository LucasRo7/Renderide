//! GPU and host-facing resource layer: pools, material tables, uploads, preprocess pipelines.
//!
//! This module owns **wgpu** [`wgpu::Device`] / [`wgpu::Queue`], mesh and texture pools, the
//! [`MaterialPropertyStore`](crate::assets::material::MaterialPropertyStore), the compiled
//! [`CompiledRenderGraph`](crate::render_graph::CompiledRenderGraph) after attach, and code paths
//! that turn shared-memory asset payloads into resident GPU resources. [`light_gpu`](crate::backend::light_gpu)
//! packs scene [`ResolvedLight`](crate::scene::ResolvedLight) values for future storage-buffer upload. It does **not**
//! own IPC queues, [`SharedMemoryAccessor`](crate::ipc::SharedMemoryAccessor), or scene graph state;
//! callers pass those in where a command requires both transport and GPU work.

mod cluster_gpu;
mod debug_hud_bundle;
mod embedded;
mod frame_gpu;
mod frame_gpu_bindings;
mod frame_gpu_error;
mod frame_resource_manager;
mod gpu_jobs;
mod history_registry;
mod light_gpu;
mod material_property_reader;
mod material_system;
pub mod mesh_deform;
mod per_draw_resources;
mod per_view_resource_map;
mod reflection_probe_sh2;
mod render_backend;
mod skybox_environment;
mod skybox_params;
mod skybox_specular;
mod view_resource_registry;

pub use crate::assets::AssetTransferQueue;
pub(crate) use crate::occlusion::HiZBuildInput;
pub use crate::occlusion::OcclusionSystem;
pub use cluster_gpu::{
    CLUSTER_COUNT_Z, CLUSTER_PARAMS_UNIFORM_SIZE, ClusterBufferCache, ClusterBufferRefs,
    MAX_LIGHTS_PER_TILE, TILE_SIZE,
};
pub use debug_hud_bundle::DebugHudBundle;
pub use embedded::{
    EmbeddedMaterialBindError, EmbeddedMaterialBindResources, EmbeddedTexturePools,
};
pub use frame_gpu::{
    EmptyMaterialBindGroup, FrameGpuResources, FrameSceneSnapshotTextureViews,
    empty_material_bind_group_layout,
};
pub use frame_gpu_bindings::{FrameGpuBindings, FrameGpuBindingsError};
pub use frame_gpu_error::FrameGpuInitError;
pub use frame_resource_manager::{
    FrameGpuBindContext, FrameResourceManager, PerViewFrameState, PreRecordViewResourceLayout,
};
pub(crate) use gpu_jobs::{
    GpuJobResources, GpuReadbackJobs, GpuReadbackOutcomes, GpuSubmitJobTracker, SubmittedGpuJob,
    SubmittedReadbackJob,
};
pub use history_registry::{
    BufferHistorySlot, BufferHistorySpec, HistoryRegistry, HistoryRegistryError,
    HistoryResourceScope, HistoryTexture, HistoryTextureMipViews, TextureHistorySlot,
    TextureHistorySpec,
};
pub use light_gpu::{
    GpuLight, MAX_LIGHTS, order_lights_for_clustered_shading,
    order_lights_for_clustered_shading_in_place,
};
pub use material_system::{MAX_PENDING_MATERIAL_BATCHES, MaterialSystem};
pub use mesh_deform::{
    INITIAL_PER_DRAW_UNIFORM_SLOTS, MeshDeformScratch, MeshPreprocessPipelines,
    PER_DRAW_UNIFORM_STRIDE, PaddedPerDrawUniforms, WgslMat3x3, advance_slab_cursor,
    blendshape_sparse_buffers_fit_device, plan_blendshape_scatter_chunks,
    write_per_draw_uniform_slab,
};
pub use per_draw_resources::PerDrawResources;
pub(crate) use reflection_probe_sh2::ReflectionProbeSh2System;
pub(crate) use render_backend::{ExtractedFrameShared, WorldMeshForwardEncodeRefs};
pub use render_backend::{
    MAX_ASSET_INTEGRATION_QUEUED, MAX_PENDING_MESH_UPLOADS, MAX_PENDING_TEXTURE_UPLOADS,
    RenderBackend, RenderBackendAttachDesc, RenderBackendAttachError,
};
pub(crate) use skybox_environment::SkyboxEnvironmentCache;
pub(crate) use skybox_specular::resolve_active_main_skybox_specular_environment;
pub(crate) use view_resource_registry::ViewResourceRegistry;
