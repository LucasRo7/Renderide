//! Host IPC handlers for asset transfers, material batches, and shader routing (delegates to the asset queue and [`crate::backend::MaterialSystem`]).

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::materials::RasterPipelineKind;
use crate::shared::{
    MaterialsUpdateBatch, MeshUnload, MeshUploadData, SetRenderTextureFormat, SetTexture2DData,
    SetTexture2DFormat, SetTexture2DProperties, UnloadRenderTexture, UnloadTexture2D,
};

use crate::assets::asset_transfer_queue::{self as asset_uploads};

use super::RenderBackend;

impl RenderBackend {
    /// Cooperative mesh/texture uploads ([`crate::runtime::RendererRuntime::run_asset_integration`]):
    /// all high-priority tasks run to completion, then normal-priority work until `normal_deadline`.
    pub fn drain_asset_tasks(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
        normal_deadline: std::time::Instant,
    ) {
        asset_uploads::drain_asset_tasks(&mut self.asset_transfers, shm, ipc, normal_deadline);
    }

    /// Handle [`SetTexture2DFormat`](crate::shared::SetTexture2DFormat).
    pub fn on_set_texture_2d_format(
        &mut self,
        f: SetTexture2DFormat,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_texture_2d_format(&mut self.asset_transfers, f, ipc);
    }

    /// Handle [`SetTexture2DProperties`](crate::shared::SetTexture2DProperties).
    pub fn on_set_texture_2d_properties(
        &mut self,
        p: SetTexture2DProperties,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_texture_2d_properties(&mut self.asset_transfers, p, ipc);
    }

    /// Handle [`SetTexture2DData`](crate::shared::SetTexture2DData). Pass shared memory when available
    /// so mips can be read from the host buffer; if GPU or texture is not ready, data is queued.
    pub fn on_set_texture_2d_data(
        &mut self,
        d: SetTexture2DData,
        shm: Option<&mut SharedMemoryAccessor>,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_texture_2d_data(&mut self.asset_transfers, d, shm, ipc);
    }

    /// Upload texture mips from shared memory and optionally notify the host on the background queue.
    ///
    /// `consume_texture_upload_budget` should be `true` for normal IPC handling; use `false` when
    /// draining deferred uploads or replaying pending uploads on GPU attach.
    pub fn try_texture_upload_with_device(
        &mut self,
        data: SetTexture2DData,
        shm: &mut SharedMemoryAccessor,
        ipc: Option<&mut DualQueueIpc>,
        consume_texture_upload_budget: bool,
    ) {
        asset_uploads::try_texture_upload_with_device(
            &mut self.asset_transfers,
            data,
            shm,
            ipc,
            consume_texture_upload_budget,
        );
    }

    /// Remove a texture asset from CPU tables and the pool.
    pub fn on_unload_texture_2d(&mut self, u: UnloadTexture2D) {
        asset_uploads::on_unload_texture_2d(&mut self.asset_transfers, u);
    }

    /// Handle [`SetRenderTextureFormat`](crate::shared::SetRenderTextureFormat).
    pub fn on_set_render_texture_format(
        &mut self,
        f: SetRenderTextureFormat,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::on_set_render_texture_format(&mut self.asset_transfers, f, ipc);
    }

    /// Handle [`UnloadRenderTexture`](crate::shared::UnloadRenderTexture).
    pub fn on_unload_render_texture(&mut self, u: UnloadRenderTexture) {
        asset_uploads::on_unload_render_texture(&mut self.asset_transfers, u);
    }

    /// Ingest mesh bytes from shared memory; notifies host when `ipc` is set.
    pub fn try_process_mesh_upload(
        &mut self,
        data: MeshUploadData,
        shm: &mut SharedMemoryAccessor,
        ipc: Option<&mut DualQueueIpc>,
    ) {
        asset_uploads::try_process_mesh_upload(&mut self.asset_transfers, data, shm, ipc);
    }

    /// Remove a mesh from the pool.
    pub fn on_mesh_unload(&mut self, u: MeshUnload) {
        asset_uploads::on_mesh_unload(&mut self.asset_transfers, u);
    }

    /// Drain pending material batches using the given shared memory and IPC.
    pub fn flush_pending_material_batches(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut DualQueueIpc,
    ) {
        self.materials.flush_pending_material_batches(shm, ipc);
    }

    /// Queue a materials batch when shared memory is not yet available. Returns `false` if queue full.
    pub fn enqueue_materials_batch_no_shm(&mut self, batch: MaterialsUpdateBatch) -> bool {
        self.materials.enqueue_materials_batch_no_shm(batch)
    }

    /// Apply one host materials batch (shared memory must be valid for the batch descriptors).
    pub fn apply_materials_update_batch(
        &mut self,
        batch: MaterialsUpdateBatch,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut DualQueueIpc,
    ) {
        self.materials.apply_materials_update_batch(batch, shm, ipc);
    }

    /// Remove material / property-block entries from the host store.
    pub fn on_unload_material(&mut self, asset_id: i32) {
        self.materials.on_unload_material(asset_id);
    }

    /// Remove a property block from the host store.
    pub fn on_unload_material_property_block(&mut self, asset_id: i32) {
        self.materials.on_unload_material_property_block(asset_id);
    }

    /// Maps shader asset to raster pipeline kind and optional HUD display name, or defers until [`super::RenderBackend::attach`].
    pub fn register_shader_route(
        &mut self,
        asset_id: i32,
        pipeline: RasterPipelineKind,
        display_name: Option<String>,
    ) {
        self.materials
            .register_shader_route(asset_id, pipeline, display_name);
    }

    /// Removes shader routing for `asset_id`.
    pub fn unregister_shader_route(&mut self, asset_id: i32) {
        self.materials.unregister_shader_route(asset_id);
    }
}
