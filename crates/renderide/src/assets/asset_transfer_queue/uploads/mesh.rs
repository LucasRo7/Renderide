//! Mesh upload IPC, budgets, and deferred drain.

use std::sync::Arc;
use std::time::Instant;

use crate::assets::mesh::{
    compute_and_validate_mesh_layout, mesh_upload_input_fingerprint, try_upload_mesh_from_raw,
};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{MeshUnload, MeshUploadData, MeshUploadResult, RendererCommand};

use super::super::AssetTransferQueue;
use super::{
    MAX_DEFERRED_MESH_UPLOADS, MAX_DEFERRED_MESH_UPLOADS_DRAIN_PER_POLL, MAX_PENDING_MESH_UPLOADS,
    MESH_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL,
    TEXTURE_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL,
};

/// Resets the per-poll budget for non-high-priority mesh and texture uploads. Call at the start of
/// each [`crate::runtime::RendererRuntime::poll_ipc`].
pub fn begin_ipc_poll_mesh_upload_budget(queue: &mut AssetTransferQueue) {
    queue.mesh_upload_budget_this_poll = MESH_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL;
    queue.texture_upload_budget_this_poll = TEXTURE_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL;
}

/// Processes mesh uploads deferred during the current IPC batch (low-priority overflow).
pub fn drain_deferred_mesh_uploads_after_poll(
    queue: &mut AssetTransferQueue,
    shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
) {
    let Some(device) = queue.gpu_device.clone() else {
        return;
    };
    let mut drained = 0usize;
    if let Some(ipc_ref) = ipc {
        while drained < MAX_DEFERRED_MESH_UPLOADS_DRAIN_PER_POLL {
            let Some(data) = queue.deferred_mesh_uploads.pop_front() else {
                break;
            };
            try_mesh_upload_with_device(&mut *queue, &device, data, shm, Some(ipc_ref), false);
            drained += 1;
        }
    } else {
        while drained < MAX_DEFERRED_MESH_UPLOADS_DRAIN_PER_POLL {
            let Some(data) = queue.deferred_mesh_uploads.pop_front() else {
                break;
            };
            try_mesh_upload_with_device(&mut *queue, &device, data, shm, None, false);
            drained += 1;
        }
    }
}

/// Ingest mesh bytes from shared memory; notifies host when `ipc` is set.
pub fn try_process_mesh_upload(
    queue: &mut AssetTransferQueue,
    data: MeshUploadData,
    shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
) {
    if data.buffer.length <= 0 {
        return;
    }
    let Some(device) = queue.gpu_device.clone() else {
        if queue.pending_mesh_uploads.len() >= MAX_PENDING_MESH_UPLOADS {
            logger::warn!(
                "mesh upload pending queue full; dropping asset {}",
                data.asset_id
            );
            return;
        }
        queue.pending_mesh_uploads.push_back(data);
        return;
    };
    if !data.high_priority && queue.mesh_upload_budget_this_poll == 0 {
        if queue.deferred_mesh_uploads.len() >= MAX_DEFERRED_MESH_UPLOADS {
            logger::warn!(
                "mesh {}: deferred mesh upload queue full; dropping",
                data.asset_id
            );
            return;
        }
        logger::trace!(
            "mesh {}: deferring low-priority mesh upload (budget exhausted)",
            data.asset_id
        );
        queue.deferred_mesh_uploads.push_back(data);
        return;
    }
    try_mesh_upload_with_device(queue, &device, data, shm, ipc, true);
}

/// `allow_defer` must be `false` when draining [`AssetTransferQueue::deferred_mesh_uploads`](super::super::AssetTransferQueue) so work is not
/// re-queued.
pub fn try_mesh_upload_with_device(
    queue: &mut AssetTransferQueue,
    device: &Arc<wgpu::Device>,
    data: MeshUploadData,
    shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
    allow_defer: bool,
) {
    let hint = data.upload_hint.flags;
    let high_priority = data.high_priority;
    let asset_id = data.asset_id;
    let started = Instant::now();

    let input_fp = mesh_upload_input_fingerprint(&data);
    let layout = if let Some(l) = queue.mesh_pool.get_cached_mesh_layout(asset_id, input_fp) {
        l
    } else {
        let Some(l) = compute_and_validate_mesh_layout(&data) else {
            logger::error!("mesh {asset_id}: invalid mesh layout or buffer descriptor");
            return;
        };
        queue
            .mesh_pool
            .set_cached_mesh_layout(asset_id, input_fp, l);
        l
    };

    let existing = queue.mesh_pool.get_mesh(asset_id);
    let queue_guard = queue
        .gpu_queue
        .as_ref()
        .map(|q| q.lock().unwrap_or_else(|e| e.into_inner()));
    let queue_ref = queue_guard.as_deref();

    let Some(limits) = queue.gpu_limits.as_ref() else {
        logger::error!(
            "mesh {asset_id}: gpu_device set but gpu_limits missing — attach ordering bug"
        );
        return;
    };
    let upload_result = shm.with_read_bytes(&data.buffer, |raw| {
        try_upload_mesh_from_raw(
            device.as_ref(),
            limits.as_ref(),
            queue_ref,
            raw,
            &data,
            existing,
            &layout,
        )
    });

    let Some(mesh) = upload_result else {
        // A `success` (or failure) field on [`MeshUploadResult`] requires updating the IPC contract
        // via SharedTypeGenerator and the host `Renderite.Shared` assembly; until then the host
        // cannot distinguish failure from a missing callback (timeouts may apply upstream).
        logger::error!(
            "mesh {asset_id}: upload failed or rejected — host callback not completed (no MeshUploadResult sent)"
        );
        return;
    };

    if allow_defer && !high_priority {
        queue.mesh_upload_budget_this_poll = queue.mesh_upload_budget_this_poll.saturating_sub(1);
    }

    let existed_before = queue.mesh_pool.insert_mesh(mesh);
    if let Some(ipc) = ipc {
        ipc.send_background(RendererCommand::MeshUploadResult(MeshUploadResult {
            asset_id,
            instance_changed: !existed_before,
        }));
    }

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    logger::trace!(
        "mesh_upload telemetry asset_id={} high_priority={} hint_flags=0x{:08x} (vlayout={} geo={} submesh={} dyn={}) replaced_existing={} time_ms={:.3}",
        asset_id,
        high_priority,
        hint.0,
        hint.vertex_layout(),
        hint.geometry(),
        hint.submesh_layout(),
        hint.dynamic(),
        existed_before,
        elapsed_ms
    );

    logger::trace!(
        "mesh {} uploaded (replaced={} resident_bytes≈{})",
        asset_id,
        existed_before,
        queue.mesh_pool.accounting().total_resident_bytes()
    );
}

/// Remove a mesh from the pool.
pub fn on_mesh_unload(queue: &mut AssetTransferQueue, u: MeshUnload) {
    if queue.mesh_pool.remove_mesh(u.asset_id) {
        logger::info!(
            "mesh {} unloaded (resident_bytes≈{})",
            u.asset_id,
            queue.mesh_pool.accounting().total_resident_bytes()
        );
    }
}
