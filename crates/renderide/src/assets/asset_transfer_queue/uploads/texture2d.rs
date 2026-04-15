//! Texture2D format/properties/data IPC and [`crate::assets::texture::write_texture2d_mips`] uploads.

use crate::assets::texture::write_texture2d_mips;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetTexture2DData, SetTexture2DFormat, SetTexture2DProperties,
    SetTexture2DResult, TextureUpdateResultType, UnloadTexture2D,
};

use super::super::AssetTransferQueue;
use super::allocations::flush_pending_texture_allocations;
use super::{
    MAX_DEFERRED_TEXTURE_UPLOADS, MAX_DEFERRED_TEXTURE_UPLOADS_DRAIN_PER_POLL,
    MAX_PENDING_TEXTURE_UPLOADS,
};

fn send_texture_2d_result(
    ipc: Option<&mut DualQueueIpc>,
    asset_id: i32,
    update: i32,
    instance_changed: bool,
) {
    let Some(ipc) = ipc else {
        return;
    };
    ipc.send_background(RendererCommand::SetTexture2DResult(SetTexture2DResult {
        asset_id,
        r#type: TextureUpdateResultType(update),
        instance_changed,
    }));
}

/// Handle [`SetTexture2DFormat`](crate::shared::SetTexture2DFormat).
pub fn on_set_texture_2d_format(
    queue: &mut AssetTransferQueue,
    f: SetTexture2DFormat,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = f.asset_id;
    queue.texture_formats.insert(id, f.clone());
    let props = queue.texture_properties.get(&id);
    let Some(device) = queue.gpu_device.clone() else {
        send_texture_2d_result(
            ipc,
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.texture_pool.get_texture(id).is_none(),
        );
        return;
    };
    let Some(limits) = queue.gpu_limits.as_ref() else {
        logger::warn!("texture {id}: gpu_limits missing; format deferred until attach");
        send_texture_2d_result(
            ipc,
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.texture_pool.get_texture(id).is_none(),
        );
        return;
    };
    let Some(tex) = crate::resources::GpuTexture2d::new_from_format(
        device.as_ref(),
        limits.as_ref(),
        &f,
        props,
    ) else {
        logger::warn!("texture {id}: SetTexture2DFormat rejected (bad size or device)");
        return;
    };
    let existed_before = queue.texture_pool.insert_texture(tex);
    send_texture_2d_result(
        ipc,
        id,
        TextureUpdateResultType::FORMAT_SET,
        !existed_before,
    );
    logger::trace!(
        "texture {} format {:?} {}×{} mips={} (resident_bytes≈{})",
        id,
        f.format,
        f.width,
        f.height,
        f.mipmap_count,
        queue.texture_pool.accounting().texture_resident_bytes()
    );
}

/// Handle [`SetTexture2DProperties`](crate::shared::SetTexture2DProperties).
pub fn on_set_texture_2d_properties(
    queue: &mut AssetTransferQueue,
    p: SetTexture2DProperties,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = p.asset_id;
    queue.texture_properties.insert(id, p.clone());
    if let Some(t) = queue.texture_pool.get_texture_mut(id) {
        t.apply_properties(&p);
    }
    send_texture_2d_result(ipc, id, TextureUpdateResultType::PROPERTIES_SET, false);
}

/// Handle [`SetTexture2DData`](crate::shared::SetTexture2DData). Pass shared memory when available
/// so mips can be read from the host buffer; if GPU or texture is not ready, data is queued.
pub fn on_set_texture_2d_data(
    queue: &mut AssetTransferQueue,
    d: SetTexture2DData,
    shm: Option<&mut SharedMemoryAccessor>,
    ipc: Option<&mut DualQueueIpc>,
) {
    if d.data.length <= 0 {
        return;
    }
    if !queue.texture_formats.contains_key(&d.asset_id) {
        logger::warn!(
            "texture {}: SetTexture2DData before format; ignored",
            d.asset_id
        );
        return;
    }
    if queue.gpu_device.is_none() || queue.gpu_queue.is_none() {
        if queue.pending_texture_uploads.len() >= MAX_PENDING_TEXTURE_UPLOADS {
            logger::warn!(
                "texture {}: pending texture upload queue full; dropping",
                d.asset_id
            );
            return;
        }
        queue.pending_texture_uploads.push_back(d);
        return;
    }
    let Some(ref device) = queue.gpu_device.clone() else {
        return;
    };
    if queue.texture_pool.get_texture(d.asset_id).is_none() {
        flush_pending_texture_allocations(queue, device);
    }
    if queue.texture_pool.get_texture(d.asset_id).is_none() {
        if queue.pending_texture_uploads.len() >= MAX_PENDING_TEXTURE_UPLOADS {
            logger::warn!(
                "texture {}: no GPU texture and pending full; dropping data",
                d.asset_id
            );
            return;
        }
        queue.pending_texture_uploads.push_back(d);
        return;
    }
    if !d.high_priority && queue.texture_upload_budget_this_poll == 0 {
        if queue.deferred_texture_uploads.len() >= MAX_DEFERRED_TEXTURE_UPLOADS {
            logger::warn!(
                "texture {}: deferred texture upload queue full; dropping",
                d.asset_id
            );
            return;
        }
        logger::trace!(
            "texture {}: deferring low-priority texture upload (budget exhausted)",
            d.asset_id
        );
        queue.deferred_texture_uploads.push_back(d);
        return;
    }
    let Some(shm) = shm else {
        logger::warn!(
            "texture {}: SetTexture2DData needs shared memory for upload",
            d.asset_id
        );
        return;
    };
    try_texture_upload_with_device(queue, d, shm, ipc, true);
}

/// Upload texture mips from shared memory and optionally notify the host on the background queue.
///
/// When `consume_texture_upload_budget` is `true`, a successful upload decrements
/// [`AssetTransferQueue::texture_upload_budget_this_poll`](super::super::AssetTransferQueue) for non-[`SetTexture2DData::high_priority`](crate::shared::SetTexture2DData)
/// payloads (mirroring [`super::mesh::try_mesh_upload_with_device`] `allow_defer`). Use `false` when draining
/// [`AssetTransferQueue::deferred_texture_uploads`](super::super::AssetTransferQueue) or replaying [`AssetTransferQueue::pending_texture_uploads`](super::super::AssetTransferQueue)
/// on attach so deferred work does not double-charge the budget.
pub fn try_texture_upload_with_device(
    queue: &mut AssetTransferQueue,
    data: SetTexture2DData,
    shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
    consume_texture_upload_budget: bool,
) {
    let id = data.asset_id;
    let Some(fmt) = queue.texture_formats.get(&id).cloned() else {
        logger::warn!("texture {id}: missing format");
        return;
    };
    let (tex_arc, wgpu_fmt) = match queue.texture_pool.get_texture(id) {
        Some(t) => (t.texture.clone(), t.wgpu_format),
        None => {
            logger::warn!("texture {id}: missing GPU texture");
            return;
        }
    };
    let Some(queue_arc) = queue.gpu_queue.as_ref() else {
        return;
    };
    logger::trace!(
        "texture_upload telemetry asset_id={} payload_bytes={} high_priority={} has_region={} hint_readable={} mip_count={} start_mip={}",
        id,
        data.data.length.max(0),
        data.high_priority,
        data.hint.has_region != 0,
        data.hint.readable != 0,
        data.mip_map_sizes.len(),
        data.start_mip_level,
    );
    let upload_out = shm.with_read_bytes(&data.data, |raw| {
        let q = queue_arc.lock().unwrap_or_else(|e| e.into_inner());
        Some(write_texture2d_mips(
            &q,
            tex_arc.as_ref(),
            &fmt,
            wgpu_fmt,
            &data,
            raw,
        ))
    });
    match upload_out {
        Some(Ok(0)) => {
            logger::trace!("texture {id}: upload skipped (empty region hint)");
        }
        Some(Ok(uploaded_mips)) => {
            if consume_texture_upload_budget && !data.high_priority && uploaded_mips > 0 {
                queue.texture_upload_budget_this_poll =
                    queue.texture_upload_budget_this_poll.saturating_sub(1);
            }
            if let Some(t) = queue.texture_pool.get_texture_mut(id) {
                let start = data.start_mip_level.max(0) as u32;
                let end_exclusive = start.saturating_add(uploaded_mips).min(t.mip_levels_total);
                t.mip_levels_resident = t.mip_levels_resident.max(end_exclusive);
            }
            send_texture_2d_result(ipc, id, TextureUpdateResultType::DATA_UPLOAD, false);
            logger::trace!("texture {id}: data upload ok ({uploaded_mips} mips)");
        }
        Some(Err(e)) => {
            logger::warn!("texture {id}: upload failed: {e}");
        }
        None => {
            logger::warn!("texture {id}: shared memory slice missing");
        }
    }
}

/// Processes texture uploads deferred when the non-high-priority texture budget was exhausted
/// mid-batch.
pub fn drain_deferred_texture_uploads_after_poll(
    queue: &mut AssetTransferQueue,
    shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
) {
    let mut drained = 0usize;
    if let Some(ipc_ref) = ipc {
        while drained < MAX_DEFERRED_TEXTURE_UPLOADS_DRAIN_PER_POLL {
            let Some(data) = queue.deferred_texture_uploads.pop_front() else {
                break;
            };
            try_texture_upload_with_device(&mut *queue, data, shm, Some(ipc_ref), false);
            drained += 1;
        }
    } else {
        while drained < MAX_DEFERRED_TEXTURE_UPLOADS_DRAIN_PER_POLL {
            let Some(data) = queue.deferred_texture_uploads.pop_front() else {
                break;
            };
            try_texture_upload_with_device(&mut *queue, data, shm, None, false);
            drained += 1;
        }
    }
}

/// Remove a texture asset from CPU tables and the pool.
pub fn on_unload_texture_2d(queue: &mut AssetTransferQueue, u: UnloadTexture2D) {
    let id = u.asset_id;
    queue.texture_formats.remove(&id);
    queue.texture_properties.remove(&id);
    if queue.texture_pool.remove_texture(id) {
        logger::info!(
            "texture {id} unloaded (mesh≈{} tex≈{} total≈{})",
            queue.mesh_pool.accounting().mesh_resident_bytes(),
            queue.texture_pool.accounting().texture_resident_bytes(),
            queue.mesh_pool.accounting().total_resident_bytes()
        );
    }
}
