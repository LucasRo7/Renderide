//! GPU timestamp queries for the world mesh forward raster pass (debug HUD).

use std::mem::size_of;
use std::time::Duration;

/// Number of timestamp slots (beginning and end of the mesh forward render pass).
const QUERY_COUNT: u32 = 2;

/// Frames between blocking readbacks of resolved timestamps (reduces CPU stalls).
pub const GPU_MESH_PASS_READBACK_INTERVAL: u32 = 60;

/// Query set, resolve buffer, staging buffer, and last readback value for mesh forward GPU time.
///
/// Created only when [`wgpu::Features::TIMESTAMP_QUERY`] is enabled on the device. Measures **only**
/// the world mesh forward render pass, not compute or other passes.
pub struct GpuMeshPassTimestamp {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    readback_frame_counter: u32,
    last_gpu_mesh_pass_ms: Option<f64>,
}

impl GpuMeshPassTimestamp {
    /// Allocates query and readback buffers when the device supports timestamp queries.
    pub fn new(device: &wgpu::Device) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("world-mesh-forward timestamps"),
            count: QUERY_COUNT,
            ty: wgpu::QueryType::Timestamp,
        });

        let size = (size_of::<u64>() as u64) * u64::from(QUERY_COUNT);
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("world-mesh-forward timestamp resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("world-mesh-forward timestamp staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Some(Self {
            query_set,
            resolve_buffer,
            staging_buffer,
            readback_frame_counter: 0,
            last_gpu_mesh_pass_ms: None,
        })
    }

    /// [`wgpu::QuerySet`] used as [`wgpu::RenderPassTimestampWrites::query_set`].
    pub fn query_set(&self) -> &wgpu::QuerySet {
        &self.query_set
    }

    /// Last successfully read GPU duration for the mesh forward pass, in milliseconds.
    pub fn last_gpu_mesh_pass_ms(&self) -> Option<f64> {
        self.last_gpu_mesh_pass_ms
    }

    /// Resolves queries into `resolve_buffer` and copies to `staging_buffer` on the same encoder as the pass.
    pub fn record_resolve_and_copy(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..QUERY_COUNT, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.staging_buffer,
            0,
            self.resolve_buffer.size(),
        );
    }

    /// Call after [`wgpu::Queue::submit`] for the frame that recorded timestamps. Throttles map/read.
    pub fn after_submit(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.readback_frame_counter = self.readback_frame_counter.saturating_add(1);
        if self.readback_frame_counter < GPU_MESH_PASS_READBACK_INTERVAL {
            return;
        }
        self.readback_frame_counter = 0;
        if let Some(ms) = readback_staging(device, queue, &self.staging_buffer) {
            self.last_gpu_mesh_pass_ms = Some(ms);
        }
    }
}

fn readback_staging(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    staging: &wgpu::Buffer,
) -> Option<f64> {
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let poll_result = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(5)),
    });
    if poll_result.is_err() {
        return None;
    }
    let view = staging.slice(..((size_of::<u64>() as u64) * u64::from(QUERY_COUNT)));
    let timestamps = {
        let mapped = view.get_mapped_range();
        bytemuck::pod_read_unaligned::<[u64; 2]>(&mapped)
    };
    staging.unmap();
    let period_ns = queue.get_timestamp_period();
    Some(gpu_timestamp_delta_to_ms(
        timestamps[1].saturating_sub(timestamps[0]),
        period_ns,
    ))
}

/// Converts a GPU timestamp delta (in query ticks) to milliseconds using the queue period in nanoseconds per tick.
pub fn gpu_timestamp_delta_to_ms(delta_ticks: u64, timestamp_period_ns: f32) -> f64 {
    let elapsed_ns = delta_ticks as f64 * f64::from(timestamp_period_ns);
    elapsed_ns / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::gpu_timestamp_delta_to_ms;

    #[test]
    fn delta_to_ms_scales_by_period() {
        let ms = gpu_timestamp_delta_to_ms(1_000_000, 1.0);
        assert!((ms - 1.0).abs() < 1e-9);
    }
}
