//! GPU hierarchical depth pyramid build and CPU readback for occlusion tests.

use std::num::NonZeroU64;
use std::sync::mpsc;
use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};

use crate::render_graph::{
    hi_z_pyramid_dimensions, hi_z_snapshot_from_linear_linear, mip_dimensions,
    mip_levels_for_extent, unpack_linear_rows_to_mips, HiZCpuSnapshot, HiZStereoCpuSnapshot,
    HiZTemporalState, OutputDepthMode,
};

const HIZ_MAX_MIPS: u32 = 8;

const MIP0_DESKTOP_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/compute/hi_z_mip0_desktop.wgsl"
));
const MIP0_STEREO_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/compute/hi_z_mip0_stereo.wgsl"
));
const DOWNSAMPLE_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/compute/hi_z_downsample_max.wgsl"
));

/// GPU + CPU Hi-Z state owned by [`crate::backend::RenderBackend`].
pub struct HiZGpuState {
    /// Last successfully read desktop pyramid (previous frame).
    pub desktop: Option<HiZCpuSnapshot>,
    /// Last successfully read stereo pyramids (previous frame).
    pub stereo: Option<HiZStereoCpuSnapshot>,
    /// View/projection snapshot for the frame that produced [`Self::desktop`] / [`Self::stereo`].
    pub temporal: Option<HiZTemporalState>,
    scratch: Option<HiZGpuScratch>,
    last_extent: (u32, u32),
    last_mode: OutputDepthMode,
    /// Ping-pong index (`0` or `1`): staging slot the next [`encode_hi_z_build`] copies into.
    write_slot: u8,
    /// `true` after at least one [`Self::on_frame_submitted`] (a prior submit wrote a pyramid).
    has_submitted_frame: bool,
}

impl Default for HiZGpuState {
    fn default() -> Self {
        Self {
            desktop: None,
            stereo: None,
            temporal: None,
            scratch: None,
            last_extent: (0, 0),
            last_mode: OutputDepthMode::DesktopSingle,
            write_slot: 0,
            has_submitted_frame: false,
        }
    }
}

impl HiZGpuState {
    /// Drops GPU scratch and CPU snapshots when resolution or depth mode changes.
    pub fn invalidate_if_needed(&mut self, extent: (u32, u32), mode: OutputDepthMode) {
        if self.last_extent != extent || self.last_mode != mode {
            self.desktop = None;
            self.stereo = None;
            self.temporal = None;
            self.scratch = None;
            self.write_slot = 0;
            self.has_submitted_frame = false;
        }
        self.last_extent = extent;
        self.last_mode = mode;
    }

    /// Clears ring readback state without mapping (e.g. device loss).
    pub fn clear_pending(&mut self) {
        self.write_slot = 0;
        self.has_submitted_frame = false;
    }

    /// Maps the staging slot filled on the **previous** submit into [`Self::desktop`] / [`Self::stereo`].
    ///
    /// Call at the **start** of each frame (before encoding the render graph). Temporal Hi-Z occlusion
    /// uses this pyramid for culling during **this** frame’s forward pass; it reflects GPU depth from
    /// the **prior** submitted frame. The first frame after startup has nothing to read.
    ///
    /// This may [`wgpu::Device::poll`] until the prior frame’s copy completes; it does **not** run
    /// after [`wgpu::Queue::submit`] for the current frame, so the main thread does not block the
    /// full current-frame GPU workload behind a post-submit readback.
    pub fn begin_frame_readback(&mut self, device: &wgpu::Device) {
        if !self.has_submitted_frame {
            return;
        }
        let read_slot = usize::from(1u8.wrapping_sub(self.write_slot));
        let Some(scratch) = self.scratch.as_ref() else {
            return;
        };
        let desktop_buf = &scratch.staging_desktop[read_slot];
        let right_opt = scratch.staging_r.as_ref().map(|r| &r[read_slot]);
        map_staging_pair_to_cpu_snapshots(
            device,
            scratch.extent,
            scratch.mip_levels,
            desktop_buf,
            right_opt,
            &mut self.desktop,
            &mut self.stereo,
        );
    }

    /// Flips [`Self::write_slot`] after a successful render-graph submit so the next frame’s
    /// [`Self::begin_frame_readback`] maps the staging buffer that was just written.
    pub fn on_frame_submitted(&mut self) {
        self.write_slot ^= 1;
        self.has_submitted_frame = true;
    }
}

/// Maps `desktop` + optional `right` staging buffers into `desktop_out` / `stereo_out`.
fn map_staging_pair_to_cpu_snapshots(
    device: &wgpu::Device,
    extent: (u32, u32),
    mip_levels: u32,
    desktop: &wgpu::Buffer,
    right: Option<&wgpu::Buffer>,
    desktop_out: &mut Option<HiZCpuSnapshot>,
    stereo_out: &mut Option<HiZStereoCpuSnapshot>,
) {
    let map_read = |buf: &wgpu::Buffer| -> Option<Vec<u8>> {
        let slice = buf.slice(..);
        let (send, recv) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = send.send(r);
        });
        loop {
            let _ = device.poll(wgpu::PollType::Poll);
            match recv.try_recv() {
                Ok(Ok(())) => break,
                Ok(Err(_)) => {
                    buf.unmap();
                    return None;
                }
                Err(mpsc::TryRecvError::Empty) => std::thread::yield_now(),
                Err(mpsc::TryRecvError::Disconnected) => return None,
            }
        }
        let range = buf.slice(..).get_mapped_range().to_vec();
        buf.unmap();
        Some(range)
    };

    let Some(desktop_raw) = map_read(desktop) else {
        return;
    };
    let mips = match unpack_linear_rows_to_mips(extent.0, extent.1, mip_levels, &desktop_raw) {
        Some(m) => m,
        None => {
            logger::warn!("Hi-Z desktop readback unpack failed");
            return;
        }
    };
    let desktop_snap = match hi_z_snapshot_from_linear_linear(extent.0, extent.1, mip_levels, mips)
    {
        Some(s) => s,
        None => {
            logger::warn!("Hi-Z desktop snapshot validation failed");
            return;
        }
    };

    if let Some(rbuf) = right {
        let Some(r_raw) = map_read(rbuf) else {
            *desktop_out = Some(desktop_snap);
            *stereo_out = None;
            return;
        };
        let mips_r = match unpack_linear_rows_to_mips(extent.0, extent.1, mip_levels, &r_raw) {
            Some(m) => m,
            None => {
                logger::warn!("Hi-Z stereo right readback unpack failed");
                *desktop_out = Some(desktop_snap);
                return;
            }
        };
        let right_snap =
            match hi_z_snapshot_from_linear_linear(extent.0, extent.1, mip_levels, mips_r) {
                Some(s) => s,
                None => {
                    logger::warn!("Hi-Z right snapshot validation failed");
                    *desktop_out = Some(desktop_snap);
                    return;
                }
            };
        *stereo_out = Some(HiZStereoCpuSnapshot {
            left: desktop_snap,
            right: right_snap,
        });
        *desktop_out = None;
    } else {
        *desktop_out = Some(desktop_snap);
        *stereo_out = None;
    }
}

/// Transient GPU resources reused while extent and mip count stay stable.
struct HiZGpuScratch {
    extent: (u32, u32),
    mip_levels: u32,
    pyramid: wgpu::Texture,
    views: Vec<wgpu::TextureView>,
    pyramid_r: Option<(wgpu::Texture, Vec<wgpu::TextureView>)>,
    /// Ping-pong staging for async readback (see [`HiZGpuState::write_slot`]).
    staging_desktop: [wgpu::Buffer; 2],
    staging_r: Option<[wgpu::Buffer; 2]>,
    layer_uniform: wgpu::Buffer,
    downsample_uniform: wgpu::Buffer,
}

struct HiZPipelines {
    mip0_desktop: wgpu::ComputePipeline,
    mip0_stereo: wgpu::ComputePipeline,
    downsample: wgpu::ComputePipeline,
    bgl_mip0_desktop: wgpu::BindGroupLayout,
    bgl_mip0_stereo: wgpu::BindGroupLayout,
    bgl_downsample: wgpu::BindGroupLayout,
}

impl HiZPipelines {
    fn get(device: &wgpu::Device) -> &'static Self {
        static CACHE: OnceLock<HiZPipelines> = OnceLock::new();
        CACHE.get_or_init(|| Self::new(device))
    }

    fn new(device: &wgpu::Device) -> Self {
        let bgl_mip0_desktop = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hi_z_mip0_desktop"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let bgl_mip0_stereo = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hi_z_mip0_stereo"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(16),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let bgl_downsample = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hi_z_downsample"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(16),
                    },
                    count: None,
                },
            ],
        });

        let layout_mip0_d = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hi_z_mip0_desktop_layout"),
            bind_group_layouts: &[Some(&bgl_mip0_desktop)],
            immediate_size: 0,
        });
        let layout_mip0_s = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hi_z_mip0_stereo_layout"),
            bind_group_layouts: &[Some(&bgl_mip0_stereo)],
            immediate_size: 0,
        });
        let layout_ds = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hi_z_downsample_layout"),
            bind_group_layouts: &[Some(&bgl_downsample)],
            immediate_size: 0,
        });

        let shader_m0d = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hi_z_mip0_desktop"),
            source: wgpu::ShaderSource::Wgsl(MIP0_DESKTOP_SRC.into()),
        });
        let shader_m0s = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hi_z_mip0_stereo"),
            source: wgpu::ShaderSource::Wgsl(MIP0_STEREO_SRC.into()),
        });
        let shader_ds = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hi_z_downsample"),
            source: wgpu::ShaderSource::Wgsl(DOWNSAMPLE_SRC.into()),
        });

        let mip0_desktop = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hi_z_mip0_desktop"),
            layout: Some(&layout_mip0_d),
            module: &shader_m0d,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mip0_stereo = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hi_z_mip0_stereo"),
            layout: Some(&layout_mip0_s),
            module: &shader_m0s,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let downsample = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hi_z_downsample"),
            layout: Some(&layout_ds),
            module: &shader_ds,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            mip0_desktop,
            mip0_stereo,
            downsample,
            bgl_mip0_desktop,
            bgl_mip0_stereo,
            bgl_downsample,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerUniform {
    layer: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DownsampleUniform {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
}

fn staging_size_pyramid(base_w: u32, base_h: u32, mip_levels: u32) -> u64 {
    let mut total = 0u64;
    for mip in 0..mip_levels {
        let (w, h) = mip_dimensions(base_w, base_h, mip).unwrap_or((0, 0));
        let row_pitch = wgpu::util::align_to(w * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as u64;
        total += row_pitch * u64::from(h);
    }
    total
}

impl HiZGpuScratch {
    fn new(device: &wgpu::Device, extent: (u32, u32), stereo: bool) -> Option<Self> {
        let (bw, bh) = extent;
        if bw == 0 || bh == 0 {
            return None;
        }
        let mip_levels = mip_levels_for_extent(bw, bh, HIZ_MAX_MIPS);
        if mip_levels == 0 {
            return None;
        }

        let make_pyramid = || -> (wgpu::Texture, Vec<wgpu::TextureView>) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("hi_z_pyramid"),
                size: wgpu::Extent3d {
                    width: bw,
                    height: bh,
                    depth_or_array_layers: 1,
                },
                mip_level_count: mip_levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let mut views = Vec::with_capacity(mip_levels as usize);
            for m in 0..mip_levels {
                let v = tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("hi_z_pyramid_mip"),
                    format: Some(wgpu::TextureFormat::R32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: m,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: None,
                    ..Default::default()
                });
                views.push(v);
            }
            (tex, views)
        };

        let (pyramid, views) = make_pyramid();
        let staging_size = staging_size_pyramid(bw, bh, mip_levels);
        let staging_desktop = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hi_z_staging_desktop_0"),
                size: staging_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hi_z_staging_desktop_1"),
                size: staging_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        ];

        let (pyramid_r, staging_r) = if stereo {
            let (t, v) = make_pyramid();
            let buf = [
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("hi_z_staging_r_0"),
                    size: staging_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("hi_z_staging_r_1"),
                    size: staging_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
            ];
            (Some((t, v)), Some(buf))
        } else {
            (None, None)
        };

        let layer_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hi_z_layer_uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let downsample_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hi_z_downsample_uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Some(Self {
            extent: (bw, bh),
            mip_levels,
            pyramid,
            views,
            pyramid_r,
            staging_desktop,
            staging_r,
            layer_uniform,
            downsample_uniform,
        })
    }
}

#[derive(Clone, Copy)]
enum DepthBinding {
    D2,
    D2Array { layer: u32 },
}

/// Records Hi-Z build + copy-to-staging into the current [`HiZGpuState::write_slot`].
/// Call [`HiZGpuState::begin_frame_readback`] at the **start** of the next frame to map the other slot.
pub fn encode_hi_z_build(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    depth_view: &wgpu::TextureView,
    extent: (u32, u32),
    mode: OutputDepthMode,
    state: &mut HiZGpuState,
) {
    state.invalidate_if_needed(extent, mode);

    let (full_w, full_h) = extent;
    if full_w == 0 || full_h == 0 {
        return;
    }

    let (bw, bh) = hi_z_pyramid_dimensions(full_w, full_h);
    if bw == 0 || bh == 0 {
        return;
    }

    let stereo = matches!(mode, OutputDepthMode::StereoArray { .. });
    let mip_levels = mip_levels_for_extent(bw, bh, HIZ_MAX_MIPS);
    if state.scratch.as_ref().map(|s| (s.extent, s.mip_levels)) != Some(((bw, bh), mip_levels))
        || state.scratch.as_ref().map(|s| s.pyramid_r.is_some()) != Some(stereo)
    {
        state.scratch = HiZGpuScratch::new(device, (bw, bh), stereo);
    }
    let Some(scratch) = state.scratch.as_mut() else {
        return;
    };
    let ws = usize::from(state.write_slot);

    let pipes = HiZPipelines::get(device);

    let dispatch_mip0_and_downsample =
        |encoder: &mut wgpu::CommandEncoder,
         pyramid_views: &[wgpu::TextureView],
         depth_bind: DepthBinding| {
            match depth_bind {
                DepthBinding::D2 => {
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hi_z_mip0_d_bg"),
                        layout: &pipes.bgl_mip0_desktop,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(depth_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&pyramid_views[0]),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("hi_z_mip0_desktop"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&pipes.mip0_desktop);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(
                            scratch.extent.0.div_ceil(8),
                            scratch.extent.1.div_ceil(8),
                            1,
                        );
                    }
                }
                DepthBinding::D2Array { layer } => {
                    let layer_u = LayerUniform {
                        layer,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    };
                    queue.write_buffer(&scratch.layer_uniform, 0, bytemuck::bytes_of(&layer_u));
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hi_z_mip0_s_bg"),
                        layout: &pipes.bgl_mip0_stereo,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(depth_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: scratch.layer_uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&pyramid_views[0]),
                            },
                        ],
                    });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("hi_z_mip0_stereo"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&pipes.mip0_stereo);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.dispatch_workgroups(
                            scratch.extent.0.div_ceil(8),
                            scratch.extent.1.div_ceil(8),
                            1,
                        );
                    }
                }
            }

            for mip in 0..scratch.mip_levels.saturating_sub(1) {
                let (sw, sh) = mip_dimensions(bw, bh, mip).unwrap_or((1, 1));
                let (dw, dh) = mip_dimensions(bw, bh, mip + 1).unwrap_or((1, 1));
                let du = DownsampleUniform {
                    src_w: sw,
                    src_h: sh,
                    dst_w: dw,
                    dst_h: dh,
                };
                queue.write_buffer(&scratch.downsample_uniform, 0, bytemuck::bytes_of(&du));
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hi_z_ds_bg"),
                    layout: &pipes.bgl_downsample,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &pyramid_views[mip as usize],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &pyramid_views[mip as usize + 1],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: scratch.downsample_uniform.as_entire_binding(),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("hi_z_downsample"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipes.downsample);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(dw.div_ceil(8), dh.div_ceil(8), 1);
                }
            }
        };

    match mode {
        OutputDepthMode::DesktopSingle => {
            dispatch_mip0_and_downsample(encoder, &scratch.views, DepthBinding::D2);
            copy_pyramid_to_staging(
                encoder,
                &scratch.pyramid,
                bw,
                bh,
                scratch.mip_levels,
                &scratch.staging_desktop[ws],
            );
        }
        OutputDepthMode::StereoArray { .. } => {
            let Some((ref pyr_r, ref views_r)) = scratch.pyramid_r else {
                return;
            };
            dispatch_mip0_and_downsample(
                encoder,
                &scratch.views,
                DepthBinding::D2Array { layer: 0 },
            );
            dispatch_mip0_and_downsample(encoder, views_r, DepthBinding::D2Array { layer: 1 });
            copy_pyramid_to_staging(
                encoder,
                &scratch.pyramid,
                bw,
                bh,
                scratch.mip_levels,
                &scratch.staging_desktop[ws],
            );
            copy_pyramid_to_staging(
                encoder,
                pyr_r,
                bw,
                bh,
                scratch.mip_levels,
                &scratch.staging_r.as_ref().expect("stereo staging")[ws],
            );
        }
    }
}

fn copy_pyramid_to_staging(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    base_w: u32,
    base_h: u32,
    mip_levels: u32,
    staging: &wgpu::Buffer,
) {
    let mut offset = 0u64;
    for mip in 0..mip_levels {
        let (w, h) = mip_dimensions(base_w, base_h, mip).unwrap_or((1, 1));
        let row_pitch = wgpu::util::align_to(w * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as u32;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: Some(row_pitch),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        offset += u64::from(row_pitch) * u64::from(h);
    }
}
