//! GPU Hi-Z pyramid build (compute) and staging readback for temporal CPU occlusion.

use wgpu::util::DeviceExt;

use crate::render_graph::hi_z_occlusion::{
    hi_z_base_dimensions, hi_z_decode_pyramid_buffer, hi_z_mip_offsets, hi_z_total_floats,
};

const HI_Z_SHADER: &str = include_str!("../../shaders/source/compute/hi_z_build.wgsl");

/// GPU pipelines and buffers for one-frame Hi-Z generation.
pub struct HiZGpuResources {
    depth_to_base_layout: wgpu::BindGroupLayout,
    depth_to_base_pipeline: wgpu::ComputePipeline,
    downsample_layout: wgpu::BindGroupLayout,
    downsample_pipeline: wgpu::ComputePipeline,
    depth_params_buf: wgpu::Buffer,
    ds_params_buf: wgpu::Buffer,
    pyramid_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    pyramid_capacity_floats: usize,
    base_dims: (u32, u32),
    depth_dims: (u32, u32),
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DepthToBaseParamsGpu {
    depth_width: u32,
    depth_height: u32,
    base_width: u32,
    base_height: u32,
    _pad: [u32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DownsampleParamsGpu {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    src_offset: u32,
    dst_offset: u32,
    _pad: [u32; 2],
}

impl HiZGpuResources {
    /// Creates pipelines and placeholder buffers; [`Self::ensure_pyramid`] resizes for viewport.
    pub fn new(device: &wgpu::Device) -> Self {
        let depth_to_base_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hi_z_build"),
            source: wgpu::ShaderSource::Wgsl(HI_Z_SHADER.into()),
        });

        let depth_to_base_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hi_z_depth_to_base"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(32).unwrap()),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let depth_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hi_z_depth_to_base_layout"),
                bind_group_layouts: &[Some(&depth_to_base_layout)],
                immediate_size: 0,
            });

        let depth_to_base_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hi_z_depth_to_base"),
                layout: Some(&depth_pipeline_layout),
                module: &depth_to_base_module,
                entry_point: Some("depth_to_base"),
                compilation_options: Default::default(),
                cache: None,
            });

        let downsample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hi_z_downsample"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(32).unwrap()),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let down_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hi_z_downsample_layout"),
            bind_group_layouts: &[Some(&downsample_layout)],
            immediate_size: 0,
        });

        let downsample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hi_z_downsample"),
                layout: Some(&down_pipeline_layout),
                module: &depth_to_base_module,
                entry_point: Some("downsample_mip"),
                compilation_options: Default::default(),
                cache: None,
            });

        let depth_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hi_z_depth_params"),
            contents: &[0u8; 32],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let ds_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hi_z_ds_params"),
            contents: &[0u8; 32],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let empty_pyramid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hi_z_pyramid"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hi_z_staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            depth_to_base_layout,
            depth_to_base_pipeline,
            downsample_layout,
            downsample_pipeline,
            depth_params_buf,
            ds_params_buf,
            pyramid_buf: empty_pyramid,
            staging_buf,
            pyramid_capacity_floats: 0,
            base_dims: (1, 1),
            depth_dims: (1, 1),
        }
    }

    /// Resizes internal buffers when depth extent or base dims change.
    pub fn ensure_pyramid(&mut self, device: &wgpu::Device, depth_w: u32, depth_h: u32) {
        let (bw, bh) = hi_z_base_dimensions(depth_w, depth_h);
        let need = hi_z_total_floats(bw, bh);
        if need == self.pyramid_capacity_floats && (depth_w, depth_h) == self.depth_dims {
            return;
        }
        self.depth_dims = (depth_w, depth_h);
        self.base_dims = (bw, bh);
        self.pyramid_capacity_floats = need;
        let size = (need * std::mem::size_of::<f32>()) as u64;
        self.pyramid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hi_z_pyramid"),
            size: size.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hi_z_staging"),
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }

    pub fn base_dims(&self) -> (u32, u32) {
        self.base_dims
    }

    /// Encodes depth → pyramid reduction and mip chain into `encoder`.
    pub fn encode_build(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        queue: &wgpu::Queue,
    ) {
        let (bw, bh) = self.base_dims;
        let (dw, dh) = self.depth_dims;
        let dp = DepthToBaseParamsGpu {
            depth_width: dw,
            depth_height: dh,
            base_width: bw,
            base_height: bh,
            _pad: [0; 4],
        };
        queue.write_buffer(&self.depth_params_buf, 0, bytemuck::bytes_of(&dp));

        let depth_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hi_z_depth_bg"),
            layout: &self.depth_to_base_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.depth_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.pyramid_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hi_z_depth_to_base"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.depth_to_base_pipeline);
            cpass.set_bind_group(0, &depth_bg, &[]);
            let wg_x = bw.div_ceil(8);
            let wg_y = bh.div_ceil(8);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Downsample chain
        let offsets = hi_z_mip_offsets(bw, bh);
        if offsets.len() <= 1 {
            return;
        }

        let mut src_w = bw;
        let mut src_h = bh;
        let mut src_off: u32 = offsets[0] as u32;

        for &dst_off_us in offsets.iter().skip(1) {
            let dst_w = src_w.div_ceil(2);
            let dst_h = src_h.div_ceil(2);
            let dst_off = dst_off_us as u32;
            let dsp = DownsampleParamsGpu {
                src_width: src_w,
                src_height: src_h,
                dst_width: dst_w,
                dst_height: dst_h,
                src_offset: src_off,
                dst_offset: dst_off,
                _pad: [0; 2],
            };
            queue.write_buffer(&self.ds_params_buf, 0, bytemuck::bytes_of(&dsp));

            let ds_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hi_z_downsample_bg"),
                layout: &self.downsample_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.ds_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.pyramid_buf.as_entire_binding(),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hi_z_downsample"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.downsample_pipeline);
                cpass.set_bind_group(0, &ds_bg, &[]);
                let wg_x = dst_w.div_ceil(8);
                let wg_y = dst_h.div_ceil(8);
                cpass.dispatch_workgroups(wg_x, wg_y, 1);
            }

            src_w = dst_w;
            src_h = dst_h;
            src_off = dst_off;
        }
    }

    /// Copies the pyramid GPU buffer to staging for CPU map (call before submit).
    pub fn encode_copy_to_staging(&self, encoder: &mut wgpu::CommandEncoder) {
        let size = (self.pyramid_capacity_floats * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&self.pyramid_buf, 0, &self.staging_buf, 0, size);
    }

    /// Staging buffer for async readback.
    pub fn staging_buffer(&self) -> &wgpu::Buffer {
        &self.staging_buf
    }

    /// Decodes mapped staging data into a snapshot (after valid map).
    pub fn decode_staging_to_snapshot(
        mapped: &[u8],
        base_w: u32,
        base_h: u32,
    ) -> crate::render_graph::HiZCpuSnapshot {
        let floats: &[f32] = bytemuck::cast_slice(mapped);
        hi_z_decode_pyramid_buffer(floats, base_w, base_h).unwrap_or_default()
    }
}
