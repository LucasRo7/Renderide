//! Cached pipelines, bind layouts, sampler, and per-pass uniform buffer for [`super::GtaoPass`].
//!
//! Two bind-group layouts are cached (mono vs multiview) because the depth sampling type differs
//! (`texture_depth_2d` vs `texture_depth_2d_array`). One GPU-side uniform buffer (`GtaoParams`) is
//! shared across every GTAO pass instance and rewritten from the CPU each record — GTAO is a
//! singleton effect in the chain, so a process-wide buffer avoids per-frame allocation churn.
//!
//! WGSL is sourced from the build-time embedded shader registry ([`embedded_target_wgsl`]) so
//! the same `shaders/source/post/gtao.wgsl` source is composed once into mono and multiview
//! variants by the build script's `#ifdef MULTIVIEW` path.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::embedded_shaders::{GTAO_DEFAULT_WGSL, GTAO_MULTIVIEW_WGSL};
use crate::render_graph::gpu_cache::{BindGroupMap, OnceGpu, RenderPipelineMap};

/// Debug label for the mono variant pipeline.
const PIPELINE_LABEL_MONO: &str = "gtao_default";
/// Debug label for the multiview variant pipeline.
const PIPELINE_LABEL_MULTIVIEW: &str = "gtao_multiview";

/// CPU mirror of the WGSL `GtaoParams` uniform (32 bytes, 16-byte aligned).
///
/// Rewritten every record from the live [`crate::config::GtaoSettings`]. Kept separate from
/// `FrameGpuUniforms` so GTAO's per-effect knobs don't bloat the shared frame-globals block.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct GtaoParamsGpu {
    /// World-space search radius (meters).
    pub radius_world: f32,
    /// Cap on the horizon search in pixels.
    pub max_pixel_radius: f32,
    /// AO strength exponent applied to the raw visibility factor.
    pub intensity: f32,
    /// Horizon steps per side.
    pub step_count: u32,
    /// Distance-falloff range as a fraction of `radius_world`; candidate samples are linearly
    /// faded toward the tangent-plane horizon over the last `falloff_range · radius_world` of
    /// the search radius (matches XeGTAO's `FalloffRange`).
    pub falloff_range: f32,
    /// Gray-albedo proxy for the multi-bounce fit (paper Eq. 10).
    pub albedo_multibounce: f32,
    /// Padding to 16-byte alignment (two f32 slots, matching WGSL `align_pad_tail: vec2<f32>`).
    pub align_pad_tail: [f32; 2],
}

/// Cache key for [`GtaoPipelineCache::bind_groups`].
///
/// `wgpu::Texture` and `wgpu::Buffer` both implement `Eq + Hash` via their internal handles, so
/// entries automatically follow the transient pool's / frame-resource manager's allocation
/// lifecycle: when any of the three backing resources is dropped and recreated, the stale
/// cache entry falls out on overflow eviction.
#[derive(Clone, Eq, Hash, PartialEq)]
struct GtaoBindGroupKey {
    /// Scene-color HDR source texture.
    scene_color_texture: wgpu::Texture,
    /// Scene-depth source texture (aspect view derived internally).
    scene_depth_texture: wgpu::Texture,
    /// Per-view frame-uniforms buffer.
    frame_uniforms: wgpu::Buffer,
    /// Mono vs multiview-stereo view shape.
    multiview_stereo: bool,
}

/// GPU state shared by all GTAO pass instances (layouts + sampler + per-format pipelines + UBO).
pub(super) struct GtaoPipelineCache {
    /// Bind-group layout for the mono pipeline (depth as `texture_depth_2d`).
    bind_group_layout_mono: OnceGpu<wgpu::BindGroupLayout>,
    /// Bind-group layout for the multiview pipeline (depth as `texture_depth_2d_array`).
    bind_group_layout_stereo: OnceGpu<wgpu::BindGroupLayout>,
    /// Linear-clamp sampler used to read the HDR scene color.
    sampler: OnceGpu<wgpu::Sampler>,
    /// Process-wide `GtaoParams` uniform buffer (rewritten every record).
    params_buffer: OnceGpu<wgpu::Buffer>,
    /// Cached pipelines keyed by output format (mono variant).
    mono: RenderPipelineMap<wgpu::TextureFormat>,
    /// Cached pipelines keyed by output format (multiview variant).
    multiview: RenderPipelineMap<wgpu::TextureFormat>,
    /// Bind groups keyed by `(scene_color, scene_depth, frame_uniforms, multiview_stereo)`.
    /// Normally one entry per active view (desktop / HMD / each secondary RT camera).
    bind_groups: BindGroupMap<GtaoBindGroupKey>,
}

impl Default for GtaoPipelineCache {
    fn default() -> Self {
        Self {
            bind_group_layout_mono: OnceGpu::default(),
            bind_group_layout_stereo: OnceGpu::default(),
            sampler: OnceGpu::default(),
            params_buffer: OnceGpu::default(),
            mono: RenderPipelineMap::default(),
            multiview: RenderPipelineMap::default(),
            bind_groups: BindGroupMap::with_max_entries(MAX_CACHED_BIND_GROUPS),
        }
    }
}

impl GtaoPipelineCache {
    /// Linear-clamp sampler used to read the HDR scene color.
    fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler.get_or_create(|| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("gtao"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            })
        })
    }

    /// Process-wide `GtaoParams` uniform buffer. Created on first access.
    pub(super) fn params_buffer(&self, device: &wgpu::Device) -> &wgpu::Buffer {
        self.params_buffer.get_or_create(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gtao-params"),
                size: std::mem::size_of::<GtaoParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
    }

    /// Bind-group layout for the selected variant. Entries 0..=4 are, in order:
    /// HDR scene-color array, linear sampler, scene depth, frame globals UBO, GTAO params UBO.
    pub(super) fn bind_group_layout(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
    ) -> &wgpu::BindGroupLayout {
        let slot = if multiview_stereo {
            &self.bind_group_layout_stereo
        } else {
            &self.bind_group_layout_mono
        };
        slot.get_or_create(|| {
            let depth_view_dim = if multiview_stereo {
                wgpu::TextureViewDimension::D2Array
            } else {
                wgpu::TextureViewDimension::D2
            };
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(if multiview_stereo {
                    "gtao-multiview"
                } else {
                    "gtao-mono"
                }),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: depth_view_dim,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
        })
    }

    /// Returns or builds a render pipeline for `output_format` and multiview stereo.
    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
        multiview_stereo: bool,
    ) -> Arc<wgpu::RenderPipeline> {
        let map = if multiview_stereo {
            &self.multiview
        } else {
            &self.mono
        };
        map.get_or_create(output_format, |output_format| {
            logger::debug!(
                "gtao: building pipeline (dst format = {:?}, multiview = {})",
                output_format,
                multiview_stereo
            );
            let (label, source) = if multiview_stereo {
                (PIPELINE_LABEL_MULTIVIEW, GTAO_MULTIVIEW_WGSL)
            } else {
                (PIPELINE_LABEL_MONO, GTAO_DEFAULT_WGSL)
            };
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(self.bind_group_layout(device, multiview_stereo))],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: *output_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: multiview_stereo
                    .then(|| std::num::NonZeroU32::new(3))
                    .flatten(),
                cache: None,
            })
        })
    }

    /// Bind group for one frame's set of textures + UBOs, cached by
    /// `(scene_color_texture, scene_depth_texture, frame_uniforms, multiview_stereo)`.
    ///
    /// Builds the per-dispatch `D2Array` color view and depth-aspect view on miss so the cached
    /// bind group outlives any single per-frame view clone. Hit is a `HashMap` lookup +
    /// `wgpu::BindGroup::clone` (Arc bump).
    pub(super) fn bind_group(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
        scene_color_texture: &wgpu::Texture,
        scene_depth_texture: &wgpu::Texture,
        frame_uniforms: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let key = GtaoBindGroupKey {
            scene_color_texture: scene_color_texture.clone(),
            scene_depth_texture: scene_depth_texture.clone(),
            frame_uniforms: frame_uniforms.clone(),
            multiview_stereo,
        };
        self.bind_groups.get_or_create(key, |key| {
            let color_layers = key.scene_color_texture.size().depth_or_array_layers.max(1);
            let color_layer_count = if key.multiview_stereo {
                2.min(color_layers)
            } else {
                1
            };
            let scene_color_view =
                key.scene_color_texture
                    .create_view(&wgpu::TextureViewDescriptor {
                        label: Some("gtao_scene_color_sampled"),
                        dimension: Some(wgpu::TextureViewDimension::D2Array),
                        array_layer_count: Some(color_layer_count),
                        ..Default::default()
                    });
            let (depth_dim, depth_layer_count) = if key.multiview_stereo {
                (wgpu::TextureViewDimension::D2Array, Some(2))
            } else {
                (wgpu::TextureViewDimension::D2, Some(1))
            };
            let scene_depth_view =
                key.scene_depth_texture
                    .create_view(&wgpu::TextureViewDescriptor {
                        label: Some("gtao_scene_depth_sampled"),
                        aspect: wgpu::TextureAspect::DepthOnly,
                        dimension: Some(depth_dim),
                        array_layer_count: depth_layer_count,
                        ..Default::default()
                    });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gtao"),
                layout: self.bind_group_layout(device, key.multiview_stereo),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&scene_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(self.sampler(device)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&scene_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: key.frame_uniforms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.params_buffer(device).as_entire_binding(),
                    },
                ],
            })
        })
    }
}

/// Upper bound for cached GTAO bind groups before the cache is flushed.
///
/// Expected occupancy is one entry per active view (desktop / HMD / each secondary RT camera).
/// The cap protects against unbounded growth when views cycle during resize / MSAA / camera
/// churn.
const MAX_CACHED_BIND_GROUPS: usize = 16;
