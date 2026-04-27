//! Cached pipelines and bind layout for [`super::AcesTonemapPass`].
//!
//! Mirrors the structure of [`crate::render_graph::passes::scene_color_compose`]'s pipeline
//! cache: per-output-format `wgpu::RenderPipeline` map for mono and multiview, with a single
//! linear-clamp sampler shared across all instances.
//!
//! WGSL is sourced from the build-time embedded shader registry
//! ([`crate::embedded_shaders::embedded_target_wgsl`]) so the same
//! `shaders/source/post/aces_tonemap.wgsl` source is composed once into mono and multiview
//! variants by the build script's `#ifdef MULTIVIEW` path (no runtime composition needed).

use std::sync::Arc;

use crate::embedded_shaders::{ACES_TONEMAP_DEFAULT_WGSL, ACES_TONEMAP_MULTIVIEW_WGSL};
use crate::render_graph::gpu_cache::{BindGroupMap, OnceGpu, RenderPipelineMap};

/// Debug label for the mono variant pipeline.
const PIPELINE_LABEL_MONO: &str = "aces_tonemap_default";
/// Debug label for the multiview variant pipeline.
const PIPELINE_LABEL_MULTIVIEW: &str = "aces_tonemap_multiview";

/// GPU state shared by all ACES tonemap passes (bind layout + sampler + per-format pipelines).
pub(super) struct AcesTonemapPipelineCache {
    /// Bind group layout shared by mono and multiview variants.
    bind_group_layout: OnceGpu<wgpu::BindGroupLayout>,
    /// Linear sampler used to read HDR scene color.
    sampler: OnceGpu<wgpu::Sampler>,
    /// Mono pipelines keyed by output color format.
    mono: RenderPipelineMap<wgpu::TextureFormat>,
    /// Multiview pipelines keyed by output color format.
    multiview: RenderPipelineMap<wgpu::TextureFormat>,
    /// Bind groups keyed by scene-color texture identity + multiview flag. `wgpu::Texture`
    /// implements `Eq + Hash` over its internal handle, so entries automatically follow the
    /// transient pool's allocation lifecycle — when the pool drops and recreates a texture,
    /// the stale entry is orphaned and cleaned up by bounded cache clearing.
    bind_groups: BindGroupMap<(wgpu::Texture, bool)>,
}

impl Default for AcesTonemapPipelineCache {
    fn default() -> Self {
        Self {
            bind_group_layout: OnceGpu::default(),
            sampler: OnceGpu::default(),
            mono: RenderPipelineMap::default(),
            multiview: RenderPipelineMap::default(),
            bind_groups: BindGroupMap::with_max_entries(MAX_CACHED_BIND_GROUPS),
        }
    }
}

impl AcesTonemapPipelineCache {
    /// Linear clamp sampler used to read the HDR scene color.
    pub(super) fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler.get_or_create(|| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("aces_tonemap"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            })
        })
    }

    /// Bind group layout for the HDR scene color texture array + sampler.
    fn bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bind_group_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aces_tonemap"),
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
                "aces_tonemap: building pipeline (dst format = {:?}, multiview = {})",
                output_format,
                multiview_stereo
            );
            let (label, source) = if multiview_stereo {
                (PIPELINE_LABEL_MULTIVIEW, ACES_TONEMAP_MULTIVIEW_WGSL)
            } else {
                (PIPELINE_LABEL_MONO, ACES_TONEMAP_DEFAULT_WGSL)
            };
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(self.bind_group_layout(device))],
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

    /// Bind group for one frame's scene-color texture, cached by `(Texture, multiview_stereo)`.
    ///
    /// Builds a fresh `D2Array` view on cache miss so the cached bind group outlives any single
    /// per-frame view clone. Hit is a `HashMap` lookup + `wgpu::BindGroup::clone` (Arc bump).
    pub(super) fn bind_group(
        &self,
        device: &wgpu::Device,
        scene_color_texture: &wgpu::Texture,
        multiview_stereo: bool,
    ) -> wgpu::BindGroup {
        let key = (scene_color_texture.clone(), multiview_stereo);
        self.bind_groups.get_or_create(key, |key| {
            let (scene_color_texture, multiview_stereo) = key;
            let layers_in_texture = scene_color_texture.size().depth_or_array_layers.max(1);
            let array_layer_count = if *multiview_stereo {
                2.min(layers_in_texture)
            } else {
                1
            };
            let view = scene_color_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("aces_tonemap_sampled"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                array_layer_count: Some(array_layer_count),
                ..Default::default()
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("aces_tonemap"),
                layout: self.bind_group_layout(device),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(self.sampler(device)),
                    },
                ],
            })
        })
    }
}

/// Upper bound for cached ACES bind groups before the cache is flushed.
///
/// The scene-color transient texture is stable across most frames — the cache normally holds
/// one or two entries (mono + multiview). This cap protects against unbounded growth when the
/// swapchain / MSAA setting flips repeatedly and the transient pool cycles allocations.
const MAX_CACHED_BIND_GROUPS: usize = 8;
