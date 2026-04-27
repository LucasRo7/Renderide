//! Cached pipelines and bind layout for [`super::SceneColorComposePass`].
//!
//! WGSL is sourced from the build-time embedded shader registry
//! ([`crate::embedded_shaders::embedded_target_wgsl`]): the single source
//! `shaders/source/post/scene_color_compose.wgsl` is composed twice with `#ifdef MULTIVIEW` into
//! `scene_color_compose_default` and `scene_color_compose_multiview` by the build script.

use std::sync::Arc;

use crate::embedded_shaders::{
    SCENE_COLOR_COMPOSE_DEFAULT_WGSL, SCENE_COLOR_COMPOSE_MULTIVIEW_WGSL,
};
use crate::render_graph::gpu_cache::{BindGroupMap, OnceGpu, RenderPipelineMap};

/// Debug label for the mono variant pipeline.
const PIPELINE_LABEL_MONO: &str = "scene_color_compose_default";
/// Debug label for the multiview variant pipeline.
const PIPELINE_LABEL_MULTIVIEW: &str = "scene_color_compose_multiview";

/// GPU state shared by all compose passes (bind layout + sampler).
pub(super) struct SceneColorComposePipelineCache {
    /// Bind group layout shared by mono and multiview variants.
    bind_group_layout: OnceGpu<wgpu::BindGroupLayout>,
    /// Linear sampler shared by all compose draws.
    sampler: OnceGpu<wgpu::Sampler>,
    /// Mono pipelines keyed by output color format.
    mono: RenderPipelineMap<wgpu::TextureFormat>,
    /// Multiview pipelines keyed by output color format.
    multiview: RenderPipelineMap<wgpu::TextureFormat>,
    /// Bind groups keyed by scene-color texture identity + multiview flag; avoids rebuilding
    /// on every frame when the transient pool reuses the same allocation.
    bind_groups: BindGroupMap<(wgpu::Texture, bool)>,
}

impl Default for SceneColorComposePipelineCache {
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

impl SceneColorComposePipelineCache {
    /// Linear clamp sampler for HDR scene color.
    pub(super) fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler.get_or_create(|| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("scene_color_compose"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            })
        })
    }

    fn bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bind_group_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene_color_compose"),
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
                "scene_color_compose: building pipeline (dst format = {:?}, multiview = {})",
                output_format,
                multiview_stereo
            );
            let (label, source) = if multiview_stereo {
                (PIPELINE_LABEL_MULTIVIEW, SCENE_COLOR_COMPOSE_MULTIVIEW_WGSL)
            } else {
                (PIPELINE_LABEL_MONO, SCENE_COLOR_COMPOSE_DEFAULT_WGSL)
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
                label: Some("scene_color_compose_sampled"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                array_layer_count: Some(array_layer_count),
                ..Default::default()
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene_color_compose"),
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

/// Upper bound for cached scene-color-compose bind groups before the cache is flushed.
///
/// Normally one or two entries (mono + multiview). The cap protects against unbounded growth
/// when the transient pool cycles allocations (resize / MSAA toggle).
const MAX_CACHED_BIND_GROUPS: usize = 8;
