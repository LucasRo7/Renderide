//! Cached pipelines and bind layout for [`super::SceneColorComposePass`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const WGSL_MONO: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/post/scene_color_compose_mono.wgsl"
));
const WGSL_MULTIVIEW: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/post/scene_color_compose_multiview.wgsl"
));

/// GPU state shared by all compose passes (bind layout + sampler).
pub(super) struct SceneColorComposePipelineCache {
    bind_group_layout: OnceLock<wgpu::BindGroupLayout>,
    sampler: OnceLock<wgpu::Sampler>,
    mono: Mutex<HashMap<wgpu::TextureFormat, Arc<wgpu::RenderPipeline>>>,
    multiview: Mutex<HashMap<wgpu::TextureFormat, Arc<wgpu::RenderPipeline>>>,
}

impl Default for SceneColorComposePipelineCache {
    fn default() -> Self {
        Self {
            bind_group_layout: OnceLock::new(),
            sampler: OnceLock::new(),
            mono: Mutex::new(HashMap::new()),
            multiview: Mutex::new(HashMap::new()),
        }
    }
}

impl SceneColorComposePipelineCache {
    /// Linear clamp sampler for HDR scene color.
    pub(super) fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler.get_or_init(|| {
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
        self.bind_group_layout.get_or_init(|| {
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
        let mut guard = map
            .lock()
            .expect("scene_color_compose pipeline cache poisoned");
        if let Some(p) = guard.get(&output_format) {
            return Arc::clone(p);
        }
        logger::debug!(
            "scene_color_compose: building pipeline (dst format = {:?}, multiview = {})",
            output_format,
            multiview_stereo
        );
        let label = if multiview_stereo {
            "scene_color_compose_multiview"
        } else {
            "scene_color_compose_mono"
        };
        let source = if multiview_stereo {
            WGSL_MULTIVIEW
        } else {
            WGSL_MONO
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
        let pipeline = Arc::new(
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
                        format: output_format,
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
            }),
        );
        guard.insert(output_format, Arc::clone(&pipeline));
        pipeline
    }

    /// Bind group for one frame’s scene-color view.
    pub(super) fn bind_group(
        &self,
        device: &wgpu::Device,
        scene_color_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_color_compose"),
            layout: self.bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(scene_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(self.sampler(device)),
                },
            ],
        })
    }
}
