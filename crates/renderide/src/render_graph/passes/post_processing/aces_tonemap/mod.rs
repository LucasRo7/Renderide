//! Stephen Hill ACES Fitted tonemap render pass.
//!
//! Reads an HDR scene-color array texture, applies the ACES Fitted curve (sRGB → AP1 → RRT+ODT
//! polynomial → AP1 → sRGB → saturate), and writes a chain HDR transient that the next post pass
//! (or [`crate::render_graph::passes::SceneColorComposePass`]) consumes. Output is in `[0, 1]`
//! linear sRGB so the existing sRGB swapchain encodes gamma correctly without a separate gamma
//! pass.

mod pipeline;

use std::num::NonZeroU32;
use std::sync::OnceLock;

use pipeline::AcesTonemapPipelineCache;

use crate::config::PostProcessingSettings;
use crate::render_graph::context::{GraphRasterPassContext, RenderPassContext};
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{PassBuilder, RenderPass};
use crate::render_graph::post_processing::{PostProcessEffect, PostProcessEffectId};
use crate::render_graph::resources::{TextureAccess, TextureHandle};

/// Graph handles for [`AcesTonemapPass`].
#[derive(Clone, Copy, Debug)]
pub struct AcesTonemapGraphResources {
    /// HDR scene-color input (the previous chain stage's output, or `scene_color_hdr` for the
    /// first effect in the chain).
    pub input: TextureHandle,
    /// HDR chain output written by this pass.
    pub output: TextureHandle,
}

/// Fullscreen render pass applying Stephen Hill ACES Fitted to `input`, writing `output`.
pub struct AcesTonemapPass {
    resources: AcesTonemapGraphResources,
    pipelines: &'static AcesTonemapPipelineCache,
}

impl AcesTonemapPass {
    /// Creates a new ACES tonemap pass instance.
    pub fn new(resources: AcesTonemapGraphResources) -> Self {
        Self {
            resources,
            pipelines: aces_tonemap_pipelines(),
        }
    }
}

/// Process-wide pipeline cache shared by every ACES pass instance.
fn aces_tonemap_pipelines() -> &'static AcesTonemapPipelineCache {
    static CACHE: OnceLock<AcesTonemapPipelineCache> = OnceLock::new();
    CACHE.get_or_init(AcesTonemapPipelineCache::default)
}

impl RenderPass for AcesTonemapPass {
    fn name(&self) -> &str {
        "AcesTonemap"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.read_texture_resource(
            self.resources.input,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        let mut r = b.raster();
        r.color(
            self.resources.output,
            wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
            Option::<TextureHandle>::None,
        );
        Ok(())
    }

    fn execute(&mut self, _ctx: &mut RenderPassContext<'_, '_, '_>) -> Result<(), RenderPassError> {
        Ok(())
    }

    fn graph_managed_raster(&self) -> bool {
        true
    }

    fn graph_raster_multiview_mask(
        &self,
        ctx: &GraphRasterPassContext<'_, '_>,
        template: &crate::render_graph::RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        let stereo = ctx
            .frame
            .as_ref()
            .is_some_and(|frame| frame.multiview_stereo);
        if stereo {
            NonZeroU32::new(3)
        } else {
            template.multiview_mask
        }
    }

    fn execute_graph_raster(
        &mut self,
        ctx: &mut GraphRasterPassContext<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        let Some(frame) = ctx.frame.as_ref() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        let Some(graph_resources) = ctx.graph_resources else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        let Some(tex) = graph_resources.transient_texture(self.resources.input) else {
            return Err(RenderPassError::MissingFrameParams {
                pass: format!(
                    "{} (missing transient input {:?})",
                    self.name(),
                    self.resources.input
                ),
            });
        };
        let target_format = output_attachment_format(self.resources.output, ctx);
        let pipeline = self
            .pipelines
            .pipeline(ctx.device, target_format, frame.multiview_stereo);
        let hdr_sample_view = tex.view_for_sampled_2d_array(frame.multiview_stereo);
        let bind_group = self.pipelines.bind_group(ctx.device, &hdr_sample_view);
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}

/// Resolves the wgpu format the ACES color attachment is bound to this frame.
///
/// Reads the resolved transient texture's actual `wgpu::Texture` format so the pipeline cache
/// keys correctly when the scene HDR format changes (e.g. `RGBA16Float` → `RG11B10Float` via the
/// renderer config HUD). Falls back to [`wgpu::TextureFormat::Rgba16Float`] when the chain
/// transient is missing, matching the default forward HDR format.
fn output_attachment_format(
    output: TextureHandle,
    ctx: &GraphRasterPassContext<'_, '_>,
) -> wgpu::TextureFormat {
    ctx.graph_resources
        .and_then(|gr| gr.transient_texture(output))
        .map(|tex| tex.texture.format())
        .unwrap_or(wgpu::TextureFormat::Rgba16Float)
}

/// Effect adapter so the ACES pass can be inserted into a [`crate::render_graph::post_processing::PostProcessChain`].
pub struct AcesTonemapEffect;

impl PostProcessEffect for AcesTonemapEffect {
    fn id(&self) -> PostProcessEffectId {
        PostProcessEffectId::AcesTonemap
    }

    fn is_enabled(&self, settings: &PostProcessingSettings) -> bool {
        crate::render_graph::post_processing::PostProcessChainSignature::from_settings(settings)
            .aces_tonemap
    }

    fn build_pass(&self, input: TextureHandle, output: TextureHandle) -> Box<dyn RenderPass> {
        Box::new(AcesTonemapPass::new(AcesTonemapGraphResources {
            input,
            output,
        }))
    }
}

#[cfg(test)]
mod setup_tests {
    use super::*;
    use crate::render_graph::pass::PassBuilder;
    use crate::render_graph::resources::{
        AccessKind, TransientArrayLayers, TransientExtent, TransientSampleCount,
        TransientTextureDesc, TransientTextureFormat,
    };
    use crate::render_graph::GraphBuilder;

    fn hdr_transient(builder: &mut GraphBuilder, label: &'static str) -> TextureHandle {
        builder.create_texture(TransientTextureDesc {
            label,
            format: TransientTextureFormat::SceneColorHdr,
            extent: TransientExtent::Custom {
                width: 4,
                height: 4,
            },
            mip_levels: 1,
            sample_count: TransientSampleCount::Fixed(1),
            dimension: wgpu::TextureDimension::D2,
            array_layers: TransientArrayLayers::Fixed(1),
            base_usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            alias: true,
        })
    }

    #[test]
    fn setup_declares_sampled_input_and_color_attachment() {
        let mut builder = GraphBuilder::new();
        let input = hdr_transient(&mut builder, "aces_input");
        let output = hdr_transient(&mut builder, "aces_output");
        let mut pass = AcesTonemapPass::new(AcesTonemapGraphResources { input, output });
        let mut b = PassBuilder::new("AcesTonemap");
        pass.setup(&mut b).expect("setup");
        let setup = b.finish().expect("finish");
        assert_eq!(setup.kind, crate::render_graph::pass::PassKind::Raster);
        assert!(
            setup.accesses.iter().any(|a| matches!(
                &a.access,
                AccessKind::Texture(TextureAccess::Sampled {
                    stages: wgpu::ShaderStages::FRAGMENT,
                    ..
                })
            )),
            "expected sampled HDR input read"
        );
        assert_eq!(setup.color_attachments.len(), 1);
    }

    #[test]
    fn aces_tonemap_effect_id_label() {
        let e = AcesTonemapEffect;
        assert_eq!(e.id(), PostProcessEffectId::AcesTonemap);
    }
}
