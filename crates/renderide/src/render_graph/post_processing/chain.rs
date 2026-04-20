//! Post-processing chain: ordered effects + graph wiring helpers.
//!
//! The chain owns a list of [`PostProcessEffect`] trait objects in execution order. At graph
//! build time, [`PostProcessChain::build_into_graph`] walks the enabled effects and inserts one
//! pass per effect, ping-ponging between two HDR transients so multi-effect chains never need
//! more than two intermediate textures.
//!
//! When the chain is empty (master toggle off, or all effects disabled), nothing is added to the
//! graph and [`ChainOutput::PassThrough`] is returned so the caller can wire the original input
//! handle straight into [`crate::render_graph::passes::SceneColorComposePass`].

use crate::config::{PostProcessingSettings, TonemapMode};
use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::ids::PassId;
use crate::render_graph::resources::{
    TextureHandle, TransientArrayLayers, TransientExtent, TransientSampleCount,
    TransientTextureDesc, TransientTextureFormat,
};

use super::effect::{PostProcessEffect, PostProcessEffectId};

/// Topology fingerprint for the post-processing chain at graph compile time.
///
/// Captures only what changes the *shape* of the graph (effect added/removed). Effect parameter
/// tweaks that only update uniforms do **not** flip this signature, so the
/// [`crate::render_graph::cache::GraphCache`] keeps the same compiled graph and only re-records
/// the per-frame uniforms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PostProcessChainSignature {
    /// Stephen Hill ACES Fitted tonemap pass active.
    pub aces_tonemap: bool,
}

impl PostProcessChainSignature {
    /// Derives the signature from live [`PostProcessingSettings`].
    ///
    /// Mirrors what each registered [`PostProcessEffect::is_enabled`] would return so the chain
    /// and the cache key never disagree.
    pub fn from_settings(settings: &PostProcessingSettings) -> Self {
        let master = settings.enabled;
        Self {
            aces_tonemap: master && matches!(settings.tonemap.mode, TonemapMode::AcesFitted),
        }
    }

    /// Returns `true` when no effects are active and the chain should be skipped entirely.
    pub fn is_empty(self) -> bool {
        !self.aces_tonemap
    }

    /// Number of active effects.
    pub fn active_count(self) -> usize {
        usize::from(self.aces_tonemap)
    }
}

/// Result of [`PostProcessChain::build_into_graph`].
#[derive(Clone, Copy, Debug)]
pub enum ChainOutput {
    /// No effects ran; the chain forwards the original input handle.
    PassThrough(TextureHandle),
    /// One or more effects ran; the chain output and pass-id range are returned so the caller
    /// can wire explicit edges (e.g. `add_edge(upstream, first_pass)` and
    /// `add_edge(last_pass, downstream)`).
    Chained {
        /// Final HDR output of the chain (the texture the next consumer should sample).
        final_handle: TextureHandle,
        /// First pass added by the chain (use as the destination for an upstream edge).
        first_pass: PassId,
        /// Last pass added by the chain (use as the source for a downstream edge).
        last_pass: PassId,
    },
}

impl ChainOutput {
    /// Returns the final HDR handle the next consumer should read.
    pub fn final_handle(self) -> TextureHandle {
        match self {
            Self::PassThrough(h) => h,
            Self::Chained { final_handle, .. } => final_handle,
        }
    }

    /// Returns the first/last pass ids when the chain produced any pass.
    pub fn pass_range(self) -> Option<(PassId, PassId)> {
        match self {
            Self::PassThrough(_) => None,
            Self::Chained {
                first_pass,
                last_pass,
                ..
            } => Some((first_pass, last_pass)),
        }
    }
}

/// Ordered, configurable list of [`PostProcessEffect`] trait objects.
///
/// Effects are evaluated in registration order. Use [`Self::with_default_effects`] to get the
/// canonical chain that ships with the renderer.
pub struct PostProcessChain {
    effects: Vec<Box<dyn PostProcessEffect>>,
}

impl PostProcessChain {
    /// Empty chain (no effects). Mostly useful for tests.
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    /// Pushes an effect onto the chain.
    pub fn push(&mut self, effect: Box<dyn PostProcessEffect>) {
        self.effects.push(effect);
    }

    /// Iterates over effect identities in execution order.
    pub fn effect_ids(&self) -> impl Iterator<Item = PostProcessEffectId> + '_ {
        self.effects.iter().map(|e| e.id())
    }

    /// Number of registered effects (regardless of enable state).
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Whether the chain has zero registered effects.
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// Inserts the chain's enabled passes into `builder`, returning the wiring info.
    ///
    /// When all effects are disabled (or the master toggle is off), no transients are allocated
    /// and the input handle is forwarded.
    ///
    /// Logs at `info` when one or more effects are wired into the graph so a developer can see
    /// (in the renderer log) which compile-time passes the chain produced.
    pub fn build_into_graph(
        &self,
        builder: &mut GraphBuilder,
        input: TextureHandle,
        settings: &PostProcessingSettings,
    ) -> ChainOutput {
        if !settings.enabled || !self.effects.iter().any(|e| e.is_enabled(settings)) {
            return ChainOutput::PassThrough(input);
        }

        let active: Vec<&'static str> = self
            .effects
            .iter()
            .filter(|e| e.is_enabled(settings))
            .map(|e| e.id().label())
            .collect();
        logger::info!(
            "post-processing chain: {} effect(s) active: {}",
            active.len(),
            active.join(", ")
        );

        let ping = builder.create_texture(post_process_color_transient_desc(
            "post_processed_color_hdr_a",
        ));
        let pong = builder.create_texture(post_process_color_transient_desc(
            "post_processed_color_hdr_b",
        ));

        let mut current_in = input;
        let mut current_out = ping;
        let mut first_pass: Option<PassId> = None;
        let mut last_pass: Option<PassId> = None;

        for (i, effect) in self
            .effects
            .iter()
            .filter(|e| e.is_enabled(settings))
            .enumerate()
        {
            let pass_id = builder.add_pass(effect.build_pass(current_in, current_out));
            first_pass.get_or_insert(pass_id);
            last_pass = Some(pass_id);

            if i == 0 {
                current_in = ping;
                current_out = pong;
            } else {
                std::mem::swap(&mut current_in, &mut current_out);
            }
        }

        let first_pass = first_pass.expect("active_effects is non-empty");
        let last_pass = last_pass.expect("active_effects is non-empty");
        ChainOutput::Chained {
            final_handle: current_in,
            first_pass,
            last_pass,
        }
    }
}

impl Default for PostProcessChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Descriptor for a single ping-pong slot in the chain.
///
/// HDR float, single-sample, frame-array-layers (1 mono / 2 stereo), aliasable. Matches the
/// `scene_color_hdr` transient so subsequent effects can read its predecessor's output without a
/// format change.
fn post_process_color_transient_desc(label: &'static str) -> TransientTextureDesc {
    TransientTextureDesc {
        label,
        format: TransientTextureFormat::SceneColorHdr,
        extent: TransientExtent::Backbuffer,
        mip_levels: 1,
        sample_count: TransientSampleCount::Fixed(1),
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
        base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        alias: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TonemapSettings;
    use crate::render_graph::error::{RenderPassError, SetupError};
    use crate::render_graph::pass::{PassBuilder, RenderPass};

    /// Test-only effect: builds a no-op pass that records sampled-read + color-attachment intent
    /// so the graph builder accepts it as raster.
    struct MockEffect {
        id: PostProcessEffectId,
        enabled: bool,
    }

    impl PostProcessEffect for MockEffect {
        fn id(&self) -> PostProcessEffectId {
            self.id
        }

        fn is_enabled(&self, _settings: &PostProcessingSettings) -> bool {
            self.enabled
        }

        fn build_pass(&self, input: TextureHandle, output: TextureHandle) -> Box<dyn RenderPass> {
            Box::new(MockPass {
                name: self.id.label(),
                input,
                output,
            })
        }
    }

    struct MockPass {
        name: &'static str,
        input: TextureHandle,
        output: TextureHandle,
    }

    impl RenderPass for MockPass {
        fn name(&self) -> &str {
            self.name
        }

        fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
            use crate::render_graph::resources::TextureAccess;
            b.read_texture_resource(
                self.input,
                TextureAccess::Sampled {
                    stages: wgpu::ShaderStages::FRAGMENT,
                },
            );
            let mut r = b.raster();
            r.color(
                self.output,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                Option::<TextureHandle>::None,
            );
            Ok(())
        }

        fn execute(
            &mut self,
            _ctx: &mut crate::render_graph::context::RenderPassContext<'_, '_, '_>,
        ) -> Result<(), RenderPassError> {
            Ok(())
        }
    }

    fn fake_input(builder: &mut GraphBuilder) -> TextureHandle {
        builder.create_texture(post_process_color_transient_desc("scene_color_hdr"))
    }

    #[test]
    fn empty_chain_returns_pass_through() {
        let mut builder = GraphBuilder::new();
        let input = fake_input(&mut builder);
        let chain = PostProcessChain::new();
        let settings = PostProcessingSettings {
            enabled: true,
            ..Default::default()
        };
        let out = chain.build_into_graph(&mut builder, input, &settings);
        assert!(matches!(out, ChainOutput::PassThrough(h) if h == input));
    }

    #[test]
    fn disabled_master_returns_pass_through_even_with_effects() {
        let mut builder = GraphBuilder::new();
        let input = fake_input(&mut builder);
        let mut chain = PostProcessChain::new();
        chain.push(Box::new(MockEffect {
            id: PostProcessEffectId::AcesTonemap,
            enabled: true,
        }));
        let settings = PostProcessingSettings {
            enabled: false,
            ..Default::default()
        };
        let out = chain.build_into_graph(&mut builder, input, &settings);
        assert!(matches!(out, ChainOutput::PassThrough(h) if h == input));
    }

    #[test]
    fn single_enabled_effect_creates_one_pass_and_chains_handles() {
        let mut builder = GraphBuilder::new();
        let input = fake_input(&mut builder);
        let mut chain = PostProcessChain::new();
        chain.push(Box::new(MockEffect {
            id: PostProcessEffectId::AcesTonemap,
            enabled: true,
        }));
        let settings = PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::AcesFitted,
            },
        };
        let out = chain.build_into_graph(&mut builder, input, &settings);
        match out {
            ChainOutput::Chained {
                final_handle,
                first_pass,
                last_pass,
            } => {
                assert_ne!(
                    final_handle, input,
                    "final handle must be a chain transient"
                );
                assert_eq!(
                    first_pass, last_pass,
                    "single effect produces a single pass"
                );
            }
            other => panic!("expected Chained variant, got {other:?}"),
        }
    }

    #[test]
    fn multiple_effects_ping_pong_to_pong_slot() {
        let mut builder = GraphBuilder::new();
        let input = fake_input(&mut builder);
        let mut chain = PostProcessChain::new();
        chain.push(Box::new(MockEffect {
            id: PostProcessEffectId::AcesTonemap,
            enabled: true,
        }));
        chain.push(Box::new(MockEffect {
            id: PostProcessEffectId::AcesTonemap,
            enabled: true,
        }));
        let settings = PostProcessingSettings {
            enabled: true,
            ..Default::default()
        };
        let out = chain.build_into_graph(&mut builder, input, &settings);
        match out {
            ChainOutput::Chained {
                final_handle,
                first_pass,
                last_pass,
            } => {
                assert_ne!(final_handle, input);
                assert_ne!(first_pass, last_pass);
            }
            other => panic!("expected Chained variant, got {other:?}"),
        }
    }

    #[test]
    fn signature_from_settings_matches_master_toggle() {
        let mut s = PostProcessingSettings {
            enabled: false,
            tonemap: TonemapSettings {
                mode: TonemapMode::AcesFitted,
            },
        };
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());

        s.enabled = true;
        let sig = PostProcessChainSignature::from_settings(&s);
        assert!(sig.aces_tonemap);
        assert_eq!(sig.active_count(), 1);

        s.tonemap.mode = TonemapMode::None;
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());
    }

    #[test]
    fn chain_output_helpers() {
        let h = TextureHandle(7);
        let pt = ChainOutput::PassThrough(h);
        assert_eq!(pt.final_handle(), h);
        assert!(pt.pass_range().is_none());

        let chained = ChainOutput::Chained {
            final_handle: h,
            first_pass: PassId(1),
            last_pass: PassId(2),
        };
        assert_eq!(chained.final_handle(), h);
        assert_eq!(chained.pass_range(), Some((PassId(1), PassId(2))));
    }
}
