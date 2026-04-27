//! Cached compiled frame graph and rebuild signature tracking.

use crate::config::PostProcessingSettings;
use crate::render_graph::post_processing::PostProcessChainSignature;
use crate::render_graph::CompiledRenderGraph;

/// Owns the compiled frame graph and the settings signature it was built from.
pub(super) struct FrameGraphCache {
    /// Compiled DAG of render passes, or `None` when build failed or attach has not run.
    graph: Option<CompiledRenderGraph>,
    /// Post-processing topology signature used to build [`Self::graph`].
    post_processing_signature: PostProcessChainSignature,
    /// Effective MSAA sample count used to build [`Self::graph`].
    msaa_sample_count: u8,
}

impl Default for FrameGraphCache {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameGraphCache {
    /// Creates an empty cache.
    pub(super) fn new() -> Self {
        Self {
            graph: None,
            post_processing_signature: PostProcessChainSignature::default(),
            msaa_sample_count: 1,
        }
    }

    /// Returns the cached graph's scheduled pass count.
    pub(super) fn pass_count(&self) -> usize {
        self.graph.as_ref().map_or(0, |graph| graph.pass_count())
    }

    /// Returns the cached graph's topological wave count.
    pub(super) fn topo_levels(&self) -> usize {
        self.graph
            .as_ref()
            .map_or(0, |graph| graph.compile_stats.topo_levels)
    }

    /// Returns the cached post-processing signature.
    pub(super) fn post_processing_signature(&self) -> PostProcessChainSignature {
        self.post_processing_signature
    }

    /// Returns the cached MSAA sample count.
    pub(super) fn msaa_sample_count(&self) -> u8 {
        self.msaa_sample_count
    }

    /// Returns true when the current cache is missing or was built for different inputs.
    pub(super) fn needs_rebuild(
        &self,
        post_processing: &PostProcessingSettings,
        msaa_sample_count: u8,
    ) -> bool {
        self.graph.is_none()
            || self.post_processing_signature
                != PostProcessChainSignature::from_settings(post_processing)
            || self.msaa_sample_count != msaa_sample_count
    }

    /// Rebuilds the graph for `post_processing` and `msaa_sample_count`.
    pub(super) fn rebuild(
        &mut self,
        post_processing: &PostProcessingSettings,
        msaa_sample_count: u8,
    ) {
        match crate::render_graph::build_default_main_graph_with(post_processing, msaa_sample_count)
        {
            Ok(graph) => {
                self.graph = Some(graph);
                self.post_processing_signature =
                    PostProcessChainSignature::from_settings(post_processing);
                self.msaa_sample_count = msaa_sample_count;
            }
            Err(error) => {
                logger::warn!("render graph build failed: {error}");
                self.graph = None;
            }
        }
    }

    /// Takes the graph for exclusive execution.
    pub(super) fn take_for_execution(&mut self) -> Option<CompiledRenderGraph> {
        self.graph.take()
    }

    /// Restores a graph after execution.
    pub(super) fn restore_after_execution(&mut self, graph: CompiledRenderGraph) {
        self.graph = Some(graph);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{TonemapMode, TonemapSettings};

    #[test]
    fn missing_graph_needs_rebuild() {
        let cache = FrameGraphCache::new();
        assert!(cache.needs_rebuild(&PostProcessingSettings::default(), 1));
    }

    #[test]
    fn rebuild_records_signature_and_msaa() {
        let mut cache = FrameGraphCache::new();
        let settings = PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::AcesFitted,
            },
            ..Default::default()
        };

        cache.rebuild(&settings, 4);

        assert_eq!(
            cache.post_processing_signature(),
            PostProcessChainSignature::from_settings(&settings)
        );
        assert_eq!(cache.msaa_sample_count(), 4);
        assert!(!cache.needs_rebuild(&settings, 4));
    }
}
