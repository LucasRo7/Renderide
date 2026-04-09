//! Builds a temporal Hi-Z pyramid from the main depth buffer and schedules CPU readback.

use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::{PassResources, ResourceSlot};

/// GPU Hi-Z reduction after the world forward pass (single-layer depth only).
#[derive(Debug, Default)]
pub struct HiZBuildPass;

impl HiZBuildPass {
    /// Creates the pass node.
    pub fn new() -> Self {
        Self
    }
}

impl RenderPass for HiZBuildPass {
    fn name(&self) -> &str {
        "HiZBuild"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: vec![ResourceSlot::Depth],
            writes: vec![],
        }
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_>) -> Result<(), RenderPassError> {
        let Some(depth) = ctx.depth_view else {
            return Err(RenderPassError::MissingDepth {
                pass: self.name().to_string(),
            });
        };
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };

        let backend = &mut frame.backend;
        let scene = frame.scene;
        let viewport_px = frame.viewport_px;
        let hc = frame.host_camera;
        let multiview = frame.multiview_stereo;

        let queue_guard = ctx
            .queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        backend.hi_z_encode_end_of_frame(
            ctx.device,
            &queue_guard,
            ctx.encoder,
            depth,
            viewport_px,
            scene,
            viewport_px,
            &hc,
            multiview,
        );
        Ok(())
    }
}
