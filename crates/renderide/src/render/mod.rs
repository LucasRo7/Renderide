//! Render loop, draw batching, and render graph.
//!
//! ## Overview
//!
//! **Render loop** ([`RenderLoop`]): drives one frame via the graph. Main window uses
//! [`RenderTarget::Surface`]; offscreen camera tasks use [`RenderTarget::Offscreen`].
//!
//! **Draw batching** ([`SpaceDrawBatch`]): per-space batches with `view_transform`, sorted by
//! `sort_key` (higher renders on top; matches Unity `sortingOrder`). Draws are ordered by
//! `(is_overlay, -sort_key, pipeline_variant, material_id, mesh_asset_id)`.
//!
//! **Layer filtering**: [`collect_draw_batches`](crate::session::Session::collect_draw_batches)
//! skips `Hidden` layers. Main view excludes private overlays; `render_private_ui` in
//! [`CameraRenderTask`](crate::shared::CameraRenderTask) controls private overlay inclusion.
//!
//! **Projection**: Orthographic for UI via [`projection_for_params`](pass::projection_for_params);
//! perspective uses reverse-Z for depth.
//!
//! **Overlay pass**: Render overlays after meshes with `LoadOp::Load` (preserve framebuffer) and
//! alpha blend so UI composites over the scene.
//!
//! **RenderTaskExecutor**: Runs [`CameraRenderTask`](crate::shared::CameraRenderTask)s offscreen
//! and copies pixels to shared memory for the host.
//!
//! ## Extension points
//!
//! - **UIPass**: Add a [`RenderPass`] that runs after [`MeshRenderPass`] with `LoadOp::Load` and
//!   alpha blending; bind UI textures via [`RenderPassContext`].
//! - **Texture binding for UI materials**: Extend pipeline bind groups or add a pass that binds
//!   atlas textures for UI sprites.
//! - **Stencil for GraphicsChunk masking**: Use stencil buffer in a pass to mask UI regions
//!   (e.g. scroll rects) before drawing overlays.

pub mod batch;
pub mod context;
pub mod pass;
pub mod r#loop;
pub mod task;
pub mod target;

pub use batch::{DrawEntry, SpaceDrawBatch};
pub use context::{current_context, set_context, with_context, FramePhase};
pub use crate::shared::RenderingContext;
pub use pass::{
    MeshRenderPass, RenderGraph, RenderGraphContext, RenderPass, RenderPassContext, RenderPassError,
    RenderTargetViews,
};
pub use r#loop::RenderLoop;
pub use task::RenderTaskExecutor;
pub use target::RenderTarget;
