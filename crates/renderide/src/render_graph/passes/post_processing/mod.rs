//! Concrete post-processing render passes registered on the
//! [`crate::render_graph::post_processing::PostProcessChain`].
//!
//! Today the chain ships with a single effect: [`AcesTonemapPass`] (Stephen Hill ACES Fitted).
//! Future effects (bloom, color grading, etc.) live alongside it as sibling sub-modules and
//! implement [`crate::render_graph::post_processing::PostProcessEffect`].

mod aces_tonemap;

pub use aces_tonemap::{AcesTonemapEffect, AcesTonemapGraphResources, AcesTonemapPass};
