//! Post-processing tab body for the renderer config window.
//!
//! Exposes the master enable toggle and tonemap mode dropdown. The whole
//! [`crate::config::RendererSettings`] struct (including the `[post_processing]` table) is saved
//! by the parent panel whenever any tab marks it dirty.

use crate::config::TonemapMode;

/// Renders the **Post-Processing** tab. Marks `dirty = true` when any control changes.
///
/// Shape mirrors the existing `renderer_config_*_section` helpers: an indented section with
/// `text_disabled` callouts so the tab reads consistently with Display / Rendering / Debug.
pub(super) fn renderer_config_post_processing_tab(
    ui: &imgui::Ui,
    g: &mut crate::config::RendererSettings,
    dirty: &mut bool,
) {
    ui.text("Post-Processing");
    ui.indent();
    if ui.checkbox(
        "Enable post-processing stack",
        &mut g.post_processing.enabled,
    ) {
        *dirty = true;
    }
    ui.text_disabled(
        "Master toggle for the post-processing chain (HDR scene color → display target). \
         Applied on the next frame (the render graph is rebuilt automatically when the chain \
         topology changes).",
    );

    ui.separator();
    ui.text_disabled("Tonemap (HDR linear → display-referred 0..1 linear).");
    for (i, &mode) in TonemapMode::ALL.iter().enumerate() {
        let _id = ui.push_id_int(i as i32);
        if ui
            .selectable_config(mode.label())
            .selected(g.post_processing.tonemap.mode == mode)
            .build()
        {
            g.post_processing.tonemap.mode = mode;
            *dirty = true;
        }
    }
    ui.text_disabled(
        "ACES Fitted is the high-quality reference curve used by AAA pipelines. \
         `None` skips tonemapping (HDR pass-through; values >1 will clip in the swapchain).",
    );
    ui.unindent();
}
