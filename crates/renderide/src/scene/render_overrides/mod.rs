//! Render-context override state mirrored from host `RenderTransformOverride*` / `RenderMaterialOverride*`.

mod apply;
mod space_impl;
mod types;

#[cfg(test)]
mod tests;

pub(crate) use apply::{
    apply_render_material_overrides_update, apply_render_transform_overrides_update,
};
pub use types::{
    MeshRendererOverrideTarget, RenderMaterialOverrideEntry, RenderTransformOverrideEntry,
};
