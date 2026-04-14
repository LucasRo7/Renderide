//! Embedded raster materials: WGSL reflection, texture resolution, uniform packing, and `@group(1)` bind groups.

mod layout;
mod material_bind;
mod texture_resolve;
mod uniform_pack;

pub use material_bind::EmbeddedMaterialBindResources;
