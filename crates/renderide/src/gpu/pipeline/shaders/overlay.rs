//! WGSL shader source for overlay stencil pipelines, sourced from `RENDERIDESHADERS`.

pub(crate) const OVERLAY_STENCIL_SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/RENDERIDESHADERS/overlay/overlay_stencil.wgsl"
));
