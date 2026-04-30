//! Hi-Z occlusion subsystem: CPU helpers (mip layout, readback unpacking, screen-space tests),
//! GPU pyramid build (`gpu`), and the [`OcclusionSystem`] facade that owns per-view temporal state.

pub mod gpu;
pub(crate) mod hi_z_cpu;
pub(crate) mod hi_z_occlusion;
mod system;

pub use gpu::{HiZBuildRecord, HiZGpuState, HiZHistoryTarget, encode_hi_z_build};
pub use hi_z_cpu::{
    HI_Z_PYRAMID_MAX_LONG_EDGE, HiZCpuSnapshot, HiZCullData, HiZStereoCpuSnapshot,
    hi_z_pyramid_dimensions, hi_z_snapshot_from_linear_linear, mip_dimensions,
    mip_levels_for_extent, unpack_linear_rows_to_mips,
};
pub use hi_z_occlusion::{
    hi_z_view_proj_matrices, mesh_fully_occluded_in_hiz, stereo_hiz_keeps_draw,
};
pub(crate) use system::HiZBuildInput;
pub use system::OcclusionSystem;
