//! Optional Dear ImGui diagnostics overlay and frame snapshots for HUD tabs.
//!
//! Enable with the `debug-hud` Cargo feature (on by default). Disable with
//! `cargo build -p renderide --no-default-features` for lean builds without `imgui`.

mod debug_hud;
mod renderer_info_snapshot;

pub use debug_hud::DebugHud;
pub use renderer_info_snapshot::RendererInfoSnapshot;

/// Pointer and window hints for ImGui, in **physical** pixels where noted.
#[derive(Clone, Copy, Debug, Default)]
pub struct DebugHudInput {
    /// Cursor position in physical pixels (or `[-∞, -∞]` when unavailable).
    pub cursor_px: [f32; 2],
    /// Drawable size in physical pixels.
    pub window_px: (u32, u32),
    pub window_focused: bool,
    pub mouse_active: bool,
    pub left: bool,
    pub right: bool,
    pub middle: bool,
    pub extra1: bool,
    pub extra2: bool,
}

impl DebugHudInput {
    /// Builds input for the HUD from winit and the accumulated window/input state.
    ///
    /// Cursor uses logical client coords from [`WindowInputAccumulator`](crate::frontend::input::WindowInputAccumulator)
    /// scaled to physical pixels via [`winit::window::Window::scale_factor`] (winit 0.30 has no `cursor_position` reader).
    pub fn from_winit(
        window: &winit::window::Window,
        acc: &crate::frontend::input::WindowInputAccumulator,
    ) -> Self {
        let sf = window.scale_factor() as f32;
        let cursor_px = if acc.mouse_active && acc.window_focused {
            [acc.window_position.x * sf, acc.window_position.y * sf]
        } else {
            [-f32::MAX, -f32::MAX]
        };
        let window_px = acc.window_resolution;
        Self {
            cursor_px,
            window_px,
            window_focused: acc.window_focused,
            mouse_active: acc.mouse_active,
            left: acc.left_held,
            right: acc.right_held,
            middle: acc.middle_held,
            extra1: acc.button4_held,
            extra2: acc.button5_held,
        }
    }
}
