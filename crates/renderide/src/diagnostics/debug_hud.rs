//! Dear ImGui overlay for developer diagnostics (feature `debug-hud`).

use super::renderer_info_snapshot::RendererInfoSnapshot;
use super::DebugHudInput;

#[cfg(feature = "debug-hud")]
use std::time::{Duration, Instant};

#[cfg(feature = "debug-hud")]
use imgui::{
    Condition, Context, FontConfig, FontSource, Io, MouseButton as ImGuiMouseButton, WindowFlags,
};
#[cfg(feature = "debug-hud")]
use imgui_wgpu::{Renderer as ImguiWgpuRenderer, RendererConfig};

/// Optional GPU debug overlay (ImGui + imgui-wgpu).
#[cfg(feature = "debug-hud")]
pub struct DebugHud {
    imgui: Context,
    renderer: ImguiWgpuRenderer,
    last_frame_at: Instant,
    latest: Option<RendererInfoSnapshot>,
}

#[cfg(feature = "debug-hud")]
fn device_type_label(kind: wgpu::DeviceType) -> &'static str {
    match kind {
        wgpu::DeviceType::Other => "other / unknown",
        wgpu::DeviceType::IntegratedGpu => "integrated GPU",
        wgpu::DeviceType::DiscreteGpu => "discrete GPU",
        wgpu::DeviceType::VirtualGpu => "virtual GPU",
        wgpu::DeviceType::Cpu => "software / CPU",
    }
}

#[cfg(feature = "debug-hud")]
fn apply_input(io: &mut Io, input: &DebugHudInput) {
    if input.mouse_active && input.window_focused {
        io.add_mouse_pos_event(input.cursor_px);
    } else {
        io.add_mouse_pos_event([-f32::MAX, -f32::MAX]);
    }
    io.add_mouse_button_event(ImGuiMouseButton::Left, input.left);
    io.add_mouse_button_event(ImGuiMouseButton::Right, input.right);
    io.add_mouse_button_event(ImGuiMouseButton::Middle, input.middle);
    io.add_mouse_button_event(ImGuiMouseButton::Extra1, input.extra1);
    io.add_mouse_button_event(ImGuiMouseButton::Extra2, input.extra2);
}

#[cfg(feature = "debug-hud")]
impl DebugHud {
    /// Builds ImGui and the wgpu render backend for the swapchain format.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let mut imgui = Context::create();
        imgui.set_ini_filename(None);
        imgui.set_log_filename(None);
        imgui.io_mut().config_windows_move_from_title_bar_only = true;
        imgui.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(FontConfig {
                oversample_h: 2,
                pixel_snap_h: true,
                size_pixels: 14.0,
                ..FontConfig::default()
            }),
        }]);

        let mut renderer_config = RendererConfig::new();
        renderer_config.texture_format = surface_format;
        let renderer = ImguiWgpuRenderer::new(&mut imgui, device, queue, renderer_config);

        Self {
            imgui,
            renderer,
            last_frame_at: Instant::now(),
            latest: None,
        }
    }

    /// Stores the snapshot shown under the **Renderer** tab.
    pub fn set_snapshot(&mut self, sample: RendererInfoSnapshot) {
        self.latest = Some(sample);
    }

    /// Records ImGui into `encoder` as a load-on-top pass over `backbuffer`.
    pub fn encode_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
        (width, height): (u32, u32),
        input: &DebugHudInput,
    ) -> Result<(), String> {
        let delta = self.last_frame_at.elapsed().max(Duration::from_millis(1));
        self.last_frame_at = Instant::now();

        let io = self.imgui.io_mut();
        io.display_size = [width as f32, height as f32];
        io.display_framebuffer_scale = [1.0, 1.0];
        io.update_delta_time(delta);
        apply_input(io, input);

        let snapshot = self.latest.clone();
        let ui = self.imgui.frame();
        const PANEL_WIDTH: f32 = 520.0;
        let panel_x = (width as f32 - PANEL_WIDTH - 12.0).max(12.0);
        let window_flags = WindowFlags::ALWAYS_AUTO_RESIZE
            | WindowFlags::NO_SCROLLBAR
            | WindowFlags::NO_RESIZE
            | WindowFlags::NO_SAVED_SETTINGS
            | WindowFlags::NO_FOCUS_ON_APPEARING
            | WindowFlags::NO_NAV;

        ui.window("Renderide debug")
            .position([panel_x, 12.0], Condition::FirstUseEver)
            .size_constraints([PANEL_WIDTH, 0.0], [PANEL_WIDTH, 1.0e9])
            .bg_alpha(0.72)
            .flags(window_flags)
            .build(|| {
                if let Some(_tab_bar) = ui.tab_bar("debug_tabs") {
                    if let Some(_tab) = ui.tab_item("Renderer") {
                        Self::renderer_tab(ui, snapshot.as_ref());
                    }
                }
            });

        let draw_data = self.imgui.render();
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("imgui-debug-hud"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: backbuffer,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            self.renderer
                .render(draw_data, queue, device, &mut pass)
                .map_err(|e| format!("imgui-wgpu render: {e}"))?;
        }
        Ok(())
    }

    fn renderer_tab(ui: &imgui::Ui, sample: Option<&RendererInfoSnapshot>) {
        let Some(s) = sample else {
            ui.text("Waiting for snapshot…");
            return;
        };

        let fps = if s.frame_time_ms > f64::EPSILON {
            1000.0 / s.frame_time_ms
        } else {
            0.0
        };
        ui.text(format!(
            "FPS {:.1}  |  frame {:.2} ms",
            fps, s.frame_time_ms
        ));
        ui.text(format!(
            "Frame index {}  |  viewport {}×{}",
            s.last_frame_index, s.viewport_px.0, s.viewport_px.1
        ));

        ui.separator();
        ui.text("IPC / init");
        ui.text(format!(
            "Connected: {}  |  init: {:?}",
            s.ipc_connected, s.init_state
        ));

        ui.separator();
        ui.text("GPU (adapter)");
        ui.text_wrapped(format!("Name: {}", s.adapter_name));
        ui.text(format!(
            "Class: {}  |  backend: {:?}",
            device_type_label(s.adapter_device_type),
            s.adapter_backend
        ));
        ui.text_wrapped(format!(
            "Driver: {} ({})",
            s.adapter_driver, s.adapter_driver_info
        ));
        ui.text(format!(
            "Surface: {:?}  |  present: {:?}",
            s.surface_format, s.present_mode
        ));

        ui.separator();
        ui.text("Scene");
        ui.text(format!("Render spaces: {}", s.render_space_count));
        ui.text(format!(
            "Mesh renderables (CPU tables): {}",
            s.mesh_renderable_count
        ));

        ui.separator();
        ui.text("Resources");
        ui.text(format!(
            "GPU meshes: {}  |  GPU textures: {}",
            s.resident_mesh_count, s.resident_texture_count
        ));

        ui.separator();
        ui.text("Materials (property store)");
        ui.text(format!(
            "Material property maps: {}  |  property blocks: {}  |  shader bindings: {}",
            s.material_property_slots, s.property_block_slots, s.material_shader_bindings
        ));

        ui.separator();
        ui.text("Frame");
        ui.text(format!(
            "Render graph passes: {}  |  GPU lights (packed): {}",
            s.frame_graph_pass_count, s.gpu_light_count
        ));
    }
}

#[cfg(not(feature = "debug-hud"))]
/// Stub when `debug-hud` is disabled.
#[derive(Debug, Default)]
pub struct DebugHud;

#[cfg(not(feature = "debug-hud"))]
impl DebugHud {
    /// No-op without ImGui.
    pub fn new(
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _surface_format: wgpu::TextureFormat,
    ) -> Self {
        Self
    }

    /// No-op without ImGui.
    pub fn set_snapshot(&mut self, _sample: RendererInfoSnapshot) {}

    /// No-op without ImGui.
    pub fn encode_overlay(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        _backbuffer: &wgpu::TextureView,
        _extent: (u32, u32),
        _input: &DebugHudInput,
    ) -> Result<(), String> {
        Ok(())
    }
}
