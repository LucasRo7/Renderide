//! Winit [`ApplicationHandler`] state: [`RendererRuntime`], lazily created window and [`GpuContext`],
//! OpenXR handles, and the per-frame tick ([`RenderideApp::tick_frame`]). See [`crate::app`] for the
//! high-level flow.
//!
//! ## Frame tick phases
//!
//! [`tick_frame`] runs these **private** stages in order (AAA-style “frame phases” / “tick stages”):
//!
//! 1. [`frame_tick_prologue`] — log level, wall-clock tick markers, GPU frame timing begin.
//! 2. [`poll_ipc_and_window`] — drain IPC; apply host output (cursor); per-frame cursor lock when requested.
//! 3. [`xr_begin_tick`] — OpenXR `wait_frame` / view poses **before** lock-step (must stay before
//!    [`lock_step_exchange`] so [`InputState::vr`] matches the same [`OpenxrFrameTick`] snapshot).
//! 4. [`lock_step_exchange`] — when allowed, [`RendererRuntime::pre_frame`] with winit input + optional VR IPC.
//! 5. Early exits — shutdown, fatal IPC, missing window/GPU (each runs epilogue timing).
//! 6. [`render_views`] — HMD multiview submit if XR+GPU; secondary cameras to render textures; vsync;
//!    debug HUD input/time for this frame (must run before desktop [`frame_loop::execute_mirror_frame_graph`]).
//! 7. [`present_and_diagnostics`] — VR mirror blit or clear vs desktop graph; OpenXR `end_frame_empty` when needed.
//! 8. [`frame_tick_epilogue`] — GPU frame timing end and debug HUD capture after the tick.
//!
//! [`tick_phase_trace`] emits `trace!` lines prefixed with [`TICK_TRACE_PREFIX`] for grep/profiling; the same
//! splits are natural boundaries for the `tracing` crate’s spans if added later.

use std::sync::Arc;
use std::time::Instant;

use logger::LogLevel;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, DeviceEvents};
use winit::window::{Window, WindowId};

use crate::frontend::input::{
    apply_device_event, apply_output_state_to_window, apply_per_frame_cursor_lock_when_locked,
    apply_window_event, vr_inputs_for_session, CursorOutputTracking, WindowInputAccumulator,
};
use crate::gpu::{GpuContext, VrMirrorBlitResources};
use crate::output_device::head_output_device_wants_openxr;
use crate::present::present_clear_frame;
use crate::render_graph::GraphExecuteError;
use crate::runtime::RendererRuntime;
use crate::shared::{HeadOutputDevice, VRControllerState};
use crate::xr::OpenxrFrameTick;
use glam::{Quat, Vec3};

use super::frame_loop;
use super::frame_pacing;
use super::startup::{
    apply_window_title_from_init, effective_output_device_for_gpu, effective_renderer_log_level,
    LOG_FLUSH_INTERVAL,
};

/// Prefix for per-phase trace lines in [`RenderideApp::tick_frame`] (grep-friendly; no log `target` in this logger).
const TICK_TRACE_PREFIX: &str = "renderide::tick";

/// Emits a trace line naming the current frame phase (see module docs).
fn tick_phase_trace(phase: &'static str) {
    logger::trace!("{} phase={phase}", TICK_TRACE_PREFIX);
}

pub(crate) struct RenderideApp {
    runtime: RendererRuntime,
    /// VSync flag used for the initial [`GpuContext::new`] before live updates from settings.
    initial_vsync: bool,
    /// GPU validation layers flag for the initial [`GpuContext::new`] (persisted; restart to apply).
    initial_gpu_validation: bool,
    /// Parsed `-LogLevel` from startup, if any. When [`Some`], always overrides [`crate::config::DebugSettings::log_verbose`].
    log_level_cli: Option<LogLevel>,
    /// Copied from host [`crate::shared::RendererInitData::output_device`] when the window is created.
    session_output_device: HeadOutputDevice,
    /// Center-eye pose for host IPC ([`crate::xr::headset_center_pose_from_stereo_views`], Unity-style
    /// [`crate::xr::openxr_pose_to_host_tracking`]), not the GPU rendering basis.
    cached_head_pose: Option<(Vec3, Quat)>,
    /// Controller states from the same XR tick’s [`crate::xr::OpenxrInput::sync_and_sample`] as `cached_head_pose`.
    cached_openxr_controllers: Vec<VRControllerState>,
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    /// Set by the winit handler; read by [`crate::app::run`] for process exit.
    pub(crate) exit_code: Option<i32>,
    last_log_flush: Option<Instant>,
    input: WindowInputAccumulator,
    /// Host cursor lock transitions (unlock warp parity with Unity mouse driver).
    cursor_output_tracking: CursorOutputTracking,
    xr_handles: Option<crate::xr::XrWgpuHandles>,
    xr_swapchain: Option<crate::xr::XrStereoSwapchain>,
    xr_stereo_depth: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// Staging texture and blit pipelines for the VR desktop mirror (left HMD eye).
    vr_mirror_blit: VrMirrorBlitResources,
    /// Previous redraw instant for HUD FPS ([`crate::diagnostics::DebugHud`]).
    hud_frame_last: Option<Instant>,
    /// Wall-clock end of the last [`Self::tick_frame`] (for desktop FPS caps).
    last_frame_end: Option<Instant>,
}

/// Reconfigures the swapchain/depth for the given physical dimensions (shared by resize path and helpers).
fn reconfigure_gpu_for_physical_size(gpu: &mut GpuContext, width: u32, height: u32) {
    gpu.reconfigure(width, height);
}

/// Reconfigures using the window’s current [`Window::inner_size`].
fn reconfigure_gpu_for_window(gpu: &mut GpuContext, window: &Window) {
    let s = window.inner_size();
    reconfigure_gpu_for_physical_size(gpu, s.width, s.height);
}

impl RenderideApp {
    /// Builds initial app state after IPC bootstrap; window and GPU are created on [`ApplicationHandler::resumed`].
    pub(crate) fn new(
        runtime: RendererRuntime,
        initial_vsync: bool,
        initial_gpu_validation: bool,
        log_level_cli: Option<LogLevel>,
    ) -> Self {
        Self {
            runtime,
            initial_vsync,
            initial_gpu_validation,
            log_level_cli,
            session_output_device: HeadOutputDevice::Screen,
            cached_head_pose: None,
            cached_openxr_controllers: Vec::new(),
            window: None,
            gpu: None,
            exit_code: None,
            last_log_flush: None,
            input: WindowInputAccumulator::default(),
            cursor_output_tracking: CursorOutputTracking::default(),
            xr_handles: None,
            xr_swapchain: None,
            xr_stereo_depth: None,
            vr_mirror_blit: VrMirrorBlitResources::new(),
            hud_frame_last: None,
            last_frame_end: None,
        }
    }

    /// Records wall-clock frame end for FPS pacing and forwards to [`RendererRuntime::tick_frame_wall_clock_end`].
    fn record_frame_tick_end(&mut self, frame_start: Instant) {
        self.last_frame_end = Some(Instant::now());
        self.runtime.tick_frame_wall_clock_end(frame_start);
    }

    fn maybe_flush_logs(&mut self) {
        let now = Instant::now();
        let should = self
            .last_log_flush
            .map(|t| now.duration_since(t) >= LOG_FLUSH_INTERVAL)
            .unwrap_or(true);
        if should {
            logger::flush();
            self.last_log_flush = Some(now);
        }
    }

    /// Applies [`effective_renderer_log_level`] from CLI and [`crate::config::DebugSettings::log_verbose`].
    fn sync_log_level_from_settings(&self) {
        let log_verbose = self
            .runtime
            .settings()
            .read()
            .map(|s| s.debug.log_verbose)
            .unwrap_or(false);
        logger::set_max_level(effective_renderer_log_level(
            self.log_level_cli,
            log_verbose,
        ));
    }

    fn ensure_window_gpu(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = winit::window::Window::default_attributes()
            .with_title("Renderide")
            .with_maximized(true)
            .with_visible(true);

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                logger::error!("create_window failed: {e}");
                self.exit_code = Some(1);
                event_loop.exit();
                return;
            }
        };

        let output_device = effective_output_device_for_gpu(self.runtime.pending_init());
        self.session_output_device = output_device;

        if let Some(init) = self.runtime.take_pending_init() {
            apply_window_title_from_init(&window, &init);
        }

        let wants_openxr = head_output_device_wants_openxr(output_device);
        if wants_openxr {
            match crate::xr::init_wgpu_openxr(self.initial_gpu_validation) {
                Ok(h) => {
                    match GpuContext::new_from_openxr_bootstrap(
                        &h.wgpu_instance,
                        &h.wgpu_adapter,
                        Arc::clone(&h.device),
                        Arc::clone(&h.queue),
                        Arc::clone(&window),
                        self.initial_vsync,
                    ) {
                        Ok(gpu) => {
                            logger::info!(
                                "GPU initialized (OpenXR Vulkan device + mirror surface)"
                            );
                            self.runtime.attach_gpu(&gpu);
                            self.gpu = Some(gpu);
                            self.xr_handles = Some(h);
                        }
                        Err(e) => {
                            logger::warn!(
                                "OpenXR mirror surface failed; falling back to desktop GPU: {e}"
                            );
                            self.init_desktop_gpu(&window, event_loop);
                        }
                    }
                }
                Err(e) => {
                    logger::warn!("OpenXR init failed; falling back to desktop: {e}");
                    self.init_desktop_gpu(&window, event_loop);
                }
            }
        } else {
            self.init_desktop_gpu(&window, event_loop);
        }

        if self.exit_code.is_some() {
            return;
        }

        self.window = Some(window);
        if let Some(w) = self.window.as_ref() {
            w.set_ime_allowed(true);
            self.input.sync_window_resolution_logical(w.as_ref());
        }
    }

    fn init_desktop_gpu(&mut self, window: &Arc<Window>, event_loop: &ActiveEventLoop) {
        match pollster::block_on(GpuContext::new(
            Arc::clone(window),
            self.initial_vsync,
            self.initial_gpu_validation,
        )) {
            Ok(gpu) => {
                logger::info!("GPU initialized (desktop)");
                self.runtime.attach_gpu(&gpu);
                self.gpu = Some(gpu);
            }
            Err(e) => {
                logger::error!("GPU init failed: {e}");
                self.exit_code = Some(1);
                event_loop.exit();
            }
        }
    }

    /// Phase: log level sync, wall-clock tick markers ([`RendererRuntime::tick_frame_wall_clock_begin`]),
    /// and [`GpuContext::begin_frame_timing`] when a device exists.
    fn frame_tick_prologue(&mut self, frame_start: Instant) {
        tick_phase_trace("frame_tick_prologue");
        self.sync_log_level_from_settings();
        self.runtime.tick_frame_wall_clock_begin(frame_start);
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.begin_frame_timing(frame_start);
        }
    }

    /// Phase: drain incoming IPC and apply host-driven window state (cursor/output) plus per-frame cursor lock.
    fn poll_ipc_and_window(&mut self) {
        tick_phase_trace("poll_ipc_and_window");
        self.runtime.poll_ipc();

        if let (Some(window), Some(out)) = (
            self.window.as_ref(),
            self.runtime.take_pending_output_state(),
        ) {
            if let Err(e) = apply_output_state_to_window(
                window.as_ref(),
                &out,
                &mut self.cursor_output_tracking,
            ) {
                logger::debug!("apply_output_state_to_window: {e:?}");
            }
        }

        if let Some(window) = self.window.as_ref() {
            if self.runtime.host_cursor_lock_requested() {
                let lock_pos = self
                    .runtime
                    .last_output_state()
                    .and_then(|s| s.lock_cursor_position);
                if let Err(e) = apply_per_frame_cursor_lock_when_locked(
                    window.as_ref(),
                    &mut self.input,
                    lock_pos,
                ) {
                    logger::debug!("apply_per_frame_cursor_lock_when_locked: {e:?}");
                }
            }
        }
    }

    /// Phase: OpenXR frame tick (view poses and sampling before lock-step). Updates cached head pose and controllers.
    ///
    /// Returns [`None`] when OpenXR is not active or the session does not produce a tick this frame.
    fn xr_begin_tick(&mut self) -> Option<OpenxrFrameTick> {
        tick_phase_trace("xr_begin_tick");
        let xr_tick = self
            .xr_handles
            .as_mut()
            .and_then(|h| frame_loop::begin_openxr_frame_tick(h, &mut self.runtime));

        if let Some(ref tick) = xr_tick {
            crate::xr::OpenxrInput::log_stereo_view_order_once(&tick.views);
            if let Some(handles) = &self.xr_handles {
                if let Some(ref input) = handles.openxr_input {
                    if handles.xr_session.session_running() {
                        match input.sync_and_sample(
                            handles.xr_session.xr_vulkan_session(),
                            handles.xr_session.stage_space(),
                            tick.predicted_display_time,
                        ) {
                            Ok(v) => self.cached_openxr_controllers = v,
                            Err(e) => logger::trace!("OpenXR input sync: {e:?}"),
                        }
                    }
                }
            }
            self.cached_head_pose =
                crate::xr::headset_center_pose_from_stereo_views(tick.views.as_slice());
            if let (Some(v0), Some(v1), Some((ipc_p, ipc_q))) =
                (tick.views.first(), tick.views.get(1), self.cached_head_pose)
            {
                // Raw OpenXR view positions (what the renderer uses for view-projection)
                let rp0 = &v0.pose.position;
                let rp1 = &v1.pose.position;
                let render_center_x = (rp0.x + rp1.x) * 0.5;
                let render_center_y = (rp0.y + rp1.y) * 0.5;
                let render_center_z = (rp0.z + rp1.z) * 0.5;
                logger::debug!(
                    "HEAD POS | render(OpenXR RH): ({:.3},{:.3},{:.3}) | ipc->host(Unity LH): ({:.3},{:.3},{:.3}) | ipc_quat: ({:.4},{:.4},{:.4},{:.4})",
                    render_center_x, render_center_y, render_center_z,
                    ipc_p.x, ipc_p.y, ipc_p.z,
                    ipc_q.x, ipc_q.y, ipc_q.z, ipc_q.w,
                );
            }
        }

        xr_tick
    }

    /// Phase: lock-step begin-frame to host when [`RendererRuntime::should_send_begin_frame`].
    fn lock_step_exchange(&mut self) {
        tick_phase_trace("lock_step_exchange");
        if self.runtime.should_send_begin_frame() {
            let lock = self.runtime.host_cursor_lock_requested();
            let mut inputs = self.input.take_input_state(lock);
            crate::diagnostics::sanitize_input_state_for_imgui_host(
                &mut inputs,
                self.runtime.debug_hud_last_want_capture_mouse(),
                self.runtime.debug_hud_last_want_capture_keyboard(),
            );
            if let Some(vr) = vr_inputs_for_session(
                self.session_output_device,
                self.cached_head_pose,
                &self.cached_openxr_controllers,
            ) {
                inputs.vr = Some(vr);
            }
            self.runtime.pre_frame(inputs);
        }
    }

    /// Phase: HMD multiview submission, secondary cameras to render textures, vsync from settings,
    /// and debug HUD input/time for this frame.
    ///
    /// Returns [`None`] if no [`GpuContext`] is available (mirror epilogue-only path). Otherwise returns
    /// whether the HMD projection layer was submitted (`hmd_projection_ended`).
    fn render_views(
        &mut self,
        window: &Arc<Window>,
        xr_tick: Option<&OpenxrFrameTick>,
    ) -> Option<bool> {
        tick_phase_trace("render_views");
        let hmd_projection_ended = match (self.gpu.as_mut(), self.xr_handles.as_mut(), xr_tick) {
            (Some(gpu), Some(handles), Some(tick)) => frame_loop::try_hmd_multiview_submit(
                gpu,
                handles,
                &mut self.runtime,
                &mut self.xr_swapchain,
                &mut self.xr_stereo_depth,
                &mut self.vr_mirror_blit,
                window.as_ref(),
                tick,
            ),
            _ => false,
        };

        let gpu = self.gpu.as_mut()?;

        if let Err(e) = self
            .runtime
            .render_secondary_cameras_to_render_textures(gpu, window.as_ref())
        {
            logger::warn!("secondary camera render-to-texture failed: {e:?}");
        }

        if let Ok(s) = self.runtime.settings().read() {
            gpu.set_vsync(s.rendering.vsync);
        }

        {
            let now = Instant::now();
            let ms = self
                .hud_frame_last
                .map(|t| now.duration_since(t).as_secs_f64() * 1000.0)
                .unwrap_or(16.67);
            self.hud_frame_last = Some(now);
            let hud_in =
                crate::diagnostics::DebugHudInput::from_winit(window.as_ref(), &self.input);
            self.runtime.set_debug_hud_frame_data(hud_in, ms);
        }

        Some(hmd_projection_ended)
    }

    /// Phase: VR mirror vs desktop world render, then OpenXR `end_frame_empty` when the HMD path did not submit.
    ///
    /// Call only after [`Self::render_views`] returned [`Some`], so a [`GpuContext`] is guaranteed.
    fn present_and_diagnostics(
        &mut self,
        window: &Arc<Window>,
        xr_tick: Option<OpenxrFrameTick>,
        hmd_projection_ended: bool,
    ) {
        tick_phase_trace("present_and_diagnostics");
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        // VR: desktop shows a blit of the left HMD eye (`VrMirrorBlitResources`); no second world pass.
        // Debug HUD overlay is not drawn on this path (see `frame_graph::compiled` for non-VR HUD).
        if self.runtime.vr_active() {
            if hmd_projection_ended {
                if let Err(e) = frame_loop::present_vr_mirror_blit(
                    gpu,
                    window.as_ref(),
                    &mut self.vr_mirror_blit,
                ) {
                    logger::debug!("VR mirror blit failed: {e:?}");
                    if let Err(pe) = present_clear_frame(gpu, window.as_ref()) {
                        logger::warn!("present_clear_frame after mirror blit: {pe:?}");
                    }
                }
            } else if let Err(e) = present_clear_frame(gpu, window.as_ref()) {
                logger::debug!("VR mirror clear (no HMD frame): {e:?}");
            }
        } else if let Err(e) =
            frame_loop::execute_mirror_frame_graph(&mut self.runtime, gpu, window.as_ref())
        {
            Self::handle_frame_graph_error(gpu, window.as_ref(), e);
        }

        if let (Some(handles), Some(tick)) = (self.xr_handles.as_mut(), xr_tick) {
            if !hmd_projection_ended {
                if let Err(e) = handles
                    .xr_session
                    .end_frame_empty(tick.predicted_display_time)
                {
                    logger::debug!("OpenXR end_frame_empty: {e:?}");
                }
            }
        }
    }

    /// Ends GPU frame timing and refreshes debug HUD snapshots; pairs with [`Self::frame_tick_prologue`].
    fn frame_tick_epilogue(&mut self, frame_start: Instant) {
        tick_phase_trace("frame_tick_epilogue");
        self.end_frame_timing_and_hud_capture();
        self.record_frame_tick_end(frame_start);
    }

    /// One winit redraw; phase order is documented on this module ([`crate::app::renderide_app`]).
    fn tick_frame(&mut self, event_loop: &ActiveEventLoop) {
        let frame_start = Instant::now();
        self.frame_tick_prologue(frame_start);
        self.poll_ipc_and_window();
        let xr_tick = self.xr_begin_tick();
        self.lock_step_exchange();

        if self.runtime.shutdown_requested() {
            logger::info!("Renderer shutdown requested by host");
            self.exit_code = Some(0);
            event_loop.exit();
            self.frame_tick_epilogue(frame_start);
            return;
        }

        if self.runtime.fatal_error() {
            logger::error!("Renderer fatal IPC error");
            self.exit_code = Some(4);
            event_loop.exit();
            self.frame_tick_epilogue(frame_start);
            return;
        }

        let Some(window) = self.window.clone() else {
            self.frame_tick_epilogue(frame_start);
            return;
        };

        let Some(hmd_projection_ended) = self.render_views(&window, xr_tick.as_ref()) else {
            self.frame_tick_epilogue(frame_start);
            return;
        };

        self.present_and_diagnostics(&window, xr_tick, hmd_projection_ended);

        self.frame_tick_epilogue(frame_start);
    }

    /// Finalizes [`GpuContext`] frame timing and refreshes debug HUD snapshots for the tick.
    fn end_frame_timing_and_hud_capture(&mut self) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.end_frame_timing();
            self.runtime.capture_debug_hud_after_frame_end(gpu);
        }
    }

    fn handle_frame_graph_error(gpu: &mut GpuContext, window: &Window, e: GraphExecuteError) {
        match e {
            GraphExecuteError::NoFrameGraph => {
                if let Err(pe) = present_clear_frame(gpu, window) {
                    logger::warn!("present fallback failed: {pe:?}");
                    reconfigure_gpu_for_window(gpu, window);
                }
            }
            _ => {
                logger::warn!("frame graph failed: {e:?}");
                reconfigure_gpu_for_window(gpu, window);
            }
        }
    }
}

impl ApplicationHandler for RenderideApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.listen_device_events(DeviceEvents::Always);
        self.ensure_window_gpu(event_loop);
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        apply_device_event(&mut self.input, &event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if window.id() != window_id {
            return;
        }

        apply_window_event(&mut self.input, window, &event);

        match event {
            WindowEvent::CloseRequested => {
                logger::info!("Window close requested");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    reconfigure_gpu_for_physical_size(gpu, size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(w) = self.window.as_ref() {
                    self.input.sync_window_resolution_logical(w.as_ref());
                }
                self.tick_frame(event_loop);
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(gpu) = self.gpu.as_mut() {
                    reconfigure_gpu_for_window(gpu, window.as_ref());
                }
            }
            _ => {}
        }

        self.maybe_flush_logs();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            if self.exit_code.is_none() && !self.runtime.vr_active() {
                let cap = match self.runtime.settings().read() {
                    Ok(s) => {
                        if self.input.window_focused {
                            s.display.focused_fps_cap
                        } else {
                            s.display.unfocused_fps_cap
                        }
                    }
                    Err(_) => 0,
                };
                let now = Instant::now();
                if let Some(deadline) =
                    frame_pacing::next_redraw_wait_until(self.last_frame_end, cap, now)
                {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
                    self.maybe_flush_logs();
                    return;
                }
            }
            window.request_redraw();
        }
        if self.exit_code.is_none() {
            event_loop.set_control_flow(ControlFlow::Poll);
        }
        self.maybe_flush_logs();
    }
}
