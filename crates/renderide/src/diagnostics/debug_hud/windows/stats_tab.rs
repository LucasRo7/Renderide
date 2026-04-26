//! Main **Stats** overlay: frame index, GPU adapter, host allocator, IPC, draw stats, resources.

use crate::diagnostics::{FrameDiagnosticsSnapshot, RendererInfoSnapshot};

use super::super::fmt as hud_fmt;
use super::labels::device_type_label;

fn main_debug_panel_frame_line(
    ui: &imgui::Ui,
    renderer: Option<&RendererInfoSnapshot>,
    frame: Option<&FrameDiagnosticsSnapshot>,
) {
    if let Some(r) = renderer {
        ui.text(format!(
            "Frame index {}  |  viewport {}×{}",
            r.last_frame_index, r.viewport_px.0, r.viewport_px.1
        ));
    } else if frame.is_some() {
        ui.text_disabled("Frame index / viewport: (need renderer snapshot)");
    }
}

/// Adapter name, limits, and surface info (Stats tab).
fn main_debug_panel_gpu_adapter(ui: &imgui::Ui, r: &RendererInfoSnapshot) {
    ui.separator();
    ui.text("GPU (adapter)");
    ui.text_wrapped(format!("Name: {}", r.adapter_name));
    ui.text(format!(
        "Class: {}  |  backend: {:?}",
        device_type_label(r.adapter_device_type),
        r.adapter_backend
    ));
    ui.text_wrapped(format!(
        "Driver: {} ({})",
        r.adapter_driver, r.adapter_driver_info
    ));
    ui.text(format!(
        "Surface: {:?}  |  present: {:?}",
        r.surface_format, r.present_mode
    ));
    ui.text(format!(
        "MSAA: requested {}×  |  effective {}×  |  max {}×",
        r.msaa_requested_samples, r.msaa_effective_samples, r.msaa_max_samples
    ));
    ui.text(format!(
        "MSAA (VR stereo): effective {}×  |  max {}×",
        r.msaa_effective_samples_stereo, r.msaa_max_samples_stereo
    ));
    ui.text(format!(
        "Limits: tex2d≤{}  max_buf={}  storage_bind={}  |  base_instance={}  multiview={}",
        r.gpu_max_texture_dim_2d,
        r.gpu_max_buffer_size,
        r.gpu_max_storage_binding,
        r.gpu_supports_base_instance,
        r.gpu_supports_multiview
    ));
}

/// Allocator and host CPU / RAM lines (Stats tab).
fn main_debug_panel_host_and_allocator(ui: &imgui::Ui, f: &FrameDiagnosticsSnapshot) {
    ui.separator();
    ui.text("Process GPU memory (wgpu allocator)");
    match (
        f.gpu_allocator.allocated_bytes,
        f.gpu_allocator.reserved_bytes,
    ) {
        (Some(alloc), Some(resv)) => ui.text(format!(
            "{} / {} GiB allocated / reserved",
            hud_fmt::gib_value(7, 2, alloc),
            hud_fmt::gib_value(7, 2, resv)
        )),
        _ => ui.text("not reported for this backend"),
    }

    ui.separator();
    ui.text("CPU / RAM (host)");
    if f.host.cpu_model.is_empty() {
        ui.text("CPU model: (unknown)");
    } else {
        ui.text_wrapped(format!("CPU model: {}", f.host.cpu_model));
    }
    ui.text(format!(
        "Logical CPUs: {:>3}  |  usage {}%",
        f.host.logical_cpus,
        hud_fmt::f64_field(6, 2, f64::from(f.host.cpu_usage_percent))
    ));
    let ram_pct = if f.host.ram_total_bytes > 0 {
        100.0 * f.host.ram_used_bytes as f64 / f.host.ram_total_bytes as f64
    } else {
        0.0
    };
    ui.text(format!(
        "RAM: {} / {} GiB  ({}%)",
        hud_fmt::gib_value(7, 2, f.host.ram_used_bytes),
        hud_fmt::gib_value(7, 2, f.host.ram_total_bytes),
        hud_fmt::f64_field(5, 1, ram_pct)
    ));
}

/// IPC connection, init state, and coarse scene counts (Stats tab).
fn main_debug_panel_ipc_and_scene(ui: &imgui::Ui, r: &RendererInfoSnapshot) {
    ui.separator();
    ui.text("IPC / init");
    ui.text(format!(
        "Connected: {}  |  init: {:?}",
        r.ipc_connected, r.init_state
    ));

    ui.separator();
    ui.text("Scene");
    ui.text(format!("Render spaces: {}", r.render_space_count));
    ui.text(format!(
        "Mesh renderables (CPU tables): {}",
        r.mesh_renderable_count
    ));
}

/// Draw batches, instance batches, and cull stats (Stats tab).
fn main_debug_panel_draw_stats(ui: &imgui::Ui, f: &FrameDiagnosticsSnapshot) {
    ui.separator();
    ui.text("Batches");
    let m = &f.mesh_draw;
    ui.text(format!(
        "{:>5} total  |  {:>5} main  |  {:>5} overlay",
        m.batch_total, m.batch_main, m.batch_overlay
    ));
    ui.text("Draws");
    ui.text(format!(
        "{:>5} total  |  {:>5} main  |  {:>5} overlay",
        m.draws_total, m.draws_main, m.draws_overlay
    ));
    ui.text(format!(
        "GPU instance batches (indexed submits): {:>5}  ({:>5} intersect)",
        m.instance_batch_total, m.intersect_pass_batches
    ));
    let compression = if m.instance_batch_total > 0 {
        m.gpu_instances_emitted as f32 / m.instance_batch_total as f32
    } else {
        0.0
    };
    ui.text(format!(
        "GPU instances emitted: {:>5}  |  avg instances/batch: {:>5.2}",
        m.gpu_instances_emitted, compression
    ));
    ui.text(format!(
        "Pipeline pass submits: {:>5}",
        m.submitted_pipeline_pass_total
    ));
    ui.text(format!(
        "Frustum cull: {:>5} considered  |  {:>5} culled  |  Hi-Z {:>5} culled  |  {:>5} submitted after cull",
        m.draws_pre_cull, m.draws_culled, m.draws_hi_z_culled, m.draws_total
    ));
    ui.text(format!(
        "Prep rigid {:>5}  skinned {:>5}",
        m.rigid_draws, m.skinned_draws
    ));
    ui.text(format!(
        "Last submit render_tasks: {}  |  pending camera readbacks: not implemented",
        f.last_submit_render_task_count
    ));
    ui.text(format!(
        "IPC outbound drops this tick: primary={} background={}  |  consecutive fail streak: primary={} background={}",
        f.ipc_primary_outbound_drop_this_tick,
        f.ipc_background_outbound_drop_this_tick,
        f.ipc_primary_consecutive_fail_streak,
        f.ipc_background_consecutive_fail_streak
    ));
    ui.text(format!(
        "Frame submit apply failures: {}  |  OpenXR wait_frame errs: {}  locate_views errs: {}  |  unhandled IPC cmds (total events): {}",
        f.frame_submit_apply_failures,
        f.xr_wait_frame_failures,
        f.xr_locate_views_failures,
        f.unhandled_ipc_command_event_total
    ));
}

/// Pool counts, materials, and render graph summary (Stats tab).
fn main_debug_panel_resources_and_graph(
    ui: &imgui::Ui,
    renderer: Option<&RendererInfoSnapshot>,
    frame: Option<&FrameDiagnosticsSnapshot>,
) {
    match (renderer, frame) {
        (Some(r), Some(f)) => {
            ui.separator();
            ui.text("Resources");
            ui.text(format!("Mesh pool: {}", f.mesh_pool_entry_count));
            ui.text(format!("Textures (pool): {}", r.resident_texture_count));
            ui.text(format!(
                "Render textures (pool): {}",
                f.render_textures_gpu_resident
            ));

            ui.separator();
            ui.text("Materials (property store)");
            ui.text(format!(
                "Material property maps: {}  |  property blocks: {}  |  shader bindings: {}",
                r.material_property_slots, r.property_block_slots, r.material_shader_bindings
            ));

            ui.separator();
            ui.text("Frame graph");
            ui.text(format!(
                "Render graph passes: {}  (compile DAG waves: {})  |  GPU lights (packed): {}",
                r.frame_graph_pass_count, r.frame_graph_topo_levels, r.gpu_light_count
            ));
        }
        (Some(r), None) => {
            ui.separator();
            ui.text("Resources");
            ui.text(format!("Mesh pool: {}", r.resident_mesh_count));
            ui.text(format!("Textures (pool): {}", r.resident_texture_count));
            ui.text(format!(
                "Render textures (pool): {}",
                r.resident_render_texture_count
            ));

            ui.separator();
            ui.text("Materials (property store)");
            ui.text(format!(
                "Material property maps: {}  |  property blocks: {}  |  shader bindings: {}",
                r.material_property_slots, r.property_block_slots, r.material_shader_bindings
            ));

            ui.separator();
            ui.text("Frame graph");
            ui.text(format!(
                "Render graph passes: {}  (compile DAG waves: {})  |  GPU lights (packed): {}",
                r.frame_graph_pass_count, r.frame_graph_topo_levels, r.gpu_light_count
            ));
        }
        (None, Some(f)) => {
            ui.separator();
            ui.text("Resources");
            ui.text(format!("Mesh pool: {}", f.mesh_pool_entry_count));
            ui.text(format!("Textures (pool): {}", f.textures_gpu_resident));
            ui.text(format!(
                "Render textures (pool): {}",
                f.render_textures_gpu_resident
            ));
        }
        (None, None) => {}
    }
}

/// Unified IPC, adapter, scene, draws, and resources (FPS / submit timing: **Frame timing** window).
pub(super) fn main_debug_panel(
    ui: &imgui::Ui,
    renderer: Option<&RendererInfoSnapshot>,
    frame: Option<&FrameDiagnosticsSnapshot>,
) {
    if renderer.is_none() && frame.is_none() {
        ui.text("Waiting for snapshot…");
        return;
    }

    main_debug_panel_frame_line(ui, renderer, frame);

    if let Some(r) = renderer {
        main_debug_panel_gpu_adapter(ui, r);
    }

    if let Some(f) = frame {
        main_debug_panel_host_and_allocator(ui, f);
    }

    if let Some(r) = renderer {
        main_debug_panel_ipc_and_scene(ui, r);
    }

    if let Some(f) = frame {
        main_debug_panel_draw_stats(ui, f);
    }

    main_debug_panel_resources_and_graph(ui, renderer, frame);
}
