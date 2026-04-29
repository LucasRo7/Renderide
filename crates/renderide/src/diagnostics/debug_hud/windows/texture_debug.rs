//! Texture pool debug window with current-view filtering.

use crate::diagnostics::TextureDebugSnapshot;
use imgui::{Condition, ListClipper};

use super::super::fmt as hud_fmt;
use super::super::layout as overlay_layout;
use super::table_helpers::scrolling_table_flags;

/// Texture pool window with current-view filtering.
pub(super) fn texture_debug_window(
    ui: &imgui::Ui,
    snapshot: &TextureDebugSnapshot,
    open: &mut bool,
    current_view_only: &mut bool,
) {
    ui.window("Textures")
        .opened(open)
        .position(
            [overlay_layout::MARGIN, overlay_layout::MARGIN + 360.0],
            Condition::FirstUseEver,
        )
        .size([860.0, 420.0], Condition::FirstUseEver)
        .bg_alpha(0.85)
        .build(|| {
            ui.checkbox("Only current view", current_view_only);
            ui.text(format!(
                "{} textures  |  {} current-view  |  {} total",
                snapshot.rows.len(),
                snapshot.current_view_texture_count,
                hud_fmt::bytes_compact(snapshot.total_resident_bytes)
            ));
            let rows: Vec<_> = snapshot
                .rows
                .iter()
                .filter(|row| !*current_view_only || row.used_by_current_view)
                .collect();
            if let Some(_table) = ui.begin_table_with_sizing(
                "texture_debug_rows",
                8,
                scrolling_table_flags(),
                [0.0, 330.0],
                0.0,
            ) {
                ui.table_setup_column("Asset");
                ui.table_setup_column("Size");
                ui.table_setup_column("Mips");
                ui.table_setup_column("Bytes");
                ui.table_setup_column("Host");
                ui.table_setup_column("GPU");
                ui.table_setup_column("Sampler");
                ui.table_setup_column("View");
                ui.table_headers_row();
                let clip = ListClipper::new(rows.len() as i32);
                let tok = clip.begin(ui);
                for row_i in tok.iter() {
                    let row = rows[row_i as usize];
                    ui.table_next_row();
                    ui.table_next_column();
                    ui.text(format!("{}", row.asset_id));
                    ui.table_next_column();
                    ui.text(format!("{}x{}", row.width, row.height));
                    ui.table_next_column();
                    ui.text(format!(
                        "{}/{}",
                        row.mip_levels_resident, row.mip_levels_total
                    ));
                    ui.table_next_column();
                    ui.text(hud_fmt::bytes_compact(row.resident_bytes));
                    ui.table_next_column();
                    ui.text(format!("{:?} {:?}", row.host_format, row.color_profile));
                    ui.table_next_column();
                    ui.text(format!("{:?}", row.wgpu_format));
                    ui.table_next_column();
                    ui.text(format!(
                        "{:?} aniso={} wrap={:?}/{:?} bias={:.2}",
                        row.filter_mode, row.aniso_level, row.wrap_u, row.wrap_v, row.mipmap_bias
                    ));
                    ui.table_next_column();
                    ui.text(if row.used_by_current_view {
                        "current"
                    } else {
                        ""
                    });
                }
            }
        });
}
