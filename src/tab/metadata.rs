use num_format::ToFormattedString;
use wgpu_3dgs_viewer as gs;

use crate::{app, util};

use super::Tab;

/// The metadata tab.
#[derive(Debug)]
pub struct Metadata;

impl Tab for Metadata {
    fn create(_state: &mut app::State) -> Self
    where
        Self: Sized,
    {
        Self
    }

    fn title(&mut self, _frame: &mut eframe::Frame, _state: &mut app::State) -> egui::WidgetText {
        "Metadata".into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame, state: &mut app::State) {
        let (file_name, count, compressions, ui_builder) = match &state.gs {
            app::Loadable::Loaded(gs) => (
                gs.selected_model().file_name.as_str(),
                gs.selected_model().gaussians.gaussians.capacity(),
                &gs.compressions,
                egui::UiBuilder::new(),
            ),
            app::Loadable::Unloaded { .. } => (
                "N/A",
                0,
                &app::Compressions::default(),
                egui::UiBuilder::new().disabled(),
            ),
        };

        ui.spacing_mut().item_spacing = egui::vec2(ui.spacing().item_spacing.x, 12.0);

        ui.scope_builder(ui_builder, |ui| {
            egui::Grid::new("metadata_grid").show(ui, |ui| {
                ui.label("Model File Name");
                ui.label(file_name);
                ui.end_row();

                ui.label("Gaussian Count");
                ui.label(count.to_formatted_string(&num_format::Locale::en));
                ui.end_row();

                ui.label("Original Size");
                ui.label(util::human_readable_size(
                    count * std::mem::size_of::<gs::PlyGaussianPod>(),
                ));
                ui.end_row();

                ui.label("Compressed Size");
                ui.label(util::human_readable_size(
                    compressions.compressed_size(count),
                ));
                ui.end_row();

                ui.label("SH Compression")
                    .on_hover_text("Spherical harmonics compression");
                ui.label(compressions.sh.to_string());
                ui.end_row();

                ui.label("Covariance 3D Compression");
                ui.label(compressions.cov3d.to_string());
                ui.end_row();
            });
        });
    }
}
