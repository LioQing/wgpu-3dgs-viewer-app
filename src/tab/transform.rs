use wgpu_3dgs_viewer as gs;

use crate::{app, util};

use super::Tab;

/// The transform editor tab.
#[derive(Debug)]
pub struct Transform;

impl Tab for Transform {
    fn create(_state: &mut app::State) -> Self
    where
        Self: Sized,
    {
        Self
    }

    fn title(&mut self, _frame: &mut eframe::Frame, _state: &mut app::State) -> egui::WidgetText {
        "Transform".into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame, state: &mut app::State) {
        let (model, gaussian, ui_builder) = match &mut state.gs {
            app::Loadable::Loaded(gs) => (
                &mut gs.model_transform,
                &mut gs.gaussian_transform,
                egui::UiBuilder::new(),
            ),
            app::Loadable::Unloaded { .. } => (
                &mut app::GaussianSplattingModelTransform::new(),
                &mut app::GaussianSplattingGaussianTransform::new(),
                egui::UiBuilder::new().disabled(),
            ),
        };

        ui.spacing_mut().item_spacing = egui::vec2(ui.spacing().item_spacing.x, 12.0);

        ui.scope_builder(ui_builder, |ui| {
            ui.label(egui::RichText::new("Model").strong());
            self.model(ui, model);

            ui.separator();

            ui.label(egui::RichText::new("Gaussian").strong());
            self.gaussian(ui, gaussian);
        });
    }
}

impl Transform {
    /// Create the UI for model transform.
    fn model(&mut self, ui: &mut egui::Ui, transform: &mut app::GaussianSplattingModelTransform) {
        egui::Grid::new("model_transform_grid").show(ui, |ui| {
            macro_rules! value {
                ($ui:expr, $axis:expr, $value:expr) => {
                    $ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x /= 2.0;

                        ui.label($axis);
                        ui.add(
                            egui::DragValue::new(&mut $value)
                                .speed(0.01)
                                .fixed_decimals(4),
                        );
                    });
                };
            }

            ui.label("Position");
            ui.horizontal(|ui| {
                value!(ui, "X", transform.pos.x);
                value!(ui, "Y", transform.pos.y);
                value!(ui, "Z", transform.pos.z);
            });
            ui.end_row();

            ui.label("Rotation");
            ui.horizontal(|ui| {
                value!(ui, "X", transform.rot.x);
                value!(ui, "Y", transform.rot.y);
                value!(ui, "Z", transform.rot.z);
            });
            ui.end_row();

            ui.label("Scale");
            ui.horizontal(|ui| {
                value!(ui, "X", transform.scale.x);
                value!(ui, "Y", transform.scale.y);
                value!(ui, "Z", transform.scale.z);
            });
            ui.end_row();
        });
    }

    /// Create the UI for the Gaussian transform.
    fn gaussian(
        &mut self,
        ui: &mut egui::Ui,
        transform: &mut app::GaussianSplattingGaussianTransform,
    ) {
        egui::Grid::new("gaussian_transform_grid").show(ui, |ui| {
            ui.spacing_mut().slider_width = 100.0;

            ui.label("Size");
            ui.add(egui::Slider::new(&mut transform.size, 0.0..=2.0).fixed_decimals(2));
            ui.end_row();

            ui.label("Display Mode");
            ui.horizontal(|ui| {
                macro_rules! value {
                    ($ui: expr, $value: expr, $label: ident) => {
                        if $ui
                            .selectable_label(
                                $value == gs::GaussianDisplayMode::$label,
                                stringify!($label),
                            )
                            .clicked()
                        {
                            $value = gs::GaussianDisplayMode::$label;
                        }
                    };
                }

                value!(ui, transform.display_mode, Splat);
                value!(ui, transform.display_mode, Ellipse);
                value!(ui, transform.display_mode, Point);
            });
            ui.end_row();

            ui.label("SH Degree")
                .on_hover_text("Degree of spherical harmonics");
            let mut deg = transform.sh_deg.degree();
            ui.add(egui::Slider::new(&mut deg, 0..=3));
            transform.sh_deg = gs::GaussianShDegree::new(deg).expect("SH degree");
            ui.end_row();

            ui.label("No SH0")
                .on_hover_text("Exclude the 0th degree of spherical harmonics");
            ui.add(util::toggle(&mut transform.no_sh0));
            ui.end_row();
        });
    }
}
