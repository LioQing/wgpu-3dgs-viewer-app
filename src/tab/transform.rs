use wgpu_3dgs_viewer as gs;

use crate::app;

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
        let (model, gaussian, loaded) = match &mut state.gs {
            app::Loadable::Loaded(gs) => {
                (&mut gs.model_transform, &mut gs.gaussian_transform, true)
            }
            app::Loadable::None { .. } => (
                &mut app::GaussianSplattingModelTransform::new(),
                &mut app::GaussianSplattingGaussianTransform::new(),
                false,
            ),
        };

        ui.spacing_mut().item_spacing = egui::vec2(ui.spacing().item_spacing.x, 12.0);

        ui.label(egui::RichText::new("Model").strong());
        self.model(ui, loaded, model);

        ui.separator();

        ui.label(egui::RichText::new("Gaussian").strong());
        self.gaussian(ui, loaded, gaussian);
    }
}

impl Transform {
    /// Create the UI for model transform.
    fn model(
        &mut self,
        ui: &mut egui::Ui,
        loaded: bool,
        transform: &mut app::GaussianSplattingModelTransform,
    ) {
        egui::Grid::new("model_transform_grid").show(ui, |ui| {
            macro_rules! value {
                ($ui: expr, $loaded: expr, $value: expr) => {
                    $ui.add_enabled(
                        $loaded,
                        egui::DragValue::new(&mut $value)
                            .speed(0.01)
                            .fixed_decimals(4),
                    );
                };
            }

            ui.label("Position");
            ui.horizontal(|ui| {
                value!(ui, loaded, transform.pos.x);
                value!(ui, loaded, transform.pos.y);
                value!(ui, loaded, transform.pos.z);
            });
            ui.end_row();

            ui.label("Rotation");
            ui.horizontal(|ui| {
                value!(ui, loaded, transform.rot.x);
                value!(ui, loaded, transform.rot.y);
                value!(ui, loaded, transform.rot.z);
            });
            ui.end_row();

            ui.label("Scale");
            ui.horizontal(|ui| {
                value!(ui, loaded, transform.scale.x);
                value!(ui, loaded, transform.scale.y);
                value!(ui, loaded, transform.scale.z);
            });
            ui.end_row();
        });
    }

    /// Create the UI for the Gaussian transform.
    fn gaussian(
        &mut self,
        ui: &mut egui::Ui,
        loaded: bool,
        transform: &mut app::GaussianSplattingGaussianTransform,
    ) {
        egui::Grid::new("gaussian_transform_grid").show(ui, |ui| {
            ui.spacing_mut().slider_width = 100.0;

            ui.label("Size");
            ui.add_enabled(
                loaded,
                egui::Slider::new(&mut transform.size, 0.0..=2.0).fixed_decimals(2),
            );
            ui.end_row();

            ui.label("Display Mode");
            ui.horizontal(|ui| {
                macro_rules! value {
                    ($ui: expr, $loaded: expr, $value: expr, $label: ident) => {
                        if $ui
                            .add_enabled(
                                $loaded,
                                egui::SelectableLabel::new(
                                    $value == gs::GaussianDisplayMode::$label,
                                    stringify!($label),
                                ),
                            )
                            .clicked()
                        {
                            $value = gs::GaussianDisplayMode::$label;
                        }
                    };
                }

                value!(ui, loaded, transform.display_mode, Splat);
                value!(ui, loaded, transform.display_mode, Ellipse);
                value!(ui, loaded, transform.display_mode, Point);
            });
        });
    }
}
