use crate::app;

use super::Tab;

/// The camera tab.
#[derive(Debug)]
pub struct Camera;

impl Tab for Camera {
    fn create(_state: &mut app::State) -> Self
    where
        Self: Sized,
    {
        Self
    }

    fn title(&mut self, _frame: &mut eframe::Frame, _state: &mut app::State) -> egui::WidgetText {
        "Camera".into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame, state: &mut app::State) {
        let (camera, loaded) = match &mut state.gs {
            app::Loadable::Loaded(gs) => (&mut gs.camera, true),
            app::Loadable::None { .. } => (&mut app::Camera::default(), false),
        };

        ui.spacing_mut().item_spacing = egui::vec2(ui.spacing().item_spacing.x, 12.0);

        egui::Grid::new("camera_grid").show(ui, |ui| {
            ui.label("Position");
            ui.horizontal(|ui| {
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

                value!(ui, loaded, camera.camera.pos.x);
                value!(ui, loaded, camera.camera.pos.y);
                value!(ui, loaded, camera.camera.pos.z);
            });
            ui.end_row();

            ui.label("Horizontal Speed");
            ui.add_enabled(
                loaded,
                egui::Slider::new(&mut camera.speed.x, 0.0..=10.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            );
            ui.end_row();

            ui.label("Vertical Speed");
            ui.add_enabled(
                loaded,
                egui::Slider::new(&mut camera.speed.y, 0.0..=10.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            );
            ui.end_row();

            ui.label("Mouse Sensitivity");
            ui.add_enabled(
                loaded,
                egui::Slider::new(&mut camera.sensitivity, 0.0..=1.0)
                    .step_by(0.01)
                    .fixed_decimals(2),
            );
            ui.end_row();
        });
    }
}
