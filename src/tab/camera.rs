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
        let (camera, ui_builder) = match &mut state.gs {
            app::Loadable::Loaded(gs) => (&mut gs.camera, egui::UiBuilder::new()),
            app::Loadable::None { .. } => (
                &mut app::Camera::default(),
                egui::UiBuilder::new().disabled(),
            ),
        };

        ui.spacing_mut().item_spacing = egui::vec2(ui.spacing().item_spacing.x, 12.0);

        ui.scope_builder(ui_builder, |ui| {
            egui::Grid::new("camera_grid").show(ui, |ui| {
                ui.label("Position");
                ui.horizontal(|ui| {
                    macro_rules! value {
                        ($ui: expr, $value: expr) => {
                            $ui.add(
                                egui::DragValue::new(&mut $value)
                                    .speed(0.01)
                                    .fixed_decimals(4),
                            );
                        };
                    }

                    value!(ui, camera.camera.pos.x);
                    value!(ui, camera.camera.pos.y);
                    value!(ui, camera.camera.pos.z);
                });
                ui.end_row();

                ui.label("Horizontal Speed");
                ui.add(
                    egui::Slider::new(&mut camera.speed.x, 0.0..=10.0)
                        .step_by(0.01)
                        .fixed_decimals(2),
                );
                ui.end_row();

                ui.label("Vertical Speed");
                ui.add(
                    egui::Slider::new(&mut camera.speed.y, 0.0..=10.0)
                        .step_by(0.01)
                        .fixed_decimals(2),
                );
                ui.end_row();

                ui.label("Mouse Sensitivity");
                ui.add(
                    egui::Slider::new(&mut camera.sensitivity, 0.0..=1.0)
                        .step_by(0.01)
                        .fixed_decimals(2),
                );
                ui.end_row();
            });
        });
    }
}
