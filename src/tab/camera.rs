use crate::app;

use super::Tab;

/// The camera tab.
#[derive(Debug)]
pub struct Camera {
    /// The saved orbit arm length.
    saved_orbit_arm_length: f32,
}

impl Tab for Camera {
    fn create(_state: &mut app::State) -> Self
    where
        Self: Sized,
    {
        Self {
            saved_orbit_arm_length: 1.0,
        }
    }

    fn title(&mut self, _frame: &mut eframe::Frame, _state: &mut app::State) -> egui::WidgetText {
        "Camera".into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame, state: &mut app::State) {
        let (camera, ui_builder) = match &mut state.gs {
            app::Loadable::Loaded(gs) => (&mut gs.camera, egui::UiBuilder::new()),
            app::Loadable::Unloaded { .. } => (
                &mut app::Camera::default(),
                egui::UiBuilder::new().disabled(),
            ),
        };

        ui.spacing_mut().item_spacing = egui::vec2(ui.spacing().item_spacing.x, 12.0);

        ui.scope_builder(ui_builder, |ui| {
            egui::Grid::new("camera_control_mode_grid").show(ui, |ui| {
                ui.label("Control Mode");
                ui.horizontal(|ui| {
                    #[derive(Debug, Clone, Copy, PartialEq)]
                    enum Mode {
                        FirstPerson,
                        Orbit,
                    }

                    macro_rules! value {
                        ($ui:expr, $value:expr, $label:ident, $display:expr, $tooltip:expr) => {
                            if $ui
                                .selectable_label($value == Mode::$label, $display)
                                .on_hover_text($tooltip)
                                .clicked()
                            {
                                $value = Mode::$label;
                            }
                        };
                    }

                    let mode = match camera.control {
                        app::CameraControl::FirstPerson(..) => Mode::FirstPerson,
                        app::CameraControl::Orbit(..) => Mode::Orbit,
                    };
                    let mut new_mode = mode;

                    value!(
                        ui,
                        new_mode,
                        Orbit,
                        "Orbit",
                        "• Hold left mouse button to rotate around the target\n\
                         • Hold right mouse button to pan\n\
                         • Hold middle mouse button to look around\n\
                         • Scroll to zoom in/out"
                    );
                    value!(
                        ui,
                        new_mode,
                        FirstPerson,
                        "First Person",
                        format!(
                            "• Click on the viewer to focus, press Esc to unfocus\n\
                             • WASD to move, Space to go up, Shift to go down\n\
                             • Mouse to look around{}",
                            if cfg!(target_arch = "wasm32") {
                                "\n• In some browsers, focusing immediately after unfocusing may \
                                not work"
                            } else {
                                ""
                            }
                        )
                    );

                    if new_mode != mode {
                        camera.control = match new_mode {
                            Mode::FirstPerson => {
                                if let app::CameraControl::Orbit(orbit) = &camera.control {
                                    self.saved_orbit_arm_length =
                                        (orbit.target - orbit.pos).length();
                                }

                                app::CameraControl::FirstPerson(camera.control.to_first_person())
                            }
                            Mode::Orbit => app::CameraControl::Orbit(
                                camera.control.to_orbit(self.saved_orbit_arm_length),
                            ),
                        };
                    }
                });
                ui.end_row();
            });

            ui.separator();

            egui::Grid::new("camera_configs_grid").show(ui, |ui| {
                if let app::CameraControl::Orbit(orbit) = &mut camera.control {
                    ui.label("Orbit Target")
                        .on_hover_text("The point which the camera orbits around");
                    ui.horizontal(|ui| {
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

                        value!(ui, "X", orbit.target.x);
                        value!(ui, "Y", orbit.target.y);
                        value!(ui, "Z", orbit.target.z);
                    });
                    ui.end_row();
                }

                ui.label("Position");
                ui.horizontal(|ui| {
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

                    value!(ui, "X", camera.control.pos_mut().x);
                    value!(ui, "Y", camera.control.pos_mut().y);
                    value!(ui, "Z", camera.control.pos_mut().z);
                });
                ui.end_row();

                ui.label("Movement Speed");
                ui.add(
                    egui::Slider::new(&mut camera.speed, 0.0..=10.0)
                        .step_by(0.01)
                        .fixed_decimals(2),
                );
                ui.end_row();

                ui.label("Rotation Sensitivity");
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
