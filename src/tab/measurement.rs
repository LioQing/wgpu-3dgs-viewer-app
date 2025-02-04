use std::sync::mpsc;

use crate::app;

use super::Tab;

/// The measurement tab.
#[derive(Debug)]
pub struct Measurement;

impl Tab for Measurement {
    fn create(_state: &mut app::State) -> Self
    where
        Self: Sized,
    {
        Self
    }

    fn title(&mut self, _frame: &mut eframe::Frame, _state: &mut app::State) -> egui::WidgetText {
        "Measurement".into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame, state: &mut app::State) {
        let (measurement, ui_builder) = match &mut state.gs {
            app::Loadable::Loaded(gs) => (&mut gs.measurement, egui::UiBuilder::new()),
            app::Loadable::Unloaded { .. } => (
                &mut app::Measurement::new(),
                egui::UiBuilder::new().disabled(),
            ),
        };

        ui.spacing_mut().item_spacing = egui::vec2(ui.spacing().item_spacing.x, 12.0);

        ui.scope_builder(ui_builder, |ui| {
            egui::Grid::new("measurement_method_grid").show(ui, |ui| {
                ui.label("Hit Method");
                ui.horizontal(|ui| {
                    macro_rules! value {
                        ($ui: expr, $value: expr, $label: ident, $display: expr) => {
                            if $ui
                                .selectable_label(
                                    $value == app::MeasurementHitMethod::$label,
                                    $display,
                                )
                                .clicked()
                            {
                                $value = app::MeasurementHitMethod::$label;
                            }
                        };
                    }

                    value!(ui, measurement.hit_method, MostAlpha, "Most Alpha");
                    value!(ui, measurement.hit_method, Closest, "Closest");
                });
            });

            ui.separator();

            let mut removed = Vec::new();
            for (index, hit_pair) in measurement.hit_pairs.iter_mut().enumerate() {
                if !self.measurement(ui, index, &mut measurement.action, hit_pair) {
                    removed.push(index);
                }
            }

            for index in removed.into_iter().rev() {
                measurement.hit_pairs.remove(index);
            }

            if ui.button("➕ Add Measurement").clicked() {
                measurement
                    .hit_pairs
                    .push(app::MeasurementHitPair::new(format!(
                        "Measurement {}",
                        measurement.hit_pairs.len()
                    )));
            }
        });
    }
}

impl Measurement {
    /// Create the UI for the measurement.
    ///
    /// Returns whether the measurement is kept alive, i.e. not removed.
    fn measurement(
        &mut self,
        ui: &mut egui::Ui,
        index: usize,
        action: &mut Option<app::MeasurementAction>,
        hit_pair: &mut app::MeasurementHitPair,
    ) -> bool {
        egui::CollapsingHeader::new(format!("{index}. {}", hit_pair.label))
            .id_salt(format!("measurement_{index}"))
            .show(ui, |ui| {
                egui::Grid::new(format!("measurement_{index}_grid"))
                    .show(ui, |ui| {
                        let mut alive = true;

                        macro_rules! value {
                            ($ui: expr, $value: expr) => {
                                $ui.add(
                                    egui::DragValue::new(&mut $value)
                                        .speed(0.01)
                                        .fixed_decimals(4),
                                );
                            };
                        }

                        ui.label("Label");
                        ui.add_sized(
                            egui::vec2(150.0, ui.spacing().interact_size.y),
                            egui::TextEdit::singleline(&mut hit_pair.label),
                        );
                        if hit_pair.label.is_empty() {
                            hit_pair.label = format!("Measurement {index}");
                        }
                        ui.end_row();

                        ui.label("Color");
                        ui.horizontal(|ui| {
                            let mut ui_builder = egui::UiBuilder::new();
                            if !hit_pair.visible {
                                ui_builder = ui_builder.disabled();
                            }

                            ui.scope_builder(ui_builder, |ui| {
                                ui.color_edit_button_srgba(&mut hit_pair.color)
                            });
                            ui.checkbox(&mut hit_pair.visible, "Visible");
                        });
                        ui.end_row();

                        ui.label("Line Width");
                        ui.add(
                            egui::Slider::new(&mut hit_pair.line_width, 0.0..=5.0)
                                .step_by(0.01)
                                .fixed_decimals(2),
                        );
                        ui.end_row();

                        for i in 0..2 {
                            ui.label(format!("Position {}", i + 1));
                            ui.horizontal(|ui| {
                                value!(ui, hit_pair.hits[i].pos.x);
                                value!(ui, hit_pair.hits[i].pos.y);
                                value!(ui, hit_pair.hits[i].pos.z);

                                match action {
                                    Some(app::MeasurementAction::LocateHit {
                                        hit_pair_index,
                                        hit_index,
                                        ..
                                    }) if *hit_pair_index == index && *hit_index == i => {
                                        if ui.button("Locating...").clicked() {
                                            *action = None;
                                        }
                                    }
                                    None | Some(app::MeasurementAction::LocateHit { .. }) => {
                                        if ui.button("Locate").clicked() {
                                            let (tx, rx) = mpsc::channel();
                                            *action = Some(app::MeasurementAction::LocateHit {
                                                hit_pair_index: index,
                                                hit_index: i,
                                                tx,
                                                rx,
                                            });
                                        }
                                    }
                                }
                            });
                            ui.end_row();
                        }

                        ui.label("Distance");
                        ui.label(format!("{:.4}", hit_pair.distance()));
                        ui.end_row();

                        if ui.button("🗑 Remove").clicked() {
                            alive = false;
                        }
                        ui.end_row();

                        alive
                    })
                    .inner
            })
            .body_returned
            .unwrap_or(true)
    }
}
