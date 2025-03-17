use glam::*;
use wgpu_3dgs_viewer as gs;

use crate::{app, util};

use super::Tab;

/// The selection tab.
#[derive(Debug)]
pub struct Selection;

impl Tab for Selection {
    fn create(_state: &mut app::State) -> Self
    where
        Self: Sized,
    {
        Self
    }

    fn title(&mut self, _frame: &mut eframe::Frame, _state: &mut app::State) -> egui::WidgetText {
        "Selection".into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame, state: &mut app::State) {
        let (selection, action, ui_builder) = match &mut state.gs {
            app::Loadable::Loaded(gs) => {
                (&mut gs.selection, &mut gs.action, egui::UiBuilder::new())
            }
            app::Loadable::Unloaded { .. } => (
                &mut app::Selection::new(),
                &mut None,
                egui::UiBuilder::new().disabled(),
            ),
        };

        ui.spacing_mut().item_spacing = egui::vec2(ui.spacing().item_spacing.x, 12.0);

        ui.scope_builder(ui_builder, |ui| {
            egui::Grid::new("selection_grid").show(ui, |ui| {
                ui.label("Selection Method");
                ui.horizontal(|ui| {
                    macro_rules! value {
                        ($ui:expr, $action:expr, $value:expr, $label:ident, $display:expr) => {
                            if $ui
                                .selectable_label($value == app::SelectionMethod::$label, $display)
                                .clicked()
                            {
                                $value = app::SelectionMethod::$label;
                            }
                        };
                    }

                    value!(ui, action, selection.method, Rect, "Rectangle");
                    value!(ui, action, selection.method, Brush, "Brush");
                });
                ui.end_row();

                ui.label("Operation");
                ui.horizontal(|ui| {
                    macro_rules! value {
                        ($ui:expr, $action:expr, $value:expr, $label:ident, $display:expr) => {
                            if $ui
                                .selectable_label($value == gs::QuerySelectionOp::$label, $display)
                                .clicked()
                            {
                                $value = gs::QuerySelectionOp::$label;
                            }
                        };
                    }

                    value!(ui, action, selection.operation, Set, "Set");
                    value!(ui, action, selection.operation, Add, "Add");
                    value!(ui, action, selection.operation, Remove, "Remove");
                });
                ui.end_row();

                ui.label("Immediate Mode")
                    .on_hover_text("The selection is immediately applied while dragging");
                ui.add(util::toggle(&mut selection.immediate));
                ui.end_row();

                ui.label("Highlight Color");
                ui.color_edit_button_srgba(&mut selection.highlight_color);
                ui.end_row();

                if selection.method == app::SelectionMethod::Brush {
                    ui.label("Brush Radius");
                    ui.add(egui::Slider::new(&mut selection.brush_radius, 1..=100).integer());
                    ui.end_row();
                }

                ui.label("Show Unedited")
                    .on_hover_text("Show the model without any edits");
                ui.add(util::toggle(&mut selection.show_unedited));
                if selection.show_unedited {
                    selection.edit = None;
                }
                ui.end_row();
            });

            ui.horizontal(|ui| {
                match action {
                    Some(app::Action::Selection { .. }) => {
                        if ui.button("Selecting...").clicked() {
                            *action = None;
                            selection.edit = None;
                        }
                    }
                    _ => {
                        if ui.button("Select").clicked() {
                            *action = Some(app::Action::Selection);
                        }
                    }
                }

                match selection.edit {
                    Some(..) => {
                        if ui.button("Editing...").clicked() {
                            selection.edit = None;
                        }
                    }
                    _ => {
                        if ui.button("Edit").clicked() {
                            selection.edit = Some(app::SelectionEdit::default());
                        }
                    }
                }
            });

            if let Some(edit) = &mut selection.edit {
                selection.show_unedited = false;

                if ui.button("Reset Parameters").clicked() {
                    *edit = app::SelectionEdit::default();
                }

                egui::Grid::new("selection_edit_grid").show(ui, |ui| {
                    ui.label("Hidden");
                    ui.checkbox(&mut edit.hidden, "");
                    ui.end_row();

                    ui.label("Color");
                    ui.horizontal(|ui| {
                        macro_rules! value {
                            ($ui:expr, $value:expr, $label:ident, $display:expr, $val:expr) => {
                                if $ui
                                    .selectable_label(
                                        matches!($value, app::SelectionColorEdit::$label(..)),
                                        $display,
                                    )
                                    .clicked()
                                {
                                    $value = app::SelectionColorEdit::$label($val);
                                }
                            };
                        }

                        value!(ui, edit.color, Hsv, "HSV", vec3(0.0, 1.0, 1.0));
                        value!(
                            ui,
                            edit.color,
                            OverrideColor,
                            "Override Color",
                            vec3(1.0, 1.0, 1.0)
                        );
                    });
                    ui.end_row();

                    match &mut edit.color {
                        app::SelectionColorEdit::Hsv(hsv) => {
                            ui.label("Hue");
                            ui.add(egui::Slider::new(&mut hsv.x, 0.0..=1.0).fixed_decimals(2));
                            ui.end_row();

                            ui.label("Saturation");
                            ui.add(egui::Slider::new(&mut hsv.y, 0.0..=2.0).fixed_decimals(2));
                            ui.end_row();

                            ui.label("Brightness");
                            ui.add(egui::Slider::new(&mut hsv.z, 0.0..=2.0).fixed_decimals(2));
                            ui.end_row();
                        }
                        app::SelectionColorEdit::OverrideColor(rgb) => {
                            ui.label("RGB Color");
                            ui.color_edit_button_rgb(bytemuck::cast_mut(rgb));
                            ui.end_row();
                        }
                    }

                    ui.label("Opacity");
                    ui.add(egui::Slider::new(&mut edit.alpha, 0.0..=2.0).fixed_decimals(2));
                    ui.end_row();

                    ui.label("Contrast");
                    ui.add(egui::Slider::new(&mut edit.contrast, -1.0..=1.0).fixed_decimals(2));
                    ui.end_row();

                    ui.label("Exposure");
                    ui.add(egui::Slider::new(&mut edit.exposure, -5.0..=5.0).fixed_decimals(2));
                    ui.end_row();

                    ui.label("Gamma");
                    ui.add(egui::Slider::new(&mut edit.gamma, 0.0..=5.0).fixed_decimals(2));
                    ui.end_row();
                });
            }
        });
    }
}
