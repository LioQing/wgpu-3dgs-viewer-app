use crate::app;

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
                        ($ui: expr, $action: expr, $value: expr, $label: ident, $display: expr) => {
                            if $ui
                                .selectable_label($value == app::SelectionMethod::$label, $display)
                                .clicked()
                            {
                                $value = app::SelectionMethod::$label;

                                if let Some(app::Action::Selection { method, .. }) = action {
                                    *method = $value;
                                }
                            }
                        };
                    }

                    value!(ui, action, selection.method, Rect, "Rectangle");
                    value!(ui, action, selection.method, Brush, "Brush");
                });
                ui.end_row();

                ui.label("Immediate Mode");
                if ui.checkbox(&mut selection.immediate, "").clicked() {
                    if let Some(app::Action::Selection { immediate, .. }) = action {
                        *immediate = selection.immediate;
                    }
                }
                ui.end_row();

                if selection.method == app::SelectionMethod::Brush {
                    ui.label("Brush Radius");
                    ui.add(egui::Slider::new(&mut selection.brush_radius, 1..=100).integer());
                    ui.end_row();

                    if let Some(app::Action::Selection { brush_radius, .. }) = action {
                        *brush_radius = selection.brush_radius;
                    }
                }
            });

            match action {
                Some(app::Action::Selection { .. }) => {
                    if ui.button("Selecting...").clicked() {
                        *action = None;
                    }
                }
                _ => {
                    if ui.button("Select").clicked() {
                        *action = Some(app::Action::Selection {
                            method: selection.method,
                            immediate: selection.immediate,
                            brush_radius: selection.brush_radius,
                        });
                    }
                }
            }
        });
    }
}
