use wgpu_3dgs_viewer as gs;

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
                        ($ui: expr, $action: expr, $value: expr, $label: ident, $display: expr) => {
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
                ui.checkbox(&mut selection.immediate, "");
                ui.end_row();

                if selection.method == app::SelectionMethod::Brush {
                    ui.label("Brush Radius");
                    ui.add(egui::Slider::new(&mut selection.brush_radius, 1..=100).integer());
                    ui.end_row();
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
                        *action = Some(app::Action::Selection);
                    }
                }
            }
        });
    }
}
