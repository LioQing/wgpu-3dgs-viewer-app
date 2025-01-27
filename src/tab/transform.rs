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
        let (transform, loaded) = match &mut state.gs {
            app::Loadable::Loaded(gs) => (&mut gs.transform, true),
            app::Loadable::None { .. } => (&mut app::GaussianSplattingTransform::new(), false),
        };

        egui::Grid::new("transform_grid")
            .spacing(egui::Vec2::new(
                egui::Spacing::default().item_spacing.x,
                12.0,
            ))
            .show(ui, |ui| {
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
                    value!(ui, loaded, transform.pos.y);
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
}
