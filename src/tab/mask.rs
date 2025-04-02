use std::sync::mpsc;

use glam::*;
use wgpu_3dgs_viewer as gs;

use super::Tab;

use crate::app;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShapeChanged {
    Unchanged,
    Removed,
    Updated,
}

/// The mask tab.
#[derive(Debug)]
pub struct Mask {
    /// Op code error.
    op_code_error: Option<String>,
}

impl Tab for Mask {
    fn create(_state: &mut app::State) -> Self
    where
        Self: Sized,
    {
        Self {
            op_code_error: None,
        }
    }

    fn title(
        &mut self,
        _frame: &mut eframe::Frame,
        _state: &mut crate::app::State,
    ) -> egui::WidgetText {
        "Mask".into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame, state: &mut crate::app::State) {
        let (mask, scene_tx, ui_builder) = match &mut state.gs {
            app::Loadable::Loaded(gs) => (
                &mut gs
                    .models
                    .get_mut(&gs.selected_model_key)
                    .expect("selected model")
                    .mask,
                &gs.scene_tx,
                egui::UiBuilder::new(),
            ),
            app::Loadable::Unloaded { .. } => (
                &mut app::GaussianSplattingMask::new(),
                &mpsc::channel().0,
                egui::UiBuilder::new().disabled(),
            ),
        };

        ui.spacing_mut().item_spacing = egui::vec2(ui.spacing().item_spacing.x, 12.0);

        ui.scope_builder(ui_builder, |ui| {
            ui.label(egui::RichText::new("Shapes").strong());
            self.shapes(ui, mask, scene_tx);

            ui.separator();

            ui.label(egui::RichText::new("Operations").strong())
                .on_hover_text(
                    "The operations are applied as set operations on the shapes, \
                     in the following precedence:\n\
                     • `!0` - Complement of shape 0\n\
                     • `0 ^ 1` - Symmetric difference of shape 0 and 1\n\
                     • `0 - 1` - Difference of shape 0 and 1\n\
                     • `0 & 1` - Intersection of shape 0 and 1\n\
                     • `0 | 1` - Union of shape 0 and 1",
                );
            self.operations(ui, mask, scene_tx);
        });
    }
}

impl Mask {
    /// Create the UI for shapes.
    fn shapes(
        &mut self,
        ui: &mut egui::Ui,
        mask: &mut app::GaussianSplattingMask,
        scene_tx: &mpsc::Sender<app::SceneCommand>,
    ) {
        let mut updated = false;
        let mut removed = Vec::new();
        for (index, shape) in mask.shapes.iter_mut().enumerate() {
            match self.shape(ui, index, shape) {
                ShapeChanged::Removed => {
                    removed.push(index);
                    updated = true;
                }
                ShapeChanged::Updated => {
                    updated = true;
                }
                _ => {}
            }
        }

        for index in removed.into_iter().rev() {
            mask.shapes.remove(index);
        }

        if ui.button("➕ Add Shape").clicked() {
            mask.shapes.push(app::GaussianSplattingMaskShape::new());

            updated = true;
        }

        if updated {
            self.update_mask(mask, scene_tx);
        }
    }

    /// Create the UI for a shape.
    fn shape(
        &mut self,
        ui: &mut egui::Ui,
        index: usize,
        shape: &mut app::GaussianSplattingMaskShape,
    ) -> ShapeChanged {
        egui::CollapsingHeader::new(format!("{index}. {:?}", shape.shape.kind))
            .id_salt(format!("mask_{index}"))
            .show(ui, |ui| {
                egui::Grid::new(format!("mask_{index}_grid"))
                    .show(ui, |ui| {
                        let mut changed = ShapeChanged::Unchanged;

                        ui.label("Shape Type");
                        ui.horizontal(|ui| {
                            macro_rules! value {
                                ($ui:expr, $value:expr, $label:ident) => {
                                    if $ui
                                        .selectable_label(
                                            shape.shape.kind == gs::MaskShapeKind::$label,
                                            stringify!($label),
                                        )
                                        .clicked()
                                    {
                                        changed = ShapeChanged::Updated;
                                        shape.shape.kind = gs::MaskShapeKind::$label;
                                    }
                                };
                            }

                            value!(ui, shape.shape.kind, Box);
                            value!(ui, shape.shape.kind, Ellipsoid);
                        });
                        ui.end_row();

                        ui.label("Color");
                        ui.horizontal(|ui| {
                            let mut ui_builder = egui::UiBuilder::new();
                            if !shape.visible {
                                ui_builder = ui_builder.disabled();
                            }

                            ui.scope_builder(ui_builder, |ui| {
                                let color_u8 = shape.shape.color.map(|c| c * 255.0).as_u8vec4();
                                let mut color_32 = egui::Color32::from_rgba_premultiplied(
                                    color_u8.x, color_u8.y, color_u8.z, color_u8.w,
                                );
                                if ui.color_edit_button_srgba(&mut color_32).changed() {
                                    changed = ShapeChanged::Updated;
                                    shape.shape.color =
                                        U8Vec4::from_array(color_32.to_array()).as_vec4() / 255.0;
                                }
                            });
                            if ui.checkbox(&mut shape.visible, "Visible").changed() {
                                changed = ShapeChanged::Updated;
                            }
                        });
                        ui.end_row();

                        macro_rules! value {
                            ($ui:expr, $axis:expr, $value:expr) => {
                                $ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x /= 2.0;

                                    ui.label($axis);
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut $value)
                                                .speed(0.01)
                                                .fixed_decimals(4),
                                        )
                                        .changed()
                                    {
                                        changed = ShapeChanged::Updated;
                                    }
                                });
                            };
                        }

                        ui.label("Position");
                        ui.horizontal(|ui| {
                            value!(ui, "X", shape.shape.pos.x);
                            value!(ui, "Y", shape.shape.pos.y);
                            value!(ui, "Z", shape.shape.pos.z);
                        });
                        ui.end_row();

                        ui.label("Rotation");
                        ui.horizontal(|ui| {
                            value!(ui, "X", shape.rot.x);
                            value!(ui, "Y", shape.rot.y);
                            value!(ui, "Z", shape.rot.z);

                            if changed == ShapeChanged::Updated {
                                shape.shape.rotation = Quat::from_euler(
                                    EulerRot::ZYX,
                                    shape.rot.z.to_radians(),
                                    shape.rot.x.to_radians(),
                                    shape.rot.y.to_radians(),
                                );
                            }
                        });
                        ui.end_row();

                        ui.label("Scale");
                        ui.horizontal(|ui| {
                            value!(ui, "X", shape.shape.scale.x);
                            value!(ui, "Y", shape.shape.scale.y);
                            value!(ui, "Z", shape.shape.scale.z);
                        });
                        ui.end_row();

                        if ui.button("🗑 Remove").clicked() {
                            changed = ShapeChanged::Removed;
                        }
                        ui.end_row();

                        changed
                    })
                    .inner
            })
            .body_returned
            .unwrap_or(ShapeChanged::Unchanged)
    }

    /// Create the UI for operations.
    fn operations(
        &mut self,
        ui: &mut egui::Ui,
        mask: &mut app::GaussianSplattingMask,
        scene_tx: &mpsc::Sender<app::SceneCommand>,
    ) {
        if ui
            .add(
                egui::TextEdit::multiline(&mut mask.op_code)
                    .font(egui::TextStyle::Monospace)
                    .code_editor()
                    .desired_rows(5)
                    .layouter(&mut |ui: &egui::Ui, string: &str, wrap_width: f32| {
                        let mut layout_job = egui_extras::syntax_highlighting::highlight(
                            ui.ctx(),
                            ui.style(),
                            &egui_extras::syntax_highlighting::CodeTheme::from_memory(
                                ui.ctx(),
                                ui.style(),
                            ),
                            string,
                            "rs",
                        );
                        layout_job.wrap.max_width = wrap_width;
                        ui.fonts(|f| f.layout_job(layout_job))
                    }),
            )
            .changed()
        {
            self.update_mask(mask, scene_tx);
        }

        if let Some(e) = &self.op_code_error {
            ui.label(egui::RichText::new(e).color(egui::Color32::RED));
        }
    }

    /// Update the mask.
    fn update_mask(
        &mut self,
        mask: &mut app::GaussianSplattingMask,
        scene_tx: &mpsc::Sender<app::SceneCommand>,
    ) {
        mask.update_pods();

        match app::GaussianSplattingMaskOp::parse(&mask.op_code) {
            Ok(Some(op)) => {
                self.op_code_error = None;
                match op.validate_shapes(mask.shapes.len()) {
                    Ok(_) => {
                        scene_tx
                            .send(app::SceneCommand::EvaluateMask(Some(op)))
                            .expect("send op");
                    }
                    Err(e) => {
                        self.op_code_error = Some(format!("Invalid shape index: {e}"));
                    }
                }
            }
            Ok(None) => {
                self.op_code_error = None;
                scene_tx
                    .send(app::SceneCommand::EvaluateMask(None))
                    .expect("send op");
            }
            Err(e) => {
                self.op_code_error = Some(e.to_string());
            }
        }
    }
}
