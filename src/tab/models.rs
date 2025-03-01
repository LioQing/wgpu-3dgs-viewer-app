use std::{collections::HashMap, io::Cursor, sync::mpsc};

use itertools::Itertools;
use wgpu_3dgs_viewer as gs;

use super::Tab;

use crate::{app, util};

/// The models tab.
#[derive(Debug)]
pub struct Models;

impl Tab for Models {
    fn create(_state: &mut app::State) -> Self
    where
        Self: Sized,
    {
        Self
    }

    fn title(
        &mut self,
        _frame: &mut eframe::Frame,
        _state: &mut crate::app::State,
    ) -> egui::WidgetText {
        "Models".into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame, state: &mut crate::app::State) {
        let (models, selected_model_key, scene_tx, ui_builder) = match &mut state.gs {
            app::Loadable::Loaded(gs) => (
                &mut gs.models,
                &mut gs.selected_model_key,
                &gs.scene_tx,
                egui::UiBuilder::new(),
            ),
            app::Loadable::Unloaded { .. } => (
                &mut HashMap::new(),
                &mut "".to_string(),
                &mpsc::channel().0,
                egui::UiBuilder::new().disabled(),
            ),
        };

        ui.scope_builder(ui_builder, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("Model Count: {}", models.len()));

                ui.separator();

                if ui.button("➕ Add model").clicked() {
                    let tx = scene_tx.clone();
                    let ctx = ui.ctx().clone();
                    let task = rfd::AsyncFileDialog::new()
                        .set_title("Open a PLY file")
                        .pick_file();

                    util::exec_task(async move {
                        if let Some(file) = task.await {
                            let mut reader = Cursor::new(file.read().await);
                            let models = gs::Gaussians::read_ply(&mut reader)
                                .map(|ply| app::GaussianSplattingModel::new(file.file_name(), ply))
                                .map_err(|e| e.to_string());

                            tx.send(app::SceneCommand::AddModel(models))
                                .expect("send gs");
                            ctx.request_repaint();
                        }
                    });
                }
            });

            let text_height = egui::TextStyle::Body
                .resolve(ui.style())
                .size
                .max(ui.spacing().interact_size.y);

            let available_height = ui.available_height();

            let mut models_ordered = models
                .iter_mut()
                .sorted_by_key(|(k, _)| (*k).clone())
                .collect::<Vec<_>>();

            egui_extras::TableBuilder::new(ui)
                .striped(true)
                .resizable(true)
                .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                .column(egui_extras::Column::auto())
                .column(egui_extras::Column::auto())
                .column(egui_extras::Column::auto())
                .column(egui_extras::Column::auto())
                .min_scrolled_height(0.0)
                .max_scroll_height(available_height)
                .sense(egui::Sense::click())
                .header(20.0, |mut header| {
                    header.col(|_| {});
                    header.col(|ui| {
                        ui.strong("File Name");
                    });
                    header.col(|ui| {
                        ui.strong("Visible");
                    });
                    header.col(|ui| {
                        ui.strong("Remove");
                    });
                })
                .body(|body| {
                    body.rows(text_height, models_ordered.len(), |mut row| {
                        let index = row.index();
                        let (key, model) = &mut models_ordered[index];

                        row.set_selected(*key == selected_model_key);

                        row.col(|ui| {
                            ui.add(egui::Label::new((index + 1).to_string()).selectable(false));
                        });
                        row.col(|ui| {
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Label::new(match model.file_name.as_str() {
                                        "" => "[Unnamed]",
                                        s => s,
                                    })
                                    .selectable(false),
                                );
                            });
                        });
                        row.col(|ui| {
                            ui.checkbox(&mut model.visible, "");
                        });

                        row.col(|ui| {
                            if ui.button("🗑").clicked() {
                                scene_tx
                                    .send(app::SceneCommand::RemoveModel(key.clone()))
                                    .unwrap();
                            }
                        });

                        if row.response().clicked() {
                            *selected_model_key = (*key).clone();
                        }
                    })
                });
        });
    }
}
