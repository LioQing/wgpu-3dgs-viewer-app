use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Cursor},
    sync::mpsc,
};

use itertools::Itertools;

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
                    let task = rfd::AsyncFileDialog::new()
                        .set_title("Open a PLY file")
                        .pick_file();

                    util::exec_task(async move {
                        if let Some(file) = task.await {
                            let file_name = file.file_name();
                            let reader = Box::new(Cursor::new(file.read().await));
                            tx.send(app::SceneCommand::AddModel { file_name, reader })
                                .expect("send gs");
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

            let hovered = ui.ctx().input(|input| !input.raw.hovered_files.is_empty());

            let dropped_file = ui
                .ctx()
                .input(|input| match &input.raw.dropped_files.as_slice() {
                    [file, ..] => Some(match cfg!(target_arch = "wasm32") {
                        true => Ok((
                            file.name.clone(),
                            Box::new(Cursor::new(
                                file.bytes.as_ref().expect("file bytes").clone(),
                            )) as Box<dyn BufRead + Send + 'static>,
                        )),
                        false => File::open(file.path.as_ref().expect("file path")).map(|f| {
                            (file.name.clone(), {
                                Box::new(BufReader::new(f)) as Box<dyn BufRead + Send + 'static>
                            })
                        }),
                    }),
                    _ => None,
                });

            match dropped_file {
                Some(Ok((file_name, reader))) => {
                    scene_tx
                        .send(app::SceneCommand::AddModel { file_name, reader })
                        .expect("send gs");
                    ui.ctx().request_repaint();
                }
                Some(Err(err)) => {
                    log::error!("Error adding model: {err}");
                }
                None => {}
            }

            let row_count = models_ordered.len() + if hovered { 1 } else { 0 };

            egui_extras::TableBuilder::new(ui)
                .striped(true)
                .resizable(true)
                .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                .columns(egui_extras::Column::auto(), 4)
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
                    body.rows(text_height, row_count, |mut row| {
                        let index = row.index();

                        if index == models_ordered.len() {
                            row.col(|_| {});
                            row.col(|ui| {
                                ui.label("Release to Add Model");
                            });
                            row.col(|_| {});
                            row.col(|_| {});
                            return;
                        }

                        let (key, model) = &mut models_ordered[index];

                        row.set_selected(*key == selected_model_key);

                        row.col(|ui| {
                            ui.add(egui::Label::new((index + 1).to_string()).selectable(false));
                        });
                        row.col(|ui| {
                            ui.horizontal(|ui| {
                                ui.add(egui::Label::new(&model.file_name).selectable(false));
                            });
                        });
                        row.col(|ui| match model.visible {
                            true => {
                                if ui.button("👁").clicked() {
                                    model.visible = false;
                                }
                            }
                            false => {
                                if ui.button("―").clicked() {
                                    model.visible = true;
                                }
                            }
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
