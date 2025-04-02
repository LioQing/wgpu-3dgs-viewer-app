use std::{
    collections::HashMap,
    io::Cursor,
    marker::PhantomData,
    sync::{Arc, Mutex, mpsc},
};

#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::JsCast;

use eframe::{egui_wgpu, wgpu};
use glam::*;
use itertools::Itertools;
use num_format::ToFormattedString;
use strum::IntoEnumIterator;
use wgpu_3dgs_viewer::{self as gs, QueryVariant, Texture};

use crate::{app, renderer, util};

use super::Tab;

/// The macro to apply the same function to [`SceneResource`] regardless of the compression.
macro_rules! apply_to_scene_resource {
    ($frame:ident, $compressions:expr, |$res:ident| $func:block) => {
        macro_rules! case {
            ($sh:ident, $cov3d:ident) => {
                crate::app::Compressions {
                    sh: crate::app::ShCompression::$sh,
                    cov3d: crate::app::Cov3dCompression::$cov3d,
                }
            };
        }

        macro_rules! apply {
            ($sh:ident, $cov3d:ident) => {
                paste::paste! {
                    let mut lock = $frame
                        .wgpu_render_state()
                        .expect("render state")
                        .renderer
                        .write();
                    let $res = lock
                        .callback_resources
                        .get_mut::<crate::tab::scene::SceneResource::<
                            wgpu_3dgs_viewer::[< GaussianPodWithSh $sh Cov3d $cov3d Configs >]
                        >>()
                        .expect("scene resource");

                    $func
                }
            };
        }

        match &$compressions {
            case!(Single, Single) => {
                apply!(Single, Single);
            }
            case!(Single, Half) => {
                apply!(Single, Half);
            }
            case!(Half, Single) => {
                apply!(Half, Single);
            }
            case!(Half, Half) => {
                apply!(Half, Half);
            }
            case!(Norm8, Single) => {
                apply!(Norm8, Single);
            }
            case!(Norm8, Half) => {
                apply!(Norm8, Half);
            }
            case!(Remove, Single) => {
                apply!(None, Single);
            }
            case!(Remove, Half) => {
                apply!(None, Half);
            }
        }
    };
}

/// The scene tab.
#[derive(Debug)]
pub struct Scene {
    /// The input state.
    input: SceneInput,

    /// The FPS update interval.
    fps_interval: f32,

    /// The previous FPS.
    fps: f32,

    /// Is scene initialized, i.e. viewer is created.
    initialized: bool,

    /// The current query.
    query: Query,

    /// The pending query result.
    query_result: Option<QueryResult>,
}

impl Tab for Scene {
    fn create(_state: &mut app::State) -> Self
    where
        Self: Sized,
    {
        Self {
            input: SceneInput::new(),
            fps_interval: 0.0,
            fps: 0.0,
            initialized: false,
            query: Query::none(),
            query_result: None,
        }
    }

    fn title(&mut self, _frame: &mut eframe::Frame, _state: &mut app::State) -> egui::WidgetText {
        "Scene".into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame, state: &mut app::State) {
        let updated_gs = match &mut state.gs {
            app::Loadable::Unloaded(unloaded) => match unloaded.rx.try_recv() {
                Ok(Ok(gs)) => {
                    log::debug!("Gaussian splatting loaded");

                    self.initialized = false;
                    self.empty(ui, unloaded, &state.compressions);

                    Some(app::Loadable::loaded(gs))
                }
                Ok(Err(err)) => {
                    log::debug!("Error loading Gaussian splatting: {err}");

                    self.empty(ui, unloaded, &state.compressions);

                    Some(app::Loadable::error(err))
                }
                _ => {
                    self.empty(ui, unloaded, &state.compressions);

                    None
                }
            },
            app::Loadable::Loaded(gs) => match self.initialized {
                false => match self.initialize(ui, frame, gs, &mut state.compressions) {
                    Ok(Some(true)) => {
                        self.initialized = true;
                        None
                    }
                    Ok(Some(false)) => Some(app::Loadable::unloaded()),
                    Ok(None) => None,
                    Err(e) => Some(app::Loadable::error(e)),
                },
                true => match self.loaded(ui, frame, gs) {
                    true => None,
                    false => Some(app::Loadable::unloaded()),
                },
            },
        };

        if let Some(gs) = updated_gs {
            state.gs = gs;
        }
    }
}

impl Scene {
    /// Create an empty scene tab.
    fn empty(
        &mut self,
        ui: &mut egui::Ui,
        unloaded: &mut app::Unloaded<app::GaussianSplatting, String>,
        compressions: &app::Compressions,
    ) {
        ui.vertical_centered(|ui| {
            ui.add_space(ui.spacing().item_spacing.y);

            ui.label("Drag & Drop");
            ui.label("OR");
            if ui.button("Browse File").clicked() {
                let tx = unloaded.tx.clone();
                let ctx = ui.ctx().clone();
                let task = rfd::AsyncFileDialog::new()
                    .set_title("Open a PLY file")
                    .pick_file();
                let compressions = compressions.clone();

                util::exec_task(async move {
                    if let Some(file) = task.await {
                        let filename = match file.file_name().trim().is_empty() {
                            true => "Unnamed".to_string(),
                            false => file.file_name().trim().to_string(),
                        };
                        let reader = Cursor::new(file.read().await);
                        let gs = app::GaussianSplatting::new(filename, reader, compressions)
                            .map_err(|e| e.to_string());

                        tx.send(gs).expect("send gs");
                        ctx.request_repaint();
                    }
                });
            }

            ui.label("");
            ui.label("to Open a PLY Model File 📦");

            if ui.ctx().input(|input| !input.raw.hovered_files.is_empty()) {
                ui.label("");
                ui.label("Release to Load");
            } else if let Some(err) = &unloaded.err {
                ui.label("");
                ui.label(egui::RichText::new(format!("Error: {err}")).color(egui::Color32::RED));
            }

            match ui
                .ctx()
                .input(|input| match &input.raw.dropped_files.as_slice() {
                    [_x, _xs, ..] => Some(Err("only one file is allowed")),
                    [file] => Some(Ok(match cfg!(target_arch = "wasm32") {
                        true => app::GaussianSplatting::new(
                            match file.name.trim().is_empty() {
                                true => "Unnamed".to_string(),
                                false => file.name.trim().to_string(),
                            },
                            Cursor::new(file.bytes.as_ref().expect("file bytes").clone()),
                            compressions.clone(),
                        )
                        .map_err(|e| e.to_string()),
                        false => std::fs::read(file.path.as_ref().expect("file path").clone())
                            .map_err(gs::Error::Io)
                            .map_err(|e| e.to_string())
                            .and_then(|data| {
                                app::GaussianSplatting::new(
                                    match file.name.trim().is_empty() {
                                        true => "Unnamed".to_string(),
                                        false => file.name.trim().to_string(),
                                    },
                                    Cursor::new(data),
                                    compressions.clone(),
                                )
                                .map_err(|e| e.to_string())
                            }),
                    })),
                    _ => None,
                }) {
                Some(Ok(result)) => {
                    unloaded.tx.send(result).expect("send gs");
                    ui.ctx().request_repaint();
                }
                Some(Err(err)) => {
                    unloaded.err = Some(err.to_string());
                }
                None => {}
            }
        });
    }

    /// Create a loaded scene tab.
    ///
    /// Returns whether the scene is still loaded, false indicates the scene should be unloaded now.
    fn loaded(
        &mut self,
        ui: &mut egui::Ui,
        frame: &mut eframe::Frame,
        gs: &mut app::GaussianSplatting,
    ) -> bool {
        let mut loaded = true;

        // Loading bar
        if let Some((loading, ..)) = &gs.model_loader {
            ui.horizontal(|ui| {
                ui.add(egui::Spinner::default());

                ui.label(format!("Loading: {loading}"));

                ui.separator();

                let model = gs.models.get(loading).expect("model");
                let progress = model.gaussians.gaussians.len() as f32
                    / model.gaussians.gaussians.capacity() as f32;

                ui.add(egui::ProgressBar::new(progress).text(format!(
                        "{} / {}",
                        model
                            .gaussians
                            .gaussians
                            .len()
                            .to_formatted_string(&num_format::Locale::en),
                        model
                            .gaussians
                            .gaussians
                            .capacity()
                            .to_formatted_string(&num_format::Locale::en),
                    )));
            });

            ui.separator();
        }

        // UI
        ui.horizontal(|ui| {
            let loaded_label = ui.label(format!(
                "📦 Loaded: {}",
                if gs.models.len() > 1 {
                    format!("{} models", gs.models.len())
                } else if gs.selected_model().file_name.len() > 20 {
                    format!("{}...", &gs.selected_model().file_name[..20])
                } else {
                    gs.selected_model().file_name.clone()
                }
            ));

            if gs.models.len() == 1 && gs.selected_model().file_name.len() > 20 {
                loaded_label.on_hover_text(&gs.selected_model().file_name);
            }

            ui.separator();

            loaded &= !ui.button("🗑 Close models").clicked();

            ui.separator();

            let dt = ui.ctx().input(|input| input.unstable_dt);
            self.fps_interval += dt;
            if self.fps_interval >= 1.0 {
                self.fps_interval -= 1.0;
                self.fps = 1.0 / dt;
            }

            ui.label("🏃 FPS:");
            ui.add(egui::Label::new(
                egui::RichText::new(format!("{:<5.1}", self.fps)).monospace(),
            ));
        });

        // Check for loading model
        if let Some((loading, rx)) = &gs.model_loader {
            let timer = chrono::Local::now();

            let model = gs.models.get_mut(loading).expect("model");
            let mut new_count = 0;

            for gaussian in rx.try_iter() {
                match gaussian {
                    Ok(gaussian) => {
                        model.gaussians.gaussians.push(gaussian);
                        new_count += 1;
                    }
                    Err(e) => {
                        log::error!("Error loading gaussian: {e}");
                    }
                }

                const BATCH_SIZE: usize = 1000;
                const MAX_TIME: f32 = 0.06;
                if new_count % BATCH_SIZE == 0
                    && (chrono::Local::now() - timer).num_milliseconds() as f32 / 100.0 > MAX_TIME
                {
                    break;
                }
            }

            let start = model.gaussians.gaussians.len() - new_count;
            apply_to_scene_resource!(frame, gs.compressions, |res| {
                res.load_model(
                    frame.wgpu_render_state().expect("render state"),
                    &loading,
                    start,
                    &model.gaussians.gaussians[start..],
                )
            });

            if model.gaussians.gaussians.len() == model.gaussians.gaussians.capacity() {
                gs.model_loader = None;
            }
        }

        // Receive scene commands
        for command in gs.scene_rx.try_iter() {
            match command {
                app::SceneCommand::AddModel { file_name, reader } => {
                    let mut i = 0;
                    let mut new_file_name = file_name.clone();
                    while gs.models.contains_key(&new_file_name) {
                        i += 1;
                        new_file_name = format!("{} ({})", file_name, i);
                    }

                    let file_name = new_file_name;

                    if let Some((other, ..)) = &gs.model_loader {
                        log::error!("Model loader is already running for {other}");
                        continue;
                    }

                    let (count, gaussian_rx) = match app::GaussianSplattingModel::init_load(reader)
                    {
                        Ok((count, rx)) => (count, rx),
                        Err(e) => {
                            log::error!("Error loading model: {e}");
                            continue;
                        }
                    };
                    let model = app::GaussianSplattingModel::new(file_name.clone(), count);

                    gs.model_loader = Some((file_name.clone(), gaussian_rx));

                    log::debug!("Additional model loaded: {file_name}");

                    apply_to_scene_resource!(frame, gs.compressions, |res| {
                        res.add_model(
                            frame.wgpu_render_state().expect("render state"),
                            file_name.clone(),
                            count,
                        )
                    });

                    gs.models.insert(file_name, model);
                }
                app::SceneCommand::RemoveModel(key) => {
                    if gs.models.len() == 1 && gs.models.contains_key(&key) {
                        loaded = false;
                    } else {
                        log::debug!("Model removed: {key}");

                        apply_to_scene_resource!(frame, gs.compressions, |res| {
                            res.remove_model(&key)
                        });

                        gs.models.remove(&key);

                        if gs.selected_model_key == key {
                            gs.selected_model_key =
                                gs.models.keys().next().expect("first key").clone();
                        }
                    }
                }
                app::SceneCommand::UpdateMeasurementHit => {
                    apply_to_scene_resource!(frame, gs.compressions, |res| {
                        res.update_measurement_visible_hit_pairs(&gs.measurement.hit_pairs)
                    });
                }
                app::SceneCommand::EvaluateMask(op) => {
                    apply_to_scene_resource!(frame, gs.compressions, |res| {
                        res.evaluate_mask(
                            frame.wgpu_render_state().expect("render state"),
                            op.as_ref(),
                            &gs.selected_model_key,
                            gs.selected_model(),
                        );
                    });
                }
            }
        }

        // Viewport
        egui::Frame::canvas(ui.style()).show(ui, |ui| {
            macro_rules! case {
                ($sh:ident, $cov3d:ident) => {
                    app::Compressions {
                        sh: app::ShCompression::$sh,
                        cov3d: app::Cov3dCompression::$cov3d,
                    }
                };
            }

            macro_rules! apply {
                ($macro:ident, $gs:expr, $($args:expr),*) => {
                    match &$gs.compressions {
                        case!(Single, Single) => {
                            $macro!(Single, Single, $($args),*)
                        }
                        case!(Single, Half) => {
                            $macro!(Single, Half, $($args),*)
                        }
                        case!(Half, Single) => {
                            $macro!(Half, Single, $($args),*)
                        }
                        case!(Half, Half) => {
                            $macro!(Half, Half, $($args),*)
                        }
                        case!(Norm8, Single) => {
                            $macro!(Norm8, Single, $($args),*)
                        }
                        case!(Norm8, Half) => {
                            $macro!(Norm8, Half, $($args),*)
                        }
                        case!(Remove, Single) => {
                            $macro!(None, Single, $($args),*)
                        }
                        case!(Remove, Half) => {
                            $macro!(None, Half, $($args),*)
                        }
                    }
                }
            }

            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            macro_rules! postprocess {
                ($sh:ident, $cov3d:ident, $self:expr, $frame:expr, $rect:expr, $gs:expr) => {
                    paste::paste! {
                        $self.loaded_postprocess::<
                            gs::[< GaussianPodWithSh $sh Cov3d $cov3d Configs >]
                        >($frame, $rect, $gs)
                    }
                };
            }

            apply!(postprocess, gs, self, frame, &rect, gs);

            if self.query_result.is_none() {
                self.input.handle(ui, gs, &mut self.query, &rect, &response);
            }

            macro_rules! preprocess {
                ($sh:ident, $cov3d:ident, $self:expr, $frame:expr, $rect:expr, $gs:expr) => {
                    paste::paste! {
                        $self.loaded_preprocess::<
                            gs::[< GaussianPodWithSh $sh Cov3d $cov3d Configs >]
                        >($frame, $rect, $gs)
                    }
                };
            }

            apply!(preprocess, gs, self, frame, &rect, gs);

            let distances = gs
                .models
                .iter()
                .map(|(k, m)| {
                    (
                        k,
                        (m.world_center() - gs.camera.control.pos()).length_squared(),
                    )
                })
                .collect::<HashMap<_, _>>();

            macro_rules! painter {
                ($sh:ident, $cov3d:ident, $ui:expr, $rect:expr, $gs:expr) => {
                    paste::paste! {
                        $ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                            $rect,
                            SceneCallback::<gs::[< GaussianPodWithSh $sh Cov3d $cov3d Configs >]> {
                                model_render_keys: $gs.models.iter()
                                    .filter(|(_, m)| m.visible)
                                    .sorted_by(|(a, _), (b, _)| {
                                        distances.get(b).expect("distance")
                                            .partial_cmp(&distances.get(a).expect("distance"))
                                            .unwrap_or(std::cmp::Ordering::Equal)
                                    })
                                    .map(|(k, _)| k.clone())
                                    .collect(),
                                query: self.query.clone(),
                                phantom: PhantomData,
                            },
                        ))
                    }
                };
            }

            apply!(painter, gs, ui, rect, gs);
        });

        loaded
    }

    /// Run the postprocess.
    ///
    /// Because eframe does not allow any compute pass after the render pass,
    /// this is run before the preprocess pass to compute the previous frame.
    fn loaded_postprocess<G: gs::GaussianPod>(
        &mut self,
        frame: &mut eframe::Frame,
        rect: &egui::Rect,
        gs: &mut app::GaussianSplatting,
    ) {
        let egui_wgpu::RenderState {
            device,
            queue,
            renderer,
            ..
        } = frame.wgpu_render_state().expect("render state");
        let mut renderer = renderer.write();
        let SceneResource::<G> { viewer, .. } = renderer
            .callback_resources
            .get_mut()
            .expect("scene resource");
        let viewer = viewer.lock().expect("viewer");

        // Postprocess, because eframe cannot do any compute pass after the render pass.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Postprocess Encoder"),
        });

        for key in gs.models.iter().filter(|(_, m)| m.visible).map(|(k, _)| k) {
            let model = &viewer.models.get(key).expect("model");

            viewer.postprocessor.postprocess(
                &mut encoder,
                &model.bind_groups.postprocessor.0,
                &model.bind_groups.postprocessor.1,
                model.gaussian_buffers.gaussians_buffer.len() as u32,
                &model.gaussian_buffers.postprocess_indirect_args_buffer,
            );
        }

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Receive query result.
        match &mut self.query_result {
            Some(QueryResult::MeasurementLocateHit) => {
                if let Query::MeasurementLocateHit {
                    pod,
                    hit_method,
                    tx,
                } = &self.query
                {
                    let (query_result_tx, rx) = oneshot::channel();
                    self.query_result = Some(QueryResult::Downloading(rx));

                    let device = device.clone();
                    let queue = queue.clone();
                    let pod = *pod;
                    let hit_method = *hit_method;
                    let tx = tx.clone();
                    let camera = gs.camera.control.clone();
                    let viewer_size = Vec2::from_array(rect.size().into()).as_uvec2();
                    let count_buffer = viewer
                        .models
                        .get(&gs.selected_model_key)
                        .expect("model")
                        .gaussian_buffers
                        .query_result_count_buffer
                        .clone();
                    let results_buffer = viewer
                        .models
                        .get(&gs.selected_model_key)
                        .expect("model")
                        .gaussian_buffers
                        .query_results_buffer
                        .clone();

                    util::exec_blocking_task(async move {
                        let mut results =
                            gs::query::download(&device, &queue, &count_buffer, &results_buffer)
                                .await
                                .expect("download")
                                .into_iter()
                                .map(gs::QueryHitResultPod::from)
                                .collect::<Vec<_>>();

                        let pos = match hit_method {
                            app::MeasurementHitMethod::MostAlpha => {
                                gs::query::hit_pos_by_alpha_range(
                                    &pod,
                                    &mut results,
                                    &camera,
                                    viewer_size,
                                    0.05,
                                )
                                .map(|(_, _, pos)| pos)
                                .unwrap_or(Vec3::ZERO)
                            }
                            app::MeasurementHitMethod::Closest => {
                                gs::query::hit_pos_by_closest(&pod, &results, &camera, viewer_size)
                                    .map(|(_, pos)| pos)
                                    .unwrap_or(Vec3::ZERO)
                            }
                        };

                        if let Err(e) = tx.send(pos) {
                            log::error!("Error sending locate hit query result: {e}");
                        }

                        query_result_tx.send(None).expect("send");
                    });
                } else {
                    self.query_result = None;
                }
            }
            None | Some(QueryResult::Downloading(..)) => {}
        }

        if let Some(QueryResult::Downloading(rx)) = &self.query_result {
            if let Ok(query_result) = rx.try_recv() {
                self.query_result = query_result;
            }
        }
    }

    /// Run the preprocess.
    fn loaded_preprocess<G: gs::GaussianPod>(
        &mut self,
        frame: &mut eframe::Frame,
        rect: &egui::Rect,
        gs: &mut app::GaussianSplatting,
    ) {
        let egui_wgpu::RenderState {
            device,
            queue,
            renderer,
            ..
        } = frame.wgpu_render_state().expect("render state");
        let mut renderer = renderer.write();
        let SceneResource::<G> {
            viewer,
            measurement_renderer,
            measurement_visible_hit_pairs,
            query_toolset,
            query_texture_overlay,
            query_cursor,
            unedited_models,
            show_unedited_model,
            ..
        } = renderer
            .callback_resources
            .get_mut()
            .expect("scene resource");

        let mut viewer = viewer.lock().expect("viewer");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Preprocess Encoder"),
        });

        if self.query_result.is_none() {
            // Update query texture size.
            let wgpu::Extent3d { width, height, .. } =
                viewer.world_buffers.query_texture.texture().size();
            let texture_size = uvec2(width, height);

            let viewer_size = Vec2::from_array(rect.size().into()).as_uvec2();
            if texture_size != viewer_size {
                viewer.update_query_texture_size(device, viewer_size);
                query_texture_overlay
                    .update_bind_group(device, &viewer.world_buffers.query_texture);

                for (key, model) in unedited_models.iter_mut() {
                    model.update_bind_group(
                        device,
                        &viewer,
                        &viewer.models.get(key).expect("model").gaussian_buffers,
                    );
                }
            }

            // Handle new query.
            if let Query::MeasurementLocateHit { .. } = self.query {
                self.query_result = Some(QueryResult::MeasurementLocateHit);
            }

            let query_pod = match &self.query {
                Query::None { pod } => pod.as_query(),
                Query::MeasurementLocateHit { pod, .. } => pod.as_query(),
                Query::Selection {
                    action,
                    op,
                    immediate,
                    brush_radius,
                    pos,
                } => {
                    query_toolset.set_use_texture(!immediate);
                    query_toolset.update_brush_radius(*brush_radius);

                    match action {
                        Some(QuerySelectionAction::Start(tool)) => {
                            query_toolset.start(*tool, *op, *pos)
                        }
                        Some(QuerySelectionAction::End) => query_toolset.end(),
                        None => query_toolset.update_pos(*pos),
                    };

                    query_cursor.update_query_toolset(queue, query_toolset, *pos);

                    query_toolset.query()
                }
            };

            viewer.update_query(queue, query_pod);

            if let Query::Selection {
                immediate: false, ..
            } = self.query
            {
                query_toolset.render(queue, &mut encoder, &viewer.world_buffers.query_texture);
            }

            // Update the viewer.
            viewer.update_camera(queue, &gs.camera.control, viewer_size);
            viewer.update_model_transform(
                queue,
                &gs.selected_model_key,
                gs.selected_model().transform.pos,
                gs.selected_model().transform.quat(),
                gs.selected_model().transform.scale,
            );
            viewer.update_gaussian_transform(
                queue,
                gs.gaussian_transform.size,
                gs.gaussian_transform.display_mode,
                gs.gaussian_transform.sh_deg,
                gs.gaussian_transform.no_sh0,
            );

            // Selections.
            match gs.action {
                Some(app::Action::Selection) => match &gs.selection.edit {
                    Some(edit) => {
                        viewer.update_selection_edit_with_pod(queue, &edit.to_pod());
                        viewer.update_selection_highlight(queue, vec4(0.0, 0.0, 0.0, 0.0));
                        gs.selection.show_unedited = false;
                    }
                    None => {
                        viewer
                            .update_selection_edit_with_pod(queue, &gs::GaussianEditPod::default());
                        viewer.update_selection_highlight_with_pod(
                            queue,
                            &gs::SelectionHighlightPod::new(
                                U8Vec4::from_array(gs.selection.highlight_color.to_array())
                                    .as_vec4()
                                    / 255.0,
                            ),
                        );
                    }
                },
                _ => {
                    viewer.update_selection_highlight(queue, vec4(0.0, 0.0, 0.0, 0.0));
                }
            }

            if !measurement_visible_hit_pairs.is_empty() {
                measurement_renderer.update_hit_pairs(
                    device,
                    measurement_visible_hit_pairs,
                    &viewer.world_buffers.camera_buffer,
                );
            }
        }

        *show_unedited_model = gs.selection.show_unedited;
        if *show_unedited_model {
            viewer.update_selection_edit_with_pod(queue, &gs::GaussianEditPod::default());
        }

        // Preprocesses.
        for key in gs.models.iter().filter(|(_, m)| m.visible).map(|(k, _)| k) {
            let model = &viewer.models.get(key).expect("model");
            let unedited_model = unedited_models.get(key).expect("unedited model");

            viewer.preprocessor.preprocess(
                &mut encoder,
                match show_unedited_model {
                    true => &unedited_model.preprocessor_bind_group,
                    false => &model.bind_groups.preprocessor,
                },
                model.gaussian_buffers.gaussians_buffer.len() as u32,
            );

            viewer.radix_sorter.sort(
                &mut encoder,
                &model.bind_groups.radix_sorter,
                &model.gaussian_buffers.radix_sort_indirect_args_buffer,
            );
        }

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    /// Initialize the scene.
    ///
    /// Gaussians loaded, currently selecting compression settings for viewer.
    ///
    /// Returns true if confirmed, false if cancelled, [`None`] if not yet confirmed.
    fn initialize(
        &mut self,
        ui: &mut egui::Ui,
        frame: &mut eframe::Frame,
        gs: &mut app::GaussianSplatting,
        compressions: &mut app::Compressions,
    ) -> Result<Option<bool>, String> {
        egui::Modal::new(egui::Id::new("initialize_scene_modal"))
            .show(ui.ctx(), |ui| {
                ui.add(egui::Label::new(
                    egui::RichText::new("Model loaded successfully ✅").heading(),
                ));
                ui.separator();
                ui.label("Please confirm the settings for initializing the scene");
                ui.label("");

                egui::Grid::new("initialize_scene_grid")
                    .striped(true)
                    .show(ui, |ui| {
                        ui.add(egui::Label::new(egui::RichText::new("Property").strong()));
                        ui.add(egui::Label::new(
                            egui::RichText::new("Compression").strong(),
                        ));
                        ui.add(egui::Label::new(egui::RichText::new("Size").strong()));
                        ui.end_row();

                        ui.label("Position");
                        ui.label("N/A");
                        ui.label(util::human_readable_size(
                            std::mem::size_of::<Vec3>()
                                * gs.selected_model().gaussians.gaussians.capacity(),
                        ));
                        ui.end_row();

                        ui.label("Color");
                        ui.label("N/A");
                        ui.label(util::human_readable_size(
                            std::mem::size_of::<U8Vec4>()
                                * gs.selected_model().gaussians.gaussians.capacity(),
                        ));
                        ui.end_row();

                        ui.label("Spherical Harmonics");
                        egui::ComboBox::from_id_salt("initialize_scene_sh_compression")
                            .width(150.0)
                            .selected_text(compressions.sh.to_string())
                            .show_ui(ui, |ui| {
                                for sh in app::ShCompression::iter() {
                                    ui.selectable_value(&mut compressions.sh, sh, sh.to_string());
                                }

                                gs.compressions.sh = compressions.sh;
                            });
                        ui.label(util::human_readable_size(
                            match compressions.sh {
                                app::ShCompression::Single => std::mem::size_of::<
                                    <gs::GaussianShSingleConfig as gs::GaussianShConfig>::Field,
                                >(),
                                app::ShCompression::Half => std::mem::size_of::<
                                    <gs::GaussianShHalfConfig as gs::GaussianShConfig>::Field,
                                >(),
                                app::ShCompression::Norm8 => std::mem::size_of::<
                                    <gs::GaussianShNorm8Config as gs::GaussianShConfig>::Field,
                                >(),
                                app::ShCompression::Remove => std::mem::size_of::<
                                    <gs::GaussianShNoneConfig as gs::GaussianShConfig>::Field,
                                >(),
                            } * gs.selected_model().gaussians.gaussians.capacity(),
                        ));
                        ui.end_row();

                        ui.label("Covariance 3D");
                        egui::ComboBox::from_id_salt("loading_scene_cov3d_compression")
                            .width(150.0)
                            .selected_text(compressions.cov3d.to_string())
                            .show_ui(ui, |ui| {
                                for cov3d in app::Cov3dCompression::iter() {
                                    ui.selectable_value(
                                        &mut compressions.cov3d,
                                        cov3d,
                                        cov3d.to_string(),
                                    );
                                }

                                gs.compressions.cov3d = compressions.cov3d;
                            });
                        ui.label(util::human_readable_size(
                            match compressions.cov3d {
                                app::Cov3dCompression::Single => std::mem::size_of::<
                                    <gs::GaussianCov3dSingleConfig as gs::GaussianCov3dConfig>
                                        ::Field,
                                >(),
                                app::Cov3dCompression::Half => std::mem::size_of::<
                                    <gs::GaussianCov3dHalfConfig as gs::GaussianCov3dConfig>
                                        ::Field,
                                >(),
                            } * gs.selected_model().gaussians.gaussians.capacity(),
                        ));
                        ui.end_row();
                    });

                ui.label("");

                ui.label(format!(
                    "Gaussian Count: {}",
                    gs.selected_model()
                        .gaussians
                        .gaussians
                        .capacity()
                        .to_formatted_string(&num_format::Locale::en)
                ));

                ui.label(format!(
                    "Original Size: {}",
                    util::human_readable_size(
                        gs.selected_model().gaussians.gaussians.capacity()
                            * std::mem::size_of::<gs::PlyGaussianPod>()
                    )
                ));
                ui.label(format!(
                    "Compressed Size: {}",
                    util::human_readable_size(
                        compressions
                            .compressed_size(gs.selected_model().gaussians.gaussians.capacity())
                    )
                ));
                ui.label("");

                ui.horizontal(|ui| {
                    if ui.button("Confirm").clicked() {
                        macro_rules! case {
                            ($sh:ident, $cov3d:ident) => {
                                app::Compressions {
                                    sh: app::ShCompression::$sh,
                                    cov3d: app::Cov3dCompression::$cov3d,
                                }
                            };
                        }

                        macro_rules! new {
                            ($sh:ident, $cov3d:ident, $frame:expr, $gs:expr) => {
                                paste::paste! {
                                    frame
                                        .wgpu_render_state()
                                        .expect("render state")
                                        .renderer
                                        .write()
                                        .callback_resources
                                        .insert(SceneResource::<
                                            gs::[< GaussianPodWithSh $sh Cov3d $cov3d Configs >]
                                        >::new(
                                            $frame.wgpu_render_state().expect("render state"),
                                            $gs.selected_model().file_name.clone(),
                                            $gs.selected_model().gaussians.gaussians.capacity(),
                                        ))
                                }
                            };
                        }

                        match compressions {
                            case!(Single, Single) => {
                                new!(Single, Single, frame, gs);
                            }
                            case!(Single, Half) => {
                                new!(Single, Half, frame, gs);
                            }
                            case!(Half, Single) => {
                                new!(Half, Single, frame, gs);
                            }
                            case!(Half, Half) => {
                                new!(Half, Half, frame, gs);
                            }
                            case!(Norm8, Single) => {
                                new!(Norm8, Single, frame, gs);
                            }
                            case!(Norm8, Half) => {
                                new!(Norm8, Half, frame, gs);
                            }
                            case!(Remove, Single) => {
                                new!(None, Single, frame, gs);
                            }
                            case!(Remove, Half) => {
                                new!(None, Half, frame, gs);
                            }
                        }

                        return Ok(Some(true));
                    }

                    if ui.button("Cancel").clicked() {
                        return Ok(Some(false));
                    }

                    Ok(None)
                })
                .inner
            })
            .inner
    }
}

/// The input state for [`Scene`].
#[derive(Debug)]
struct SceneInput {
    /// Whether the scene is focused.
    focused: bool,

    /// The previous modifier state.
    ///
    /// Currently this is for selection operation only.
    prev_modifiers: egui::Modifiers,

    /// The web event listener.
    ///
    /// This is only available on the web.
    #[cfg(target_arch = "wasm32")]
    web_event_listener: SceneInputWebEventListener,
}

impl SceneInput {
    /// Create a new scene input state.
    fn new() -> Self {
        Self {
            focused: false,

            prev_modifiers: egui::Modifiers::default(),

            #[cfg(target_arch = "wasm32")]
            web_event_listener: SceneInputWebEventListener::new(),
        }
    }

    /// Handle the scene input.
    fn handle(
        &mut self,
        ui: &mut egui::Ui,
        gs: &mut app::GaussianSplatting,
        query: &mut Query,
        rect: &egui::Rect,
        response: &egui::Response,
    ) {
        #[cfg(target_arch = "wasm32")]
        let web_result = self.web_event_listener.update();

        if gs.action.is_some() {
            self.action(ui, gs, query, rect, response);
        } else {
            self.control(
                ui,
                gs,
                rect,
                response,
                #[cfg(target_arch = "wasm32")]
                &web_result,
            );
        }

        self.prev_modifiers = ui.ctx().input(|input| input.modifiers);
    }

    /// Handle action.
    fn action(
        &mut self,
        ui: &mut egui::Ui,
        gs: &mut app::GaussianSplatting,
        query: &mut Query,
        rect: &egui::Rect,
        response: &egui::Response,
    ) {
        // Receive query result
        match &gs.action {
            Some(app::Action::MeasurementLocateHit {
                hit_pair_index,
                hit_index,
                rx,
                ..
            }) => {
                if let Ok(hit) = rx.try_recv() {
                    gs.measurement.hit_pairs[*hit_pair_index].hits[*hit_index].pos = hit;
                    gs.action = None;
                    gs.scene_tx
                        .send(app::SceneCommand::UpdateMeasurementHit)
                        .expect("send gs");
                }
            }
            None | Some(app::Action::Selection { .. }) => {}
        }

        // Do action
        match &mut gs.action {
            Some(app::Action::MeasurementLocateHit { tx, .. }) => {
                if !response.clicked_by(egui::PointerButton::Primary) {
                    *query = Query::none();
                    return;
                }

                let interact_pos = response.interact_pointer_pos().expect("pointer pos");

                if !rect.contains(interact_pos) {
                    *query = Query::none();
                    return;
                }

                let pos = (interact_pos - rect.min).to_pos2();
                *query = Query::measurement_locate_hit(pos, gs.measurement.hit_method, tx.clone());
            }
            Some(app::Action::Selection) => {
                let app::Selection {
                    method,
                    operation,
                    immediate,
                    brush_radius,
                    ..
                } = &mut gs.selection;

                // Pos
                let Some(hover_pos) = response.hover_pos() else {
                    *query = Query::none();
                    return;
                };

                if !rect.contains(hover_pos) {
                    *query = Query::none();
                    return;
                }

                let pos = Vec2::from_array((hover_pos - rect.min).to_pos2().into());

                // Brush radius
                if *method == app::SelectionMethod::Brush {
                    let scroll_delta = ui
                        .ctx()
                        .input(|input| (input.raw_scroll_delta.y as i32).signum());

                    *brush_radius = (*brush_radius as i32 + scroll_delta).clamp(1, 200) as u32;
                }

                // Operation
                let (shift, ctrl) = ui
                    .ctx()
                    .input(|input| (input.modifiers.shift_only(), input.modifiers.command_only()));

                if shift {
                    *operation = gs::QuerySelectionOp::Add;
                } else if ctrl {
                    *operation = gs::QuerySelectionOp::Remove;
                } else if self.prev_modifiers.shift_only() || self.prev_modifiers.command_only() {
                    *operation = gs::QuerySelectionOp::Set;
                }

                // End
                if ui
                    .ctx()
                    .input(|input| input.pointer.button_released(egui::PointerButton::Primary))
                {
                    *query = Query::selection(
                        Some(QuerySelectionAction::End),
                        *operation,
                        *immediate,
                        *brush_radius,
                        pos,
                    );
                    return;
                }

                // Update
                if !ui
                    .ctx()
                    .input(|input| input.pointer.button_down(egui::PointerButton::Primary))
                {
                    *query = Query::selection(None, *operation, *immediate, *brush_radius, pos);
                    return;
                }

                // Start
                let action = match query {
                    Query::None { .. } | Query::Selection { action: None, .. } => {
                        Some(match method {
                            app::SelectionMethod::Rect => {
                                QuerySelectionAction::Start(gs::QueryToolsetTool::Rect)
                            }
                            app::SelectionMethod::Brush => {
                                QuerySelectionAction::Start(gs::QueryToolsetTool::Brush)
                            }
                        })
                    }
                    _ => None,
                };

                *query = Query::selection(action, *operation, *immediate, *brush_radius, pos);
            }
            None => {
                *query = Query::none();
            }
        }
    }

    /// Handle focus.
    fn focus(
        &mut self,
        ui: &mut egui::Ui,
        response: &egui::Response,
        #[cfg(target_arch = "wasm32")] web_result: &SceneInputWebEventResult,
    ) {
        #[allow(unused_mut)]
        let mut prev_focused = self.focused;

        if ui.ctx().input(|input| input.pointer.any_down()) {
            self.focused = response.is_pointer_button_down_on();
        }

        if ui.ctx().input(|input| input.key_pressed(egui::Key::Escape)) {
            self.focused = false;
        }

        #[cfg(target_arch = "wasm32")]
        if let Some(locked) = web_result.pointer_lock_changed {
            if prev_focused != locked {
                prev_focused = locked;
                self.focused = locked;
            }
        }

        if prev_focused != self.focused {
            #[cfg(not(target_arch = "wasm32"))]
            {
                ui.ctx()
                    .send_viewport_cmd(egui::ViewportCommand::CursorGrab(match self.focused {
                        true => egui::CursorGrab::Confined,
                        false => egui::CursorGrab::None,
                    }));
                ui.ctx()
                    .send_viewport_cmd(egui::ViewportCommand::CursorVisible(!self.focused));
            }

            #[cfg(target_arch = "wasm32")]
            match self.focused {
                true => {
                    app::App::get_canvas().request_pointer_lock();
                }
                false => {
                    app::App::get_document().exit_pointer_lock();
                }
            }
        }
    }

    /// Handle the scene camera control.
    fn control(
        &mut self,
        ui: &mut egui::Ui,
        gs: &mut app::GaussianSplatting,
        rect: &egui::Rect,
        response: &egui::Response,
        #[cfg(target_arch = "wasm32")] web_result: &SceneInputWebEventResult,
    ) {
        if !response.contains_pointer() {
            return;
        }

        match gs.camera.control {
            app::CameraControl::FirstPerson(_) => {
                self.control_by_first_person(
                    ui,
                    gs,
                    response,
                    #[cfg(target_arch = "wasm32")]
                    web_result,
                );
            }
            app::CameraControl::Orbit(_) => {
                self.control_by_orbit(ui, gs, rect, response);
            }
        }
    }

    /// Handle the scene camera by first person control.
    fn control_by_first_person(
        &mut self,
        ui: &mut egui::Ui,
        gs: &mut app::GaussianSplatting,
        response: &egui::Response,
        #[cfg(target_arch = "wasm32")] web_result: &SceneInputWebEventResult,
    ) {
        self.focus(
            ui,
            response,
            #[cfg(target_arch = "wasm32")]
            web_result,
        );

        if !self.focused {
            return;
        }

        let control = match &mut gs.camera.control {
            app::CameraControl::FirstPerson(control) => control,
            _ => {
                log::error!("First person control expected");
                return;
            }
        };
        let dt = ui.ctx().input(|input| input.unstable_dt);

        let mut movement = Vec3::ZERO;

        let mut forward = 0.0;
        if ui.ctx().input(|input| input.key_down(egui::Key::W)) {
            forward += 1.0;
        }
        if ui.ctx().input(|input| input.key_down(egui::Key::S)) {
            forward -= 1.0;
        }

        movement += control.get_forward().with_y(0.0).normalize_or_zero() * forward;

        let mut right = 0.0;
        if ui.ctx().input(|input| input.key_down(egui::Key::D)) {
            right += 1.0;
        }
        if ui.ctx().input(|input| input.key_down(egui::Key::A)) {
            right -= 1.0;
        }

        movement += control.get_right().with_y(0.0).normalize_or_zero() * right;

        movement = movement.normalize_or_zero() * gs.camera.speed;

        let mut up = 0.0;
        if ui.ctx().input(|input| input.key_down(egui::Key::Space)) {
            up += 1.0;
        }
        if ui.ctx().input(|input| input.modifiers.shift_only()) {
            up -= 1.0;
        }

        movement.y += up * gs.camera.speed;

        control.pos += movement * dt;

        // Camera rotation
        #[cfg(not(target_arch = "wasm32"))]
        let mouse_delta = ui.ctx().input(|input| {
            input
                .raw
                .events
                .iter()
                .filter_map(|e| match e {
                    egui::Event::MouseMoved(delta) => Some(Vec2::from_array(delta.into())),
                    _ => None,
                })
                .sum::<Vec2>()
        });

        #[cfg(target_arch = "wasm32")]
        let mouse_delta = web_result.mouse_move;

        let mut rotation = -mouse_delta * 0.01;

        if ui.ctx().input(|input| input.key_down(egui::Key::I)) {
            rotation.y += 1.0;
        }
        if ui.ctx().input(|input| input.key_down(egui::Key::K)) {
            rotation.y -= 1.0;
        }

        if ui.ctx().input(|input| input.key_down(egui::Key::J)) {
            rotation.x += 1.0;
        }
        if ui.ctx().input(|input| input.key_down(egui::Key::L)) {
            rotation.x -= 1.0;
        }

        rotation *= gs.camera.sensitivity;

        control.yaw_by(rotation.x);
        control.pitch_by(rotation.y);
    }

    /// Handle the scene camera by orbit control.
    fn control_by_orbit(
        &mut self,
        ui: &mut egui::Ui,
        gs: &mut app::GaussianSplatting,
        rect: &egui::Rect,
        response: &egui::Response,
    ) {
        let control = match &mut gs.camera.control {
            app::CameraControl::Orbit(orbit) => orbit,
            _ => {
                log::error!("Orbit control expected");
                return;
            }
        };

        // Hover cursor.
        if response.hovered() {
            let icon = match response.is_pointer_button_down_on() {
                true => egui::CursorIcon::Grabbing,
                false => egui::CursorIcon::Grab,
            };

            ui.ctx().output_mut(|out| out.cursor_icon = icon);
        }

        /// Find the updated orbit vector from pos to target.
        fn orbit(pos: Vec3, target: Vec3, delta: Vec2) -> Vec3 {
            let diff = target - pos;
            let direction = diff.normalize();

            let azimuth = direction.x.atan2(direction.z);
            let elevation = direction.y.asin();

            let azimuth = (azimuth - delta.x) % (2.0 * std::f32::consts::PI);
            let elevation = (elevation - delta.y).clamp(
                -std::f32::consts::FRAC_PI_2 + 1e-6,
                std::f32::consts::FRAC_PI_2 - 1e-6,
            );

            let direction = Vec3::new(
                elevation.cos() * azimuth.sin(),
                elevation.sin(),
                elevation.cos() * azimuth.cos(),
            );

            direction * diff.length()
        }

        // Orbit
        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = Vec2::from_array(response.drag_delta().into());
            let rotation = delta * gs.camera.sensitivity * 0.01;
            control.pos = control.target - orbit(control.pos, control.target, rotation);
        }

        // Look
        if response.dragged_by(egui::PointerButton::Middle) {
            let delta = Vec2::from_array(response.drag_delta().into());
            let rotation = delta * gs.camera.sensitivity * 0.01 * vec2(1.0, -1.0);
            control.target = control.pos - orbit(control.target, control.pos, rotation);
        }

        // Pan
        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = Vec2::from_array(response.drag_delta().into());

            let right = (control.pos - control.target).cross(Vec3::Y).normalize();
            let up = (control.target - control.pos).cross(right).normalize();

            let target_distance = (control.pos - control.target).length();

            let view_height = 2.0 * target_distance * f32::tan(control.vertical_fov * 0.5);
            let aspect_ratio = rect.width() / rect.height();
            let view_width = view_height * aspect_ratio;

            let scale_x = view_width / rect.width();
            let scale_y = view_height / rect.height();

            let world_delta = Vec2::new(delta.x * scale_x, delta.y * scale_y);

            let movement = (right * world_delta.x + up * world_delta.y) * gs.camera.speed;

            control.pos += movement;
            control.target += movement;
        }

        // Zoom
        const MAX_ZOOM: f32 = 0.1;

        let delta = ui.ctx().input(|input| input.smooth_scroll_delta.y);
        let diff = control.target - control.pos;
        let diff_length = diff.length();

        let distance_scale = diff_length * 0.001;
        let zoom_amount = delta * gs.camera.speed * distance_scale;

        if delta > 0.0 && diff_length <= zoom_amount + MAX_ZOOM {
            control.pos = control.target - diff.normalize() * MAX_ZOOM;
        } else {
            control.pos += diff.normalize() * zoom_amount;
        }
    }
}

impl Default for SceneInput {
    fn default() -> Self {
        Self::new()
    }
}

/// The action of [`Query::Selection`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuerySelectionAction {
    /// Start a selection.
    Start(gs::QueryToolsetTool),

    /// End the selection.
    End,
}

/// The query callback resources.
#[derive(Debug, Clone)]
enum Query {
    /// The query is none.
    None { pod: gs::QueryNonePod },

    /// The locate hit query.
    MeasurementLocateHit {
        /// The query POD.
        pod: gs::QueryHitPod,

        /// The query method.
        hit_method: app::MeasurementHitMethod,

        /// The query result sender.
        tx: mpsc::Sender<Vec3>,
    },

    /// The selection query.
    Selection {
        /// The action.
        action: Option<QuerySelectionAction>,

        /// The operation.
        op: gs::QuerySelectionOp,

        /// Whether it is immediate.
        immediate: bool,

        /// The brush size.
        brush_radius: u32,

        /// The mouse position.
        pos: Vec2,
    },
}

impl Query {
    /// Create a none query.
    fn none() -> Self {
        Self::None {
            pod: gs::QueryNonePod::new(),
        }
    }

    /// Create a [`Query::MeasurementLocateHit`] query.
    fn measurement_locate_hit(
        coords: egui::Pos2,
        hit_method: app::MeasurementHitMethod,
        tx: mpsc::Sender<Vec3>,
    ) -> Self {
        Self::MeasurementLocateHit {
            pod: gs::QueryHitPod::new(Vec2::from_array(coords.into())),
            hit_method,
            tx,
        }
    }

    /// Create a [`Query::Selection`] query.
    fn selection(
        action: Option<QuerySelectionAction>,
        op: gs::QuerySelectionOp,
        immediate: bool,
        brush_radius: u32,
        pos: Vec2,
    ) -> Self {
        Self::Selection {
            action,
            op,
            immediate,
            brush_radius,
            pos,
        }
    }
}

/// The query result.
#[derive(Debug)]
pub enum QueryResult {
    /// Downloading previous result.
    Downloading(oneshot::Receiver<Option<QueryResult>>),

    /// The measurement locate hit result.
    MeasurementLocateHit,
}

/// The web event listener for [`SceneInput`].
///
/// This is only available on the web.
#[cfg(target_arch = "wasm32")]
struct SceneInputWebEventListener {
    /// The sender for the web events.
    tx: mpsc::Sender<SceneInputWebEvent>,

    /// The receiver for the web events.
    rx: mpsc::Receiver<SceneInputWebEvent>,

    /// The "mousemove" event listener.
    mousemove_listener: eframe::wasm_bindgen::prelude::Closure<dyn FnMut(web_sys::MouseEvent)>,

    /// The "pointerlockchange" event listener.
    pointerlockchange_listener: eframe::wasm_bindgen::prelude::Closure<dyn FnMut(web_sys::Event)>,
}

#[cfg(target_arch = "wasm32")]
impl SceneInputWebEventListener {
    /// Create a new web event listener.
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();

        let mousemove_listener = {
            let tx = tx.clone();
            eframe::wasm_bindgen::prelude::Closure::wrap(Box::new(move |e: web_sys::MouseEvent| {
                tx.send(SceneInputWebEvent::MouseMove(Vec2::new(
                    e.movement_x() as f32,
                    e.movement_y() as f32,
                )))
                .expect("send mouse move");
            })
                as Box<dyn FnMut(web_sys::MouseEvent)>)
        };

        app::App::get_canvas()
            .add_event_listener_with_callback(
                "mousemove",
                mousemove_listener.as_ref().unchecked_ref(),
            )
            .expect("add mousemove listener");

        let pointerlockchange_listener = {
            let tx = tx.clone();
            eframe::wasm_bindgen::prelude::Closure::wrap(Box::new(move |_: web_sys::Event| {
                let pointer_locked = app::App::get_document().pointer_lock_element().is_some();

                tx.send(SceneInputWebEvent::PointerLockChange(pointer_locked))
                    .expect("send pointer lock change");
            })
                as Box<dyn FnMut(web_sys::Event)>)
        };

        app::App::get_document()
            .add_event_listener_with_callback(
                "pointerlockchange",
                pointerlockchange_listener.as_ref().unchecked_ref(),
            )
            .expect("add pointerlockchange listener");

        Self {
            tx,
            rx,

            mousemove_listener,
            pointerlockchange_listener,
        }
    }

    /// Update the web event listener.
    ///
    /// Call this once per frame to take the web events.
    fn update(&mut self) -> SceneInputWebEventResult {
        let mut result = SceneInputWebEventResult {
            mouse_move: Vec2::ZERO,
            pointer_lock_changed: None,
        };

        for event in self.rx.try_iter() {
            match event {
                SceneInputWebEvent::MouseMove(delta) => {
                    result.mouse_move += delta;
                }
                SceneInputWebEvent::PointerLockChange(locked) => {
                    result.pointer_lock_changed = Some(locked);
                }
            }
        }

        result
    }
}

#[cfg(target_arch = "wasm32")]
impl Drop for SceneInputWebEventListener {
    fn drop(&mut self) {
        app::App::get_canvas()
            .remove_event_listener_with_callback(
                "mousemove",
                self.mousemove_listener.as_ref().unchecked_ref(),
            )
            .expect("remove mousemove listener");

        app::App::get_document()
            .remove_event_listener_with_callback(
                "pointerlockchange",
                self.pointerlockchange_listener.as_ref().unchecked_ref(),
            )
            .expect("remove pointerlockchange listener");
    }
}

#[cfg(target_arch = "wasm32")]
impl std::fmt::Debug for SceneInputWebEventListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SceneInputWebEventListener")
            .field("tx", &self.tx)
            .field("rx", &self.rx)
            .finish()
    }
}

/// The result of the web events.
///
/// This is only available on the web.
#[cfg(target_arch = "wasm32")]
struct SceneInputWebEventResult {
    mouse_move: Vec2,
    pointer_lock_changed: Option<bool>,
}

/// The web event.
///
/// This is only available on the web.
#[cfg(target_arch = "wasm32")]
enum SceneInputWebEvent {
    MouseMove(Vec2),
    PointerLockChange(bool),
}

/// The buffer and bind groups for showing unedited model.
///
/// Not really a model, but data for showing unedited version of the actual model.
#[derive(Debug)]
pub struct UneditedModel {
    /// The unedited Gaussian edit buffer.
    pub gaussians_edit_buffer: gs::GaussiansEditBuffer,

    /// The preprocessor bind group.
    pub preprocessor_bind_group: wgpu::BindGroup,

    /// The renderer bind group.
    pub renderer_bind_group: wgpu::BindGroup,
}

impl UneditedModel {
    /// Create a new unedited model.
    fn new<G: gs::GaussianPod>(
        viewer: &gs::MultiModelViewer<G>,
        render_state: &egui_wgpu::RenderState,
        gaussian_buffers: &gs::MultiModelViewerGaussianBuffers<G>,
    ) -> Self {
        let gaussians_edit_buffer = gs::GaussiansEditBuffer::new(
            &render_state.device,
            gaussian_buffers.gaussians_buffer.len() as u32,
        );

        let preprocessor_bind_group = viewer.preprocessor.create_bind_group(
            &render_state.device,
            &viewer.world_buffers.camera_buffer,
            &gaussian_buffers.model_transform_buffer,
            &gaussian_buffers.gaussians_buffer,
            &gaussian_buffers.indirect_args_buffer,
            &gaussian_buffers.radix_sort_indirect_args_buffer,
            &gaussian_buffers.indirect_indices_buffer,
            &gaussian_buffers.gaussians_depth_buffer,
            &viewer.world_buffers.query_buffer,
            &gaussian_buffers.query_result_count_buffer,
            &gaussian_buffers.query_results_buffer,
            &gaussians_edit_buffer,
            &gaussian_buffers.selection_buffer,
            &viewer.world_buffers.selection_edit_buffer,
            &viewer.world_buffers.query_texture,
            &gaussian_buffers.mask_buffer,
        );

        let renderer_bind_group = viewer.renderer.create_bind_group(
            &render_state.device,
            &viewer.world_buffers.camera_buffer,
            &gaussian_buffers.model_transform_buffer,
            &viewer.world_buffers.gaussian_transform_buffer,
            &gaussian_buffers.gaussians_buffer,
            &gaussian_buffers.indirect_indices_buffer,
            &viewer.world_buffers.query_buffer,
            &gaussian_buffers.query_result_count_buffer,
            &gaussian_buffers.query_results_buffer,
            &viewer.world_buffers.selection_highlight_buffer,
            &gaussian_buffers.selection_buffer,
            &gaussians_edit_buffer,
        );

        Self {
            gaussians_edit_buffer,
            preprocessor_bind_group,
            renderer_bind_group,
        }
    }

    /// Update the bind group.
    ///
    /// This is for when query texture is updated.
    fn update_bind_group<G: gs::GaussianPod>(
        &mut self,
        device: &wgpu::Device,
        viewer: &gs::MultiModelViewer<G>,
        gaussian_buffers: &gs::MultiModelViewerGaussianBuffers<G>,
    ) {
        self.preprocessor_bind_group = viewer.preprocessor.create_bind_group(
            device,
            &viewer.world_buffers.camera_buffer,
            &gaussian_buffers.model_transform_buffer,
            &gaussian_buffers.gaussians_buffer,
            &gaussian_buffers.indirect_args_buffer,
            &gaussian_buffers.radix_sort_indirect_args_buffer,
            &gaussian_buffers.indirect_indices_buffer,
            &gaussian_buffers.gaussians_depth_buffer,
            &viewer.world_buffers.query_buffer,
            &gaussian_buffers.query_result_count_buffer,
            &gaussian_buffers.query_results_buffer,
            &self.gaussians_edit_buffer,
            &gaussian_buffers.selection_buffer,
            &viewer.world_buffers.selection_edit_buffer,
            &viewer.world_buffers.query_texture,
            &gaussian_buffers.mask_buffer,
        );
    }
}

/// The mask gizmos resource for [`SceneResource`].
#[derive(Debug)]
pub struct MaskGizmosResource {
    /// The gizmo.
    pub gizmo: gs::MaskGizmo,

    /// The box gizmos.
    pub box_gizmos: Vec<gs::MaskGizmoPod>,

    /// The sphere gizmos.
    pub ellipsoid_gizmos: Vec<gs::MaskGizmoPod>,
}

/// The scene resource.
///
/// This is for the [`SceneCallback`].
#[derive(Debug)]
pub struct SceneResource<G: gs::GaussianPod> {
    /// The viewer.
    ///
    /// The viewer should not be used in multiple threads in native, always use blocking code.
    ///
    /// Required to use [`Mutex`] because the callback resources requires [`Send`] and [`Sync`]
    /// on native.
    pub viewer: Arc<Mutex<gs::MultiModelViewer<G>>>,

    /// The measurement renderer.
    pub measurement_renderer: renderer::Measurement,

    /// The visible measurement hit pair.
    pub measurement_visible_hit_pairs: Vec<app::MeasurementHitPair>,

    /// The query toolset.
    pub query_toolset: gs::QueryToolset,

    /// The query texture overlay.
    pub query_texture_overlay: gs::QueryTextureOverlay,

    /// The query cursor.
    pub query_cursor: gs::QueryCursor,

    /// The unedited models, that is, unedited.
    pub unedited_models: HashMap<String, UneditedModel>,

    /// Whether the unedited model is shown, compared to the edited model.
    ///
    /// This is updated by [`Scene::loaded_preprocess`] so that [`SceneCallback`] can know whether
    /// to use this.
    pub show_unedited_model: bool,

    /// The mask evaluator.
    pub mask_evaluator: gs::MaskEvaluator,

    /// The mask gizmos.
    pub mask_gizmos: HashMap<String, MaskGizmosResource>,
}

impl<G: gs::GaussianPod> SceneResource<G> {
    /// Create a new scene resource.
    fn new(render_state: &egui_wgpu::RenderState, key: String, count: usize) -> Self {
        log::debug!("Creating viewer");
        // In WASM, the viewer is not Send nor Sync, but in native, it is.
        #[allow(clippy::arc_with_non_send_sync)]
        let viewer = Arc::new(Mutex::new(gs::MultiModelViewer::new_with(
            &render_state.device,
            render_state.target_format,
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            uvec2(1, 1),
        )));

        let mut locked_viewer = viewer.lock().expect("viewer");

        log::debug!("Creating measurement renderer");
        let measurement_renderer = renderer::Measurement::new(
            &render_state.device,
            render_state.target_format,
            &locked_viewer.world_buffers.camera_buffer,
        );

        let measurement_visible_hit_pairs = Vec::new();

        log::debug!("Creating query toolset");
        let query_toolset = {
            gs::QueryToolset::new(
                &render_state.device,
                &locked_viewer.world_buffers.query_texture,
                &locked_viewer.world_buffers.camera_buffer,
            )
        };

        log::debug!("Creating query texture overlay");
        let query_texture_overlay = gs::QueryTextureOverlay::new_with(
            &render_state.device,
            render_state.target_format,
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            &locked_viewer.world_buffers.query_texture,
        );

        log::debug!("Creating query cursor");
        let query_cursor = gs::QueryCursor::new_with(
            &render_state.device,
            render_state.target_format,
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            &locked_viewer.world_buffers.camera_buffer,
        );

        log::debug!("Creating unedited models");
        let mut unedited_models = HashMap::new();

        log::debug!("Creating mask evaluator");
        let mask_evaluator = gs::MaskEvaluator::new::<G>(&render_state.device);

        log::debug!("Creating mask gizmos");
        let mut mask_gizmos = HashMap::new();

        log::debug!("Initializing first model");
        Self::add_model_with_viewer(
            &mut locked_viewer,
            &mut unedited_models,
            &mut mask_gizmos,
            render_state,
            &mask_evaluator,
            key,
            count,
        );

        std::mem::drop(locked_viewer);

        log::info!("Scene loaded");

        Self {
            viewer,
            measurement_renderer,
            measurement_visible_hit_pairs,
            query_toolset,
            query_texture_overlay,
            query_cursor,
            unedited_models,
            show_unedited_model: false,
            mask_evaluator,
            mask_gizmos,
        }
    }

    /// Load Gaussians for a model.
    fn load_model(
        &mut self,
        render_state: &egui_wgpu::RenderState,
        key: &str,
        start: usize,
        gaussians: &[gs::Gaussian],
    ) {
        self.viewer
            .lock()
            .expect("viewer")
            .models
            .get(key)
            .expect("model")
            .gaussian_buffers
            .gaussians_buffer
            .update_range(&render_state.queue, start, gaussians);
    }

    /// Add a new model.
    fn add_model(&mut self, render_state: &egui_wgpu::RenderState, key: String, count: usize) {
        let mut viewer = self.viewer.lock().expect("viewer");
        Self::add_model_with_viewer(
            &mut viewer,
            &mut self.unedited_models,
            &mut self.mask_gizmos,
            render_state,
            &self.mask_evaluator,
            key,
            count,
        );
    }

    /// Add a new model with a viewer.
    fn add_model_with_viewer(
        viewer: &mut gs::MultiModelViewer<G>,
        unedited_models: &mut HashMap<String, UneditedModel>,
        mask_gizmos: &mut HashMap<String, MaskGizmosResource>,
        render_state: &egui_wgpu::RenderState,
        mask_evaluator: &gs::MaskEvaluator,
        key: String,
        count: usize,
    ) {
        let gaussian_buffers =
            gs::MultiModelViewerGaussianBuffers::new_empty(&render_state.device, count);
        let bind_groups = gs::MultiModelViewerBindGroups::new(
            &render_state.device,
            &viewer.preprocessor,
            &viewer.radix_sorter,
            &viewer.renderer,
            &viewer.postprocessor,
            &gaussian_buffers,
            &viewer.world_buffers,
        );
        let unedited_model = UneditedModel::new(viewer, render_state, &gaussian_buffers);

        mask_evaluator.evaluate(
            &render_state.device,
            &render_state.queue,
            &gs::MaskOpTree::Reset,
            &gaussian_buffers.mask_buffer,
            &gaussian_buffers.model_transform_buffer,
            &gaussian_buffers.gaussians_buffer,
        );

        viewer.models.insert(
            key.clone(),
            gs::MultiModelViewerModel {
                gaussian_buffers,
                bind_groups,
            },
        );
        unedited_models.insert(key.clone(), unedited_model);
        mask_gizmos.insert(
            key,
            MaskGizmosResource {
                gizmo: gs::MaskGizmo::new_with(
                    &render_state.device,
                    render_state.target_format,
                    &viewer.world_buffers.camera_buffer,
                    Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                ),
                box_gizmos: Vec::new(),
                ellipsoid_gizmos: Vec::new(),
            },
        );
    }

    /// Remove a model.
    fn remove_model(&mut self, key: &String) {
        if self.viewer.lock().expect("viewer").models.len() == 1 {
            log::error!("Cannot remove the last model");
            return;
        }

        self.viewer.lock().expect("viewer").remove_model(key);
    }

    /// Update the measurement visible hit pair.
    fn update_measurement_visible_hit_pairs(&mut self, hit_pairs: &[app::MeasurementHitPair]) {
        self.measurement_visible_hit_pairs.clear();
        self.measurement_visible_hit_pairs.extend(
            hit_pairs
                .iter()
                .filter(|hit_pair| hit_pair.visible)
                .cloned(),
        );
    }

    /// Evaluate the mask given the op code.
    fn evaluate_mask(
        &mut self,
        render_state: &egui_wgpu::RenderState,
        op: Option<&app::GaussianSplattingMaskOp>,
        key: &str,
        model: &app::GaussianSplattingModel,
    ) {
        let viewer = self.viewer.lock().expect("viewer");
        let gaussian_buffers = &viewer.models.get(key).expect("model").gaussian_buffers;

        self.mask_evaluator.evaluate(
            &render_state.device,
            &render_state.queue,
            &op.map(|op| op.to_tree(&model.mask.op_shape_pods))
                .unwrap_or(gs::MaskOpTree::Reset),
            &gaussian_buffers.mask_buffer,
            &gaussian_buffers.model_transform_buffer,
            &gaussian_buffers.gaussians_buffer,
        );

        let gizmo = self.mask_gizmos.get_mut(key).expect("gizmo");

        gizmo.box_gizmos = model
            .mask
            .shapes
            .iter()
            .filter(|shape| shape.shape.kind == gs::MaskShapeKind::Box && shape.visible)
            .map(|shape| shape.shape.to_mask_gizmo_pod())
            .collect();

        gizmo.ellipsoid_gizmos = model
            .mask
            .shapes
            .iter()
            .filter(|shape| shape.shape.kind == gs::MaskShapeKind::Ellipsoid && shape.visible)
            .map(|shape| shape.shape.to_mask_gizmo_pod())
            .collect();

        if !gizmo.box_gizmos.is_empty() {
            gizmo.gizmo.update(
                &render_state.device,
                &render_state.queue,
                &viewer.world_buffers.camera_buffer,
                gs::MaskShapeKind::Box,
                &gizmo.box_gizmos,
            );
        }

        if !gizmo.ellipsoid_gizmos.is_empty() {
            gizmo.gizmo.update(
                &render_state.device,
                &render_state.queue,
                &viewer.world_buffers.camera_buffer,
                gs::MaskShapeKind::Ellipsoid,
                &gizmo.ellipsoid_gizmos,
            );
        }
    }
}

/// The scene callback.
struct SceneCallback<G: gs::GaussianPod + Send + Sync> {
    /// The model render keys.
    model_render_keys: Vec<String>,

    /// The query.
    query: Query,

    /// The phantom data.
    phantom: PhantomData<G>,
}

impl<G: gs::GaussianPod + Send + Sync> egui_wgpu::CallbackTrait for SceneCallback<G> {
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let SceneResource::<G> {
            viewer,
            measurement_renderer,
            measurement_visible_hit_pairs,
            query_toolset,
            query_texture_overlay,
            query_cursor,
            unedited_models,
            show_unedited_model,
            mask_gizmos,
            ..
        } = callback_resources.get().expect("scene resource");

        for key in self.model_render_keys.iter() {
            let gizmo = mask_gizmos.get(key).expect("gizmo");

            if !gizmo.box_gizmos.is_empty() {
                gizmo.gizmo.render_box_with_pass(render_pass);
            }

            if !gizmo.ellipsoid_gizmos.is_empty() {
                gizmo.gizmo.render_ellipsoid_with_pass(render_pass);
            }
        }

        if !measurement_visible_hit_pairs.is_empty() {
            measurement_renderer.render(render_pass, measurement_visible_hit_pairs.len() as u32);
        }

        {
            let viewer = viewer.lock().expect("viewer");

            for key in self.model_render_keys.iter() {
                let model = &viewer.models.get(key).expect("model");
                let unedited_model = unedited_models.get(key).expect("unedited model");

                viewer.renderer.render_with_pass(
                    render_pass,
                    match show_unedited_model {
                        true => &unedited_model.renderer_bind_group,
                        false => &model.bind_groups.renderer,
                    },
                    &model.gaussian_buffers.indirect_args_buffer,
                );
            }
        }

        if let Query::Selection { .. } = self.query {
            if let Some((gs::QueryToolsetUsedTool::QueryTextureTool { .. }, ..)) =
                query_toolset.state()
            {
                query_texture_overlay.render_with_pass(render_pass);
            } else {
                query_cursor.render_with_pass(render_pass);
            }
        }
    }
}
