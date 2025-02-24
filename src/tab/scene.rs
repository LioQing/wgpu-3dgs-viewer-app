use std::{
    io::Cursor,
    sync::{mpsc, Arc, Mutex},
};

#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::JsCast;

use eframe::{
    egui_wgpu,
    wgpu::{self},
};
use glam::*;
use num_format::ToFormattedString;
use strum::IntoEnumIterator;
use wgpu_3dgs_viewer::{self as gs, QueryVariant, Texture};

use crate::{
    app::{self, Unloaded},
    renderer, util,
};

use super::Tab;

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
                true => match self.loaded(ui, gs) {
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
        unloaded: &mut Unloaded<app::GaussianSplatting, String>,
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
                        let mut reader = Cursor::new(file.read().await);
                        let gs = app::GaussianSplatting::new(
                            file.file_name(),
                            &mut reader,
                            compressions,
                        )
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
                            file.name.clone(),
                            &mut Cursor::new(file.bytes.as_ref().expect("file bytes").to_vec()),
                            compressions.clone(),
                        )
                        .map_err(|e| e.to_string()),
                        false => std::fs::read(file.path.as_ref().expect("file path").clone())
                            .map_err(gs::Error::Io)
                            .map_err(|e| e.to_string())
                            .and_then(|data| {
                                app::GaussianSplatting::new(
                                    file.name.clone(),
                                    &mut Cursor::new(data),
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
    fn loaded(&mut self, ui: &mut egui::Ui, gs: &mut app::GaussianSplatting) -> bool {
        let mut loaded = true;

        ui.horizontal(|ui| {
            ui.label(format!(
                "📦 Loaded: {}",
                if gs.file_name.len() > 20 {
                    format!("{}...", &gs.file_name[..20])
                } else {
                    gs.file_name.clone()
                }
            ))
            .on_hover_text(&gs.file_name);

            ui.separator();

            loaded &= !ui.button("🗑 Close model").clicked();

            ui.separator();

            let dt = ui.ctx().input(|input| input.unstable_dt);
            self.fps_interval += dt;
            if self.fps_interval >= 1.0 {
                self.fps_interval -= 1.0;
                self.fps = 1.0 / dt;
            }

            ui.label(format!("🏃 FPS: {:.2}", self.fps));
        });

        egui::Frame::canvas(ui.style()).show(ui, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            self.input.handle(ui, gs, &mut self.query, &rect, &response);

            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect,
                SceneCallback {
                    measurement_visible_hit_pairs: gs
                        .measurement
                        .visible_hit_pairs()
                        .cloned()
                        .collect(),
                    model_transform: gs.model_transform.clone(),
                    gaussian_transform: gs.gaussian_transform.clone(),
                    camera: gs.camera.control.clone(),
                    viewer_size: Vec2::from_array(rect.size().into()),
                    gaussian_count: gs.gaussians.gaussians.len(),
                    query: self.query.clone(),
                },
            ));
        });

        loaded
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
                            std::mem::size_of::<Vec3>() * gs.gaussians.gaussians.len(),
                        ));
                        ui.end_row();

                        ui.label("Color");
                        ui.label("N/A");
                        ui.label(util::human_readable_size(
                            std::mem::size_of::<U8Vec4>() * gs.gaussians.gaussians.len(),
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
                            } * gs.gaussians.gaussians.len(),
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
                                    <gs::GaussianCov3dSingleConfig as gs::GaussianCov3dConfig>::Field,
                                >(),
                                app::Cov3dCompression::Half => std::mem::size_of::<
                                    <gs::GaussianCov3dHalfConfig as gs::GaussianCov3dConfig>::Field,
                                >(),
                            } * gs.gaussians.gaussians.len(),
                        ));
                        ui.end_row();
                    });

                ui.label("");

                ui.label(format!(
                    "Gaussian Count: {}",
                    gs.gaussians.gaussians.len().to_formatted_string(&num_format::Locale::en)
                ));

                ui.label(format!(
                    "Original Size: {}",
                    util::human_readable_size(
                        gs.gaussians.gaussians.len() * std::mem::size_of::<gs::PlyGaussianPod>()
                    )
                ));
                ui.label(format!(
                    "Compressed Size: {}",
                    util::human_readable_size(
                        compressions.compressed_size(gs.gaussians.gaussians.len())
                    )
                ));
                ui.label("");

                ui.horizontal(|ui| {
                    if ui.button("Confirm").clicked() {
                        match SceneResource::new(
                            frame.wgpu_render_state().expect("render state"),
                            &gs.gaussians,
                            compressions,
                        ) {
                            Ok(scene_resource) => {
                                log::debug!("Scene resource initialized");

                                frame
                                    .wgpu_render_state()
                                    .expect("render state")
                                    .renderer
                                    .write()
                                    .callback_resources
                                    .insert(scene_resource);

                                return Ok(Some(true));
                            }
                            Err(e) => {
                                log::debug!("Error initializing scene resource: {e}");

                                return Err(e.to_string());
                            }
                        }
                    }

                    if ui.button("Cancel").clicked() {
                        return Ok(Some(false));
                    }

                    Ok(None)
                }).inner
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
                self.control_by_orbit(ui, gs, response);
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

        movement += control.get_forward().with_y(0.0).normalize() * forward;

        let mut right = 0.0;
        if ui.ctx().input(|input| input.key_down(egui::Key::D)) {
            right += 1.0;
        }
        if ui.ctx().input(|input| input.key_down(egui::Key::A)) {
            right -= 1.0;
        }

        movement += control.get_right().with_y(0.0).normalize() * right;

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

        let mut rotation = -mouse_delta * 0.5;

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

        control.yaw_by(rotation.x * dt);
        control.pitch_by(rotation.y * dt);
    }

    /// Handle the scene camera by orbit control.
    fn control_by_orbit(
        &mut self,
        ui: &mut egui::Ui,
        gs: &mut app::GaussianSplatting,
        response: &egui::Response,
    ) {
        let control = match &mut gs.camera.control {
            app::CameraControl::Orbit(orbit) => orbit,
            _ => {
                log::error!("Orbit control expected");
                return;
            }
        };
        let dt = ui.ctx().input(|input| input.unstable_dt);

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
            let rotation = delta * gs.camera.sensitivity * dt;
            control.pos = control.target - orbit(control.pos, control.target, rotation);
        }

        // Look
        if response.dragged_by(egui::PointerButton::Middle) {
            let delta = Vec2::from_array(response.drag_delta().into());
            let rotation = delta * gs.camera.sensitivity * dt * vec2(1.0, -1.0);
            control.target = control.pos - orbit(control.target, control.pos, rotation);
        }

        // Pan
        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = Vec2::from_array(response.drag_delta().into());
            let right = (control.pos - control.target).cross(Vec3::Y).normalize();
            let up = (control.target - control.pos).cross(right).normalize();
            let movement = (right * delta.x + up * delta.y) * 0.5 * gs.camera.speed * dt;
            control.pos += movement;
            control.target += movement;
        }

        // Zoom
        const MAX_ZOOM: f32 = 0.1;

        let delta = ui.ctx().input(|input| input.smooth_scroll_delta.y);
        let diff = control.target - control.pos;
        let zoom = diff.normalize() * delta * 0.5 * gs.camera.speed * dt;

        if delta > 0.0 && diff.length_squared() <= zoom.length_squared() + MAX_ZOOM * MAX_ZOOM {
            control.pos = control.target - diff.normalize() * MAX_ZOOM;
        } else {
            control.pos += zoom;
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

/// The viewer with different compression settings.
#[derive(Debug)]
enum Viewer {
    ShSingleCov3dSingle(gs::Viewer<gs::GaussianPodWithShSingleCov3dSingleConfigs>),
    ShSingleCov3dHalf(gs::Viewer<gs::GaussianPodWithShSingleCov3dHalfConfigs>),
    ShHalfCov3dSingle(gs::Viewer<gs::GaussianPodWithShHalfCov3dSingleConfigs>),
    ShHalfCov3dHalf(gs::Viewer<gs::GaussianPodWithShHalfCov3dHalfConfigs>),
    ShNorm8Cov3dSingle(gs::Viewer<gs::GaussianPodWithShNorm8Cov3dSingleConfigs>),
    ShNorm8Cov3dHalf(gs::Viewer<gs::GaussianPodWithShNorm8Cov3dHalfConfigs>),
    ShRemoveCov3dSingle(gs::Viewer<gs::GaussianPodWithShNoneCov3dSingleConfigs>),
    ShRemoveCov3dHalf(gs::Viewer<gs::GaussianPodWithShNoneCov3dHalfConfigs>),
}

macro_rules! viewer_call {
    ($self:expr, ref $field:ident) => {
        match $self {
            Self::ShSingleCov3dSingle(viewer) => &viewer.$field,
            Self::ShSingleCov3dHalf(viewer) => &viewer.$field,
            Self::ShHalfCov3dSingle(viewer) => &viewer.$field,
            Self::ShHalfCov3dHalf(viewer) => &viewer.$field,
            Self::ShNorm8Cov3dSingle(viewer) => &viewer.$field,
            Self::ShNorm8Cov3dHalf(viewer) => &viewer.$field,
            Self::ShRemoveCov3dSingle(viewer) => &viewer.$field,
            Self::ShRemoveCov3dHalf(viewer) => &viewer.$field,
        }
    };
    ($self:expr, ref mut $field:ident) => {
        match $self {
            Self::ShSingleCov3dSingle(viewer) => &mut viewer.$field,
            Self::ShSingleCov3dHalf(viewer) => &mut viewer.$field,
            Self::ShHalfCov3dSingle(viewer) => &mut viewer.$field,
            Self::ShHalfCov3dHalf(viewer) => &mut viewer.$field,
            Self::ShNorm8Cov3dSingle(viewer) => &mut viewer.$field,
            Self::ShNorm8Cov3dHalf(viewer) => &mut viewer.$field,
            Self::ShRemoveCov3dSingle(viewer) => &mut viewer.$field,
            Self::ShRemoveCov3dHalf(viewer) => &mut viewer.$field,
        }
    };
    ($self:expr, $field:ident) => {
        match $self {
            Self::ShSingleCov3dSingle(viewer) => viewer.$field,
            Self::ShSingleCov3dHalf(viewer) => viewer.$field,
            Self::ShHalfCov3dSingle(viewer) => viewer.$field,
            Self::ShHalfCov3dHalf(viewer) => viewer.$field,
            Self::ShNorm8Cov3dSingle(viewer) => viewer.$field,
            Self::ShNorm8Cov3dHalf(viewer) => viewer.$field,
            Self::ShRemoveCov3dSingle(viewer) => viewer.$field,
            Self::ShRemoveCov3dHalf(viewer) => viewer.$field,
        }
    };
    ($self:expr, fn $fn:ident, $($args:expr),*) => {
        match $self {
            Self::ShSingleCov3dSingle(viewer) => viewer.$fn($($args),*),
            Self::ShSingleCov3dHalf(viewer) => viewer.$fn($($args),*),
            Self::ShHalfCov3dSingle(viewer) => viewer.$fn($($args),*),
            Self::ShHalfCov3dHalf(viewer) => viewer.$fn($($args),*),
            Self::ShNorm8Cov3dSingle(viewer) => viewer.$fn($($args),*),
            Self::ShNorm8Cov3dHalf(viewer) => viewer.$fn($($args),*),
            Self::ShRemoveCov3dSingle(viewer) => viewer.$fn($($args),*),
            Self::ShRemoveCov3dHalf(viewer) => viewer.$fn($($args),*),
        }
    };
    ($self:expr, async fn $fn:ident, $($args:expr),*) => {
        match $self {
            Self::ShSingleCov3dSingle(viewer) => viewer.$fn($($args),*).await,
            Self::ShSingleCov3dHalf(viewer) => viewer.$fn($($args),*).await,
            Self::ShHalfCov3dSingle(viewer) => viewer.$fn($($args),*).await,
            Self::ShHalfCov3dHalf(viewer) => viewer.$fn($($args),*).await,
            Self::ShNorm8Cov3dSingle(viewer) => viewer.$fn($($args),*).await,
            Self::ShNorm8Cov3dHalf(viewer) => viewer.$fn($($args),*).await,
            Self::ShRemoveCov3dSingle(viewer) => viewer.$fn($($args),*).await,
            Self::ShRemoveCov3dHalf(viewer) => viewer.$fn($($args),*).await,
        }
    };
}

macro_rules! viewer_getters {
    ($field:ident) => {
        paste::paste! {
            pub fn $field(&self) -> &gs::[< $field:camel >] {
                viewer_call!(self, ref $field)
            }

            pub fn [< $field _mut >](&mut self) -> &mut gs::[< $field:camel >] {
                viewer_call!(self, ref mut $field)
            }
        }
    };
}

#[allow(dead_code)]
impl Viewer {
    /// Create a new viewer.
    pub fn new(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        texture_size: UVec2,
        gaussians: &gs::Gaussians,
        compressions: &app::Compressions,
    ) -> Result<Self, gs::Error> {
        macro_rules! case {
            ($sh:ident, $cov3d:ident) => {
                app::Compressions {
                    sh: app::ShCompression::$sh,
                    cov3d: app::Cov3dCompression::$cov3d,
                }
            };
        }

        macro_rules! new {
            ($sh:ident, $cov3d:ident) => {
                paste::paste! {
                    Self::[<Sh $sh Cov3d $cov3d>](gs::Viewer::new(
                        device, texture_format, texture_size, gaussians
                    )?)
                }
            };
        }

        Ok(match compressions {
            case!(Single, Single) => new!(Single, Single),
            case!(Single, Half) => new!(Single, Half),
            case!(Half, Single) => new!(Half, Single),
            case!(Half, Half) => new!(Half, Half),
            case!(Norm8, Single) => new!(Norm8, Single),
            case!(Norm8, Half) => new!(Norm8, Half),
            case!(Remove, Single) => new!(Remove, Single),
            case!(Remove, Half) => new!(Remove, Half),
        })
    }

    viewer_getters!(camera_buffer);
    viewer_getters!(model_transform_buffer);
    viewer_getters!(gaussian_transform_buffer);
    viewer_getters!(indirect_args_buffer);
    viewer_getters!(radix_sort_indirect_args_buffer);
    viewer_getters!(indirect_indices_buffer);
    viewer_getters!(gaussians_depth_buffer);
    viewer_getters!(query_buffer);
    viewer_getters!(query_result_count_buffer);
    viewer_getters!(query_results_buffer);
    viewer_getters!(postprocess_indirect_args_buffer);
    viewer_getters!(selection_highlight_buffer);
    viewer_getters!(selection_buffer);
    viewer_getters!(query_texture);

    viewer_getters!(preprocessor);
    viewer_getters!(radix_sorter);
    viewer_getters!(renderer);
    viewer_getters!(postprocessor);

    /// Update the camera.
    pub fn update_camera(
        &mut self,
        queue: &wgpu::Queue,
        camera: &impl gs::CameraTrait,
        texture_size: UVec2,
    ) {
        viewer_call!(self, fn update_camera, queue, camera, texture_size);
    }

    /// Update the query.
    pub fn update_query(&mut self, queue: &wgpu::Queue, query: &gs::QueryPod) {
        viewer_call!(self, fn update_query, queue, query);
    }

    /// Update the model transform.
    pub fn update_model_transform(
        &mut self,
        queue: &wgpu::Queue,
        pos: Vec3,
        quat: Quat,
        scale: Vec3,
    ) {
        viewer_call!(self, fn update_model_transform, queue, pos, quat, scale);
    }

    /// Update the Gaussian transform.
    pub fn update_gaussian_transform(
        &mut self,
        queue: &wgpu::Queue,
        size: f32,
        display_mode: gs::GaussianDisplayMode,
        sh_deg: gs::GaussianShDegree,
        no_sh0: bool,
    ) {
        viewer_call!(
            self,
            fn update_gaussian_transform,
            queue,
            size,
            display_mode,
            sh_deg,
            no_sh0
        );
    }

    /// Update the selection highlight.
    pub fn update_selection_highlight(&mut self, queue: &wgpu::Queue, color: Vec4) {
        viewer_call!(self, fn update_selection_highlight, queue, color);
    }

    /// Update the query texture size.
    ///
    /// This requires the `query-texture` feature.
    pub fn update_query_texture_size(&mut self, device: &wgpu::Device, size: UVec2) {
        viewer_call!(self, fn update_query_texture_size, device, size);
    }

    /// Render the viewer.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture_view: &wgpu::TextureView,
        gaussian_count: u32,
    ) {
        viewer_call!(self, fn render, encoder, texture_view, gaussian_count);
    }

    /// Download the query results from the GPU.
    pub async fn download_query_results(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<gs::QueryResultPod>, gs::Error> {
        viewer_call!(self, async fn download_query_results, device, queue)
    }
}

/// The scene resource.
///
/// This is for the [`SceneCallback`].
#[derive(Debug)]
struct SceneResource {
    /// The viewer.
    viewer: Arc<Mutex<Viewer>>,

    /// The measurement renderer.
    measurement_renderer: renderer::Measurement,

    /// The query resource.
    ///
    /// When the query is not none, all following query will be ignored until the result is
    /// received.
    query_resource: Option<SceneQueryResource>,

    /// The query toolset.
    query_toolset: gs::QueryToolset,

    /// The query texture overlay.
    query_texture_overlay: gs::QueryTextureOverlay,

    /// The query cursor.
    query_cursor: gs::QueryCursor,
}

impl SceneResource {
    /// Create a new scene resource.
    fn new(
        render_state: &egui_wgpu::RenderState,
        gaussians: &gs::Gaussians,
        compressions: &app::Compressions,
    ) -> Result<Self, String> {
        log::debug!("Creating viewer");
        // In WASM, the viewer is not Send nor Sync, but in native, it is.
        #[allow(clippy::arc_with_non_send_sync)]
        let viewer = Arc::new(Mutex::new(
            Viewer::new(
                &render_state.device,
                render_state.target_format,
                uvec2(1, 1),
                gaussians,
                compressions,
            )
            .map_err(|e| e.to_string())?,
        ));

        log::debug!("Creating measurement renderer");
        let measurement_renderer = renderer::Measurement::new(
            &render_state.device,
            render_state.target_format,
            viewer.lock().expect("viewer").camera_buffer(),
        );

        log::debug!("Creating query resource");
        let query_resource = None;

        log::debug!("Creating query toolset");
        let query_toolset = {
            let viewer = viewer.lock().expect("viewer");

            gs::QueryToolset::new(
                &render_state.device,
                viewer.query_texture(),
                viewer.camera_buffer(),
            )
        };

        log::debug!("Creating query texture overlay");
        let query_texture_overlay = gs::QueryTextureOverlay::new(
            &render_state.device,
            render_state.target_format,
            viewer.lock().expect("viewer").query_texture(),
        );

        log::debug!("Creating query cursor");
        let query_cursor = gs::QueryCursor::new(
            &render_state.device,
            render_state.target_format,
            viewer.lock().expect("viewer").camera_buffer(),
        );

        log::info!("Scene loaded");

        Ok(Self {
            viewer,
            measurement_renderer,
            query_resource,
            query_toolset,
            query_texture_overlay,
            query_cursor,
        })
    }
}

/// The scene query resource.
///
/// This is the query from the previous scene callbacks, since it is not possible to map the
/// buffer synchronously, we may need multiple frames to get the query result.
#[derive(Debug)]
struct SceneQueryResource {
    /// The query.
    query: Query,

    #[cfg(target_arch = "wasm32")]
    /// The query stage.
    stage: SceneQueryStage,
}

#[cfg(target_arch = "wasm32")]
/// The scene query stage.
#[derive(Debug)]
enum SceneQueryStage {
    /// The viewer is querying.
    Querying,
    /// The query is downloading count.
    DownloadingCount {
        /// The receiver.
        rx: oneshot::Receiver<u32>,
    },
    /// The locate hit query is downloading the results.
    MeasurementLocateHitDownloadingResults {
        /// The receiver.
        rx: oneshot::Receiver<Vec<gs::QueryHitResultPod>>,
    },
}

/// The scene callback.
struct SceneCallback {
    /// The highlighted measurement hit pair.
    measurement_visible_hit_pairs: Vec<app::MeasurementHitPair>,

    /// The Gaussian splatting Gaussian transform.
    gaussian_transform: app::GaussianSplattingGaussianTransform,

    /// The Gaussian splatting model transform.
    model_transform: app::GaussianSplattingModelTransform,

    /// The camera.
    camera: app::CameraControl,

    /// The viewer size.
    viewer_size: Vec2,

    /// The Gaussian count.
    gaussian_count: usize,

    /// The query.
    query: Query,
}

impl egui_wgpu::CallbackTrait for SceneCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let SceneResource {
            viewer,
            measurement_renderer,
            query_resource,
            query_toolset,
            query_texture_overlay,
            query_cursor,
        } = callback_resources.get_mut().expect("scene resource");

        // Postprocess, because eframe cannot do any compute pass after the render pass.
        {
            let viewer = viewer.lock().expect("viewer");

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Postprocess Encoder"),
            });
            viewer.postprocessor().postprocess(
                &mut encoder,
                self.gaussian_count as u32,
                viewer.postprocess_indirect_args_buffer(),
            );
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
        }

        // Update query texture size.
        let wgpu::Extent3d { width, height, .. } = viewer
            .lock()
            .expect("viewer")
            .query_texture()
            .texture()
            .size();
        let texture_size = uvec2(width, height);

        if texture_size != self.viewer_size.as_uvec2() {
            viewer
                .lock()
                .expect("viewer")
                .update_query_texture_size(device, self.viewer_size.as_uvec2());
            query_texture_overlay
                .update_bind_group(device, viewer.lock().expect("viewer").query_texture());
        }

        // Handle query.
        self.query(
            device,
            queue,
            egui_encoder,
            viewer,
            query_resource,
            query_toolset,
            query_cursor,
        );

        // Update the viewer.
        viewer.lock().expect("viewer").update_camera(
            queue,
            &self.camera,
            self.viewer_size.as_uvec2(),
        );
        viewer.lock().expect("viewer").update_model_transform(
            queue,
            self.model_transform.pos,
            self.model_transform.quat(),
            self.model_transform.scale,
        );
        viewer.lock().expect("viewer").update_gaussian_transform(
            queue,
            self.gaussian_transform.size,
            self.gaussian_transform.display_mode,
            self.gaussian_transform.sh_deg,
            self.gaussian_transform.no_sh0,
        );

        if !self.measurement_visible_hit_pairs.is_empty() {
            measurement_renderer.update_hit_pairs(
                device,
                &self.measurement_visible_hit_pairs,
                viewer.lock().expect("viewer").camera_buffer(),
            );
        }

        // Preprocesses.
        {
            let viewer = viewer.lock().expect("viewer");

            viewer
                .preprocessor()
                .preprocess(egui_encoder, self.gaussian_count as u32);

            viewer
                .radix_sorter()
                .sort(egui_encoder, viewer.radix_sort_indirect_args_buffer());
        }

        vec![]
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let SceneResource {
            viewer,
            measurement_renderer,
            query_toolset,
            query_texture_overlay,
            query_cursor,
            ..
        } = callback_resources.get().expect("scene resource");

        {
            let viewer = viewer.lock().expect("viewer");

            viewer
                .renderer()
                .render_with_pass(render_pass, viewer.indirect_args_buffer());
        }

        if !self.measurement_visible_hit_pairs.is_empty() {
            measurement_renderer
                .render(render_pass, self.measurement_visible_hit_pairs.len() as u32);
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

impl SceneCallback {
    /// Handle new query.
    ///
    /// This is called when `query_resource` is `None`.
    fn handle_new_query(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        viewer: &mut Arc<Mutex<Viewer>>,
        query_resource: &mut Option<SceneQueryResource>,
        query_toolset: &mut gs::QueryToolset,
        query_cursor: &mut gs::QueryCursor,
    ) {
        if let Query::MeasurementLocateHit { .. } = self.query {
            *query_resource = Some(SceneQueryResource {
                query: self.query.clone(),

                #[cfg(target_arch = "wasm32")]
                stage: SceneQueryStage::Querying,
            });
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

        viewer
            .lock()
            .expect("viewer")
            .update_query(queue, query_pod);

        if let Query::Selection {
            immediate: false, ..
        } = self.query
        {
            query_toolset.render(
                queue,
                encoder,
                viewer.lock().expect("viewer").query_texture(),
            );
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl SceneCallback {
    /// Handle query.
    #[allow(clippy::too_many_arguments)]
    fn query(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        viewer: &mut Arc<Mutex<Viewer>>,
        query_resource: &mut Option<SceneQueryResource>,
        query_toolset: &mut gs::QueryToolset,
        query_cursor: &mut gs::QueryCursor,
    ) {
        if let Some(SceneQueryResource { query }) = query_resource {
            match query {
                Query::MeasurementLocateHit {
                    pod,
                    hit_method,
                    tx,
                } => {
                    #[allow(clippy::await_holding_lock)]
                    let mut results = futures::executor::block_on(async {
                        viewer
                            .lock()
                            .expect("viewer")
                            .download_query_results(device, queue)
                            .await
                            .expect("query results")
                            .into_iter()
                            .map(gs::QueryHitResultPod::from)
                            .collect::<Vec<_>>()
                    });

                    let pos = match hit_method {
                        app::MeasurementHitMethod::MostAlpha => gs::query::hit_pos_by_most_alpha(
                            pod,
                            &mut results,
                            &self.camera,
                            self.viewer_size.as_uvec2(),
                        )
                        .map(|(_, _, pos)| pos)
                        .unwrap_or(Vec3::ZERO),
                        app::MeasurementHitMethod::Closest => gs::query::hit_pos_by_closest(
                            pod,
                            &results,
                            &self.camera,
                            self.viewer_size.as_uvec2(),
                        )
                        .map(|(_, pos)| pos)
                        .unwrap_or(Vec3::ZERO),
                    };

                    if let Err(e) = tx.send(pos) {
                        log::error!("Error occurred while sending hit pos: {}", e.0);
                    }
                }
                Query::None { .. } | Query::Selection { .. } => {}
            }
        }

        *query_resource = None;

        self.handle_new_query(
            queue,
            encoder,
            viewer,
            query_resource,
            query_toolset,
            query_cursor,
        );
    }
}

/// The result from query.
#[cfg(target_arch = "wasm32")]
#[derive(Debug)]
enum SceneCallbackQueryResult {
    /// Next stage.
    Stage(SceneQueryStage),

    /// Next query resource.
    QueryResource(Option<SceneQueryResource>),

    /// Do nothing.
    None,
}

#[cfg(target_arch = "wasm32")]
impl SceneCallback {
    /// Handle query.
    #[allow(clippy::too_many_arguments)]
    fn query(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        viewer: &mut Arc<Mutex<Viewer>>,
        query_resource: &mut Option<SceneQueryResource>,
        query_toolset: &mut gs::QueryToolset,
        query_cursor: &mut gs::QueryCursor,
    ) {
        // Macro rules to handle result of the query.
        macro_rules! handle_query_result {
            ($stage:expr, $query_resource:expr, $query:expr) => {
                match $query {
                    SceneCallbackQueryResult::Stage(next_stage) => {
                        *$stage = next_stage;
                    }
                    SceneCallbackQueryResult::QueryResource(resource) => {
                        *$query_resource = resource;
                    }
                    SceneCallbackQueryResult::None => {}
                }
            };
        }

        // The query results.
        if let Some(SceneQueryResource { query, stage }) = query_resource {
            match stage {
                SceneQueryStage::Querying => handle_query_result!(
                    stage,
                    query_resource,
                    self.querying(device, queue, viewer, query)
                ),
                SceneQueryStage::DownloadingCount { rx } => handle_query_result!(
                    stage,
                    query_resource,
                    self.downloading_count(device, queue, rx, viewer, query)
                ),
                SceneQueryStage::MeasurementLocateHitDownloadingResults { rx, .. } => {
                    handle_query_result!(
                        stage,
                        query_resource,
                        self.measurement_locate_hit_downloading_results(
                            device, queue, rx, viewer, query
                        )
                    )
                }
            };

            viewer
                .lock()
                .expect("viewer")
                .update_query(queue, &gs::QueryPod::none());
        };

        if query_resource.is_none() {
            self.handle_new_query(
                queue,
                encoder,
                viewer,
                query_resource,
                query_toolset,
                query_cursor,
            );
        }
    }

    /// Handle querying.
    fn querying(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        viewer: &mut Arc<Mutex<Viewer>>,
        query: &mut Query,
    ) -> SceneCallbackQueryResult {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Query Result Count Download Encoder"),
        });
        viewer
            .lock()
            .expect("viewer")
            .query_result_count_buffer()
            .prepare_download(&mut encoder);
        queue.submit(Some(encoder.finish()));

        let (tx, rx) = oneshot::channel();

        viewer
            .lock()
            .expect("viewer")
            .query_result_count_buffer()
            .download_buffer()
            .slice(..)
            .map_async(wgpu::MapMode::Read, {
                let viewer = viewer.clone();
                move |_| {
                    let count = bytemuck::pod_read_unaligned(
                        &viewer
                            .lock()
                            .expect("viewer")
                            .query_result_count_buffer()
                            .download_buffer()
                            .slice(..)
                            .get_mapped_range(),
                    );
                    viewer
                        .lock()
                        .expect("viewer")
                        .query_result_count_buffer()
                        .download_buffer()
                        .unmap();

                    if let Err(e) = tx.send(count) {
                        log::error!("Error occurred while sending query result count: {e:?}");
                    }
                }
            });
        device.poll(wgpu::Maintain::Wait);

        match query {
            Query::MeasurementLocateHit { .. } => {
                SceneCallbackQueryResult::Stage(SceneQueryStage::DownloadingCount { rx })
            }
            Query::None { .. } | Query::Selection { .. } => {
                log::error!("Invalid query");
                SceneCallbackQueryResult::QueryResource(None)
            }
        }
    }

    /// Handling the query result count.
    fn downloading_count(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rx: &mut oneshot::Receiver<u32>,
        viewer: &mut Arc<Mutex<Viewer>>,
        query: &mut Query,
    ) -> SceneCallbackQueryResult {
        match rx.try_recv() {
            Ok(0) => SceneCallbackQueryResult::QueryResource(None),
            Ok(count) => {
                match query {
                    Query::MeasurementLocateHit { .. } => {
                        // In WASM, the viewer is not Send nor Sync, but in native, it is.
                        #[allow(clippy::arc_with_non_send_sync)]
                        let download = Arc::new(
                            viewer
                                .lock()
                                .expect("viewer")
                                .query_results_buffer()
                                .create_download_buffer(device, count),
                        );

                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Query Results Download Encoder"),
                            });
                        viewer
                            .lock()
                            .expect("viewer")
                            .query_results_buffer()
                            .prepare_download(&mut encoder, &download);
                        queue.submit(Some(encoder.finish()));

                        let (tx, rx) = oneshot::channel();
                        download.slice(..).map_async(wgpu::MapMode::Read, {
                            let download = download.clone();
                            move |_| {
                                let results = bytemuck::allocation::pod_collect_to_vec(
                                    &download.slice(..).get_mapped_range(),
                                );

                                if let Err(e) = tx.send(results) {
                                    log::error!(
                                        "Error occurred while sending query results: {e:?}"
                                    );
                                }
                            }
                        });
                        device.poll(wgpu::Maintain::Wait);

                        SceneCallbackQueryResult::Stage(
                            SceneQueryStage::MeasurementLocateHitDownloadingResults { rx },
                        )
                    }
                    Query::None { .. } | Query::Selection { .. } => {
                        log::error!("Invalid query");
                        SceneCallbackQueryResult::QueryResource(None)
                    }
                }
            }
            _ => SceneCallbackQueryResult::None,
        }
    }

    fn measurement_locate_hit_downloading_results(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        rx: &mut oneshot::Receiver<Vec<gs::QueryHitResultPod>>,
        _viewer: &mut Arc<Mutex<Viewer>>,
        query: &mut Query,
    ) -> SceneCallbackQueryResult {
        if let Ok(mut results) = rx.try_recv() {
            if let Query::MeasurementLocateHit {
                pod,
                hit_method,
                tx,
                ..
            } = query
            {
                let pos = match hit_method {
                    app::MeasurementHitMethod::MostAlpha => gs::query::hit_pos_by_most_alpha(
                        pod,
                        &mut results,
                        &self.camera,
                        self.viewer_size.as_uvec2(),
                    )
                    .map(|(_, _, pos)| pos),
                    app::MeasurementHitMethod::Closest => gs::query::hit_pos_by_closest(
                        pod,
                        &results,
                        &self.camera,
                        self.viewer_size.as_uvec2(),
                    )
                    .map(|(_, pos)| pos),
                }
                .unwrap_or(Vec3::ZERO);

                if let Err(e) = tx.send(pos) {
                    log::error!("Error occurred while sending hit pos: {e:?}");
                }
            }

            SceneCallbackQueryResult::QueryResource(None)
        } else {
            SceneCallbackQueryResult::None
        }
    }
}
