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
use wgpu_3dgs_viewer as gs;

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
        }
    }

    fn title(&mut self, _frame: &mut eframe::Frame, _state: &mut app::State) -> egui::WidgetText {
        "Scene".into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame, state: &mut app::State) {
        let updated_gs = match &mut state.gs {
            app::Loadable::Unloaded(unloaded) => match unloaded.rx.try_recv() {
                Ok(Ok(mut gs)) => {
                    log::debug!("Gaussian splatting loaded");

                    frame
                        .wgpu_render_state()
                        .expect("render state")
                        .renderer
                        .write()
                        .callback_resources
                        .insert(SceneResource::new(
                            frame.wgpu_render_state().expect("render state"),
                            &gs.gaussians,
                        ));

                    match self.loaded(ui, &mut gs) {
                        false => None,
                        true => Some(app::Loadable::loaded(gs)),
                    }
                }
                Ok(Err(err)) => {
                    log::debug!("Error loading Gaussian splatting: {err}");

                    self.empty(ui, unloaded);

                    Some(app::Loadable::error(err))
                }
                _ => {
                    self.empty(ui, unloaded);

                    None
                }
            },
            app::Loadable::Loaded(_) => {
                let loaded = self.loaded(ui, state.gs.as_mut().unwrap());

                if !loaded {
                    Some(app::Loadable::unloaded())
                } else {
                    None
                }
            }
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
    ) {
        ui.vertical_centered(|ui| {
            ui.add_space(ui.available_height() * 0.4);

            ui.label("Drag & Drop");
            ui.label("OR");
            if ui.button("Browse File").clicked() {
                let tx = unloaded.tx.clone();
                let ctx = ui.ctx().clone();
                let task = rfd::AsyncFileDialog::new()
                    .set_title("Open a PLY file")
                    .pick_file();

                util::exec_task(async move {
                    if let Some(file) = task.await {
                        let mut reader = Cursor::new(file.read().await);
                        let gs = app::GaussianSplatting::new(file.file_name(), &mut reader)
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
                        )
                        .map_err(|e| e.to_string()),
                        false => std::fs::read(file.path.as_ref().expect("file path").clone())
                            .map_err(gs::Error::Io)
                            .map_err(|e| e.to_string())
                            .and_then(|data| {
                                app::GaussianSplatting::new(
                                    file.name.clone(),
                                    &mut Cursor::new(data),
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
            ui.label(format!("📦 Loaded: {}", gs.file_name));

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

            let query = self.input.handle(ui, gs, &rect, &response);

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
                    query,
                },
            ));
        });

        loaded
    }
}

/// The input state for [`Scene`].
#[derive(Debug)]
struct SceneInput {
    /// Whether the scene is focused.
    focused: bool,

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

            #[cfg(target_arch = "wasm32")]
            web_event_listener: SceneInputWebEventListener::new(),
        }
    }

    /// Handle the scene input.
    fn handle(
        &mut self,
        ui: &mut egui::Ui,
        gs: &mut app::GaussianSplatting,
        rect: &egui::Rect,
        response: &egui::Response,
    ) -> Query {
        #[cfg(target_arch = "wasm32")]
        let web_result = self.web_event_listener.update();

        if gs.measurement.action.is_some() {
            return self.measure(ui, gs, rect, response);
        }

        self.control(
            ui,
            gs,
            response,
            #[cfg(target_arch = "wasm32")]
            &web_result,
        );

        Query::none()
    }

    /// Handle measurement action.
    fn measure(
        &mut self,
        _ui: &mut egui::Ui,
        gs: &mut app::GaussianSplatting,
        rect: &egui::Rect,
        response: &egui::Response,
    ) -> Query {
        if let Some(app::MeasurementAction::LocateHit {
            hit_pair_index,
            hit_index,
            rx,
            ..
        }) = &gs.measurement.action
        {
            if let Ok(hit) = rx.try_recv() {
                gs.measurement.hit_pairs[*hit_pair_index].hits[*hit_index].pos = hit;
                gs.measurement.action = None;
            }
        }

        match &gs.measurement.action {
            Some(app::MeasurementAction::LocateHit { tx, .. })
                if response.clicked_by(egui::PointerButton::Primary) =>
            {
                let interact_pos = response.interact_pointer_pos().expect("pointer pos");

                if !rect.contains(interact_pos) {
                    return Query::none();
                }

                let pos = (interact_pos - rect.min).to_pos2();
                Query::locate_hit(pos, gs.measurement.hit_method, tx.clone())
            }
            _ => Query::none(),
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

        let rotation = mouse_delta * gs.camera.sensitivity * dt;
        control.yaw_by(-rotation.x);
        control.pitch_by(-rotation.y);
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

/// The query callback resources.
#[derive(Debug, Clone)]
enum Query {
    /// The query is none.
    None { pod: gs::QueryNonePod },

    /// The locate hit query.
    LocateHit {
        /// The query POD.
        pod: gs::QueryHitPod,

        /// The query method.
        hit_method: app::MeasurementHitMethod,

        /// The query result sender.
        tx: mpsc::Sender<Vec3>,
    },
}

impl Query {
    /// Create a none query.
    fn none() -> Self {
        Self::None {
            pod: gs::QueryNonePod::new(),
        }
    }

    /// Create a locate hit query.
    fn locate_hit(
        coords: egui::Pos2,
        hit_method: app::MeasurementHitMethod,
        tx: mpsc::Sender<Vec3>,
    ) -> Self {
        Self::LocateHit {
            pod: gs::QueryHitPod::new(Vec2::from_array(coords.into())),
            hit_method,
            tx,
        }
    }

    /// Get the POD.
    fn pod(&self) -> &gs::QueryPod {
        match self {
            Self::None { pod } => pod.as_query(),
            Self::LocateHit { pod, .. } => pod.as_query(),
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

/// The scene resource.
///
/// This is for the [`SceneCallback`].
#[derive(Debug)]
struct SceneResource {
    /// The viewer.
    viewer: Arc<Mutex<gs::Viewer>>,

    /// The measurement renderer.
    measurement_renderer: renderer::Measurement,

    /// The query resource.
    ///
    /// When the query is not none, all following query will be ignored until the result is
    /// received.
    query_resource: Option<SceneQueryResource>,
}

impl SceneResource {
    /// Create a new scene resource.
    fn new(render_state: &egui_wgpu::RenderState, gaussians: &gs::Gaussians) -> Self {
        log::debug!("Creating viewer");
        // In WASM, the viewer is not Send nor Sync, but in native, it is.
        #[allow(clippy::arc_with_non_send_sync)]
        let viewer = Arc::new(Mutex::new(gs::Viewer::new(
            render_state.device.as_ref(),
            render_state.target_format,
            gaussians,
        )));

        log::debug!("Creating measurement renderer");
        let measurement_renderer = renderer::Measurement::new(
            render_state.device.as_ref(),
            render_state.target_format,
            &viewer.lock().expect("viewer").camera_buffer,
        );

        log::debug!("Creating query resource");
        let query_resource = None;

        log::info!("Scene loaded");

        Self {
            viewer,
            measurement_renderer,
            query_resource,
        }
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
    /// The locate hit query is downloading count.
    LocateHitDownloadingCount {
        /// The receiver.
        rx: oneshot::Receiver<u32>,
    },
    /// The locate hit query is downloading the results.
    LocateHitDownloadingResults {
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
        } = callback_resources.get_mut().expect("scene resource");

        // The query results.
        #[cfg(target_arch = "wasm32")]
        match query_resource {
            Some(SceneQueryResource { query, stage }) => {
                match stage {
                    SceneQueryStage::Querying => {
                        log::debug!("Querying");

                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Query Result Count Download Encoder"),
                            });
                        viewer
                            .lock()
                            .expect("viewer")
                            .query_result_count_buffer
                            .prepare_download(&mut encoder);
                        queue.submit(Some(encoder.finish()));

                        let (tx, rx) = oneshot::channel();

                        viewer
                            .lock()
                            .expect("viewer")
                            .query_result_count_buffer
                            .download_buffer()
                            .slice(..)
                            .map_async(wgpu::MapMode::Read, {
                                let viewer = viewer.clone();
                                move |_| {
                                    let count = bytemuck::pod_read_unaligned(
                                        &viewer
                                            .lock()
                                            .expect("viewer")
                                            .query_result_count_buffer
                                            .download_buffer()
                                            .slice(..)
                                            .get_mapped_range(),
                                    );
                                    viewer
                                        .lock()
                                        .expect("viewer")
                                        .query_result_count_buffer
                                        .download_buffer()
                                        .unmap();

                                    if let Err(e) = tx.send(count) {
                                        log::error!(
                                            "Error occurred while sending query result count: {e:?}"
                                        );
                                    }
                                }
                            });
                        device.poll(wgpu::Maintain::Wait);

                        *stage = SceneQueryStage::LocateHitDownloadingCount { rx };
                    }
                    SceneQueryStage::LocateHitDownloadingCount { rx } => match rx.try_recv() {
                        Ok(0) => {
                            log::debug!("Locate hit query returned 0 hits");
                            *query_resource = None;
                        }
                        Ok(count) => {
                            log::debug!("Locate hit query returned {count} hits");
                            // In WASM, the viewer is not Send nor Sync, but in native, it is.
                            #[allow(clippy::arc_with_non_send_sync)]
                            let download = Arc::new(
                                viewer
                                    .lock()
                                    .expect("viewer")
                                    .query_results_buffer
                                    .create_download_buffer(device, count),
                            );

                            let mut encoder =
                                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("Query Results Download Encoder"),
                                });
                            viewer
                                .lock()
                                .expect("viewer")
                                .query_results_buffer
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

                            *stage = SceneQueryStage::LocateHitDownloadingResults { rx };
                        }
                        _ => {}
                    },
                    SceneQueryStage::LocateHitDownloadingResults { rx, .. } => {
                        if let Ok(mut results) = rx.try_recv() {
                            if let Query::LocateHit {
                                pod,
                                hit_method,
                                tx,
                                ..
                            } = query
                            {
                                let pos = match hit_method {
                                    app::MeasurementHitMethod::MostAlpha => {
                                        gs::query::hit_pos_by_most_alpha(
                                            pod,
                                            &mut results,
                                            &self.camera,
                                            self.viewer_size.as_uvec2(),
                                        )
                                        .map(|(_, _, pos)| pos)
                                    }
                                    app::MeasurementHitMethod::Closest => {
                                        gs::query::hit_pos_by_closest(
                                            pod,
                                            &results,
                                            &self.camera,
                                            self.viewer_size.as_uvec2(),
                                        )
                                        .map(|(_, pos)| pos)
                                    }
                                }
                                .unwrap_or(Vec3::ZERO);

                                if let Err(e) = tx.send(pos) {
                                    log::error!("Error occurred while sending hit pos: {e:?}");
                                }
                            }

                            *query_resource = None;
                        }
                    }
                };

                viewer
                    .lock()
                    .expect("viewer")
                    .update_query(queue, &gs::QueryPod::none());
            }
            query_resource => {
                if self.query.pod().query_type() != gs::QueryType::None {
                    *query_resource = Some(SceneQueryResource {
                        query: self.query.clone(),
                        stage: SceneQueryStage::Querying,
                    });
                }

                viewer
                    .lock()
                    .expect("viewer")
                    .update_query(queue, self.query.pod());
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(SceneQueryResource {
                query:
                    Query::LocateHit {
                        pod,
                        hit_method,
                        tx,
                    },
            }) = query_resource
            {
                log::debug!("Querying");

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

            *query_resource = None;

            if self.query.pod().query_type() != gs::QueryType::None {
                *query_resource = Some(SceneQueryResource {
                    query: self.query.clone(),
                });
            }

            viewer
                .lock()
                .expect("viewer")
                .update_query(queue, self.query.pod());
        }

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
        );

        if !self.measurement_visible_hit_pairs.is_empty() {
            measurement_renderer.update_hit_pairs(
                device,
                &self.measurement_visible_hit_pairs,
                &viewer.lock().expect("viewer").camera_buffer,
            );
        }

        // Preprocesses.
        {
            let viewer = viewer.lock().expect("viewer");

            viewer
                .preprocessor
                .preprocess(egui_encoder, self.gaussian_count as u32);

            viewer
                .radix_sorter
                .sort(egui_encoder, &viewer.radix_sort_indirect_args_buffer);
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
            ..
        } = callback_resources.get().expect("scene resource");

        {
            let viewer = viewer.lock().expect("viewer");

            viewer
                .renderer
                .render_with_pass(render_pass, &viewer.indirect_args_buffer);
        }

        if !self.measurement_visible_hit_pairs.is_empty() {
            measurement_renderer
                .render(render_pass, self.measurement_visible_hit_pairs.len() as u32);
        }
    }
}
