use std::{future::Future, io::Cursor, sync::mpsc::Sender};

#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::JsCast;

use eframe::egui_wgpu;
use glam::*;
use wgpu_3dgs_viewer as gs;

use crate::app;

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
        match &state.gs {
            app::Loadable::None { tx, rx, err } => match rx.try_recv() {
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

                    state.gs = match self.loaded(ui, &mut gs) {
                        false => app::Loadable::none(),
                        true => app::Loadable::loaded(gs),
                    };
                }
                Ok(Err(err)) => {
                    log::debug!("Error loading Gaussian splatting: {err}");

                    self.empty(ui, tx, Some(&err));
                    state.gs = app::Loadable::error(err);
                }
                _ => self.empty(ui, tx, err.as_ref()),
            },
            app::Loadable::Loaded(_) => {
                let loaded = match &mut state.gs {
                    app::Loadable::Loaded(gs) => self.loaded(ui, gs),
                    _ => false,
                };

                if !loaded {
                    state.gs = app::Loadable::none();
                }
            }
        }
    }
}

impl Scene {
    /// Create an empty scene tab.
    fn empty(
        &mut self,
        ui: &mut egui::Ui,
        tx: &Sender<Result<app::GaussianSplatting, gs::Error>>,
        err: Option<&gs::Error>,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        fn exec_task(f: impl Future<Output = ()> + Send + 'static) {
            std::thread::spawn(move || futures::executor::block_on(f));
        }

        #[cfg(target_arch = "wasm32")]
        fn exec_task(f: impl Future<Output = ()> + 'static) {
            wasm_bindgen_futures::spawn_local(f);
        }

        ui.vertical_centered(|ui| {
            ui.add_space(ui.available_height() * 0.4);

            ui.label("Drag & Drop");
            ui.label("OR");
            if ui.button("Browse File").clicked() {
                let tx = tx.clone();
                let ctx = ui.ctx().clone();
                let task = rfd::AsyncFileDialog::new().pick_file();

                exec_task(async move {
                    if let Some(file) = task.await {
                        let mut reader = Cursor::new(file.read().await);
                        let gs = app::GaussianSplatting::new(file.file_name(), &mut reader);

                        tx.send(gs).expect("send gs");
                        ctx.request_repaint();
                    }
                });
            }

            ui.label("");
            ui.label("to Open a PLY File 📦");

            if let Some(err) = err {
                ui.label("");
                ui.label(egui::RichText::new(format!("Error: {err}")).color(egui::Color32::RED));
            }

            if let Some(result) = ui.ctx().input(|input| {
                let mut err = None;

                for file in input.raw.dropped_files.iter() {
                    #[cfg(not(target_arch = "wasm32"))]
                    let gs = std::fs::read(file.path.as_ref().expect("file path").clone())
                        .map_err(gs::Error::Io)
                        .and_then(|data| {
                            app::GaussianSplatting::new(file.name.clone(), &mut Cursor::new(data))
                        });

                    #[cfg(target_arch = "wasm32")]
                    let gs = app::GaussianSplatting::new(
                        file.name.clone(),
                        &mut Cursor::new(file.bytes.as_ref().expect("file bytes").to_vec()),
                    );

                    match gs {
                        Ok(gs) => {
                            return Some(Ok(gs));
                        }
                        Err(e) => {
                            err = Some(e);
                        }
                    }
                }

                if let Some(err) = err {
                    return Some(Err(err));
                }

                None
            }) {
                tx.send(result).expect("send gs");
                ui.ctx().request_repaint();
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

            loaded &= !ui.button("🗑 Unload").clicked();

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

            self.input.handle(ui, gs, &response);

            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect,
                SceneCallback {
                    model_transform: gs.model_transform.clone(),
                    gaussian_transform: gs.gaussian_transform.clone(),
                    camera: gs.camera.camera.clone(),
                    viewer_size: Vec2::from_array(rect.size().into()),
                    gaussian_count: gs.gaussians.gaussians.len(),
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
        response: &egui::Response,
    ) {
        let dt = ui.ctx().input(|input| input.unstable_dt);

        #[cfg(target_arch = "wasm32")]
        let web_result = self.web_event_listener.update();

        // Focus
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

        if !self.focused {
            return;
        }

        // Camera movement
        let mut movement = Vec3::ZERO;

        let mut forward = 0.0;
        if ui.ctx().input(|input| input.key_down(egui::Key::W)) {
            forward += 1.0;
        }
        if ui.ctx().input(|input| input.key_down(egui::Key::S)) {
            forward -= 1.0;
        }

        movement += gs.camera.camera.get_forward().with_y(0.0).normalize() * forward;

        let mut right = 0.0;
        if ui.ctx().input(|input| input.key_down(egui::Key::D)) {
            right += 1.0;
        }
        if ui.ctx().input(|input| input.key_down(egui::Key::A)) {
            right -= 1.0;
        }

        movement += gs.camera.camera.get_right().with_y(0.0).normalize() * right;

        movement = movement.normalize_or_zero() * gs.camera.speed.x;

        let mut up = 0.0;
        if ui.ctx().input(|input| input.key_down(egui::Key::Space)) {
            up += 1.0;
        }
        if ui.ctx().input(|input| input.modifiers.shift_only()) {
            up -= 1.0;
        }

        movement.y += up * gs.camera.speed.y;

        gs.camera.camera.pos += movement * dt;

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
        gs.camera.camera.yaw_by(-rotation.x);
        gs.camera.camera.pitch_by(-rotation.y);
    }
}

impl Default for SceneInput {
    fn default() -> Self {
        Self::new()
    }
}

/// The web event listener for [`SceneInput`].
///
/// This is only available on the web.
#[cfg(target_arch = "wasm32")]
struct SceneInputWebEventListener {
    /// The sender for the web events.
    tx: std::sync::mpsc::Sender<SceneInputWebEvent>,

    /// The receiver for the web events.
    rx: std::sync::mpsc::Receiver<SceneInputWebEvent>,

    /// The "mousemove" event listener.
    mousemove_listener: eframe::wasm_bindgen::prelude::Closure<dyn FnMut(web_sys::MouseEvent)>,

    /// The "pointerlockchange" event listener.
    pointerlockchange_listener: eframe::wasm_bindgen::prelude::Closure<dyn FnMut(web_sys::Event)>,
}

#[cfg(target_arch = "wasm32")]
impl SceneInputWebEventListener {
    /// Create a new web event listener.
    fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();

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
    viewer: gs::Viewer,
}

impl SceneResource {
    /// Create a new scene resource.
    fn new(render_state: &egui_wgpu::RenderState, gaussians: &gs::Gaussians) -> Self {
        log::debug!("Creating viewer");
        let viewer = gs::Viewer::new(
            render_state.device.as_ref(),
            render_state.target_format,
            gaussians,
        );

        log::info!("Scene loaded");

        Self { viewer }
    }
}

/// The scene callback.
struct SceneCallback {
    /// The Gaussian splatting Gaussian transform.
    gaussian_transform: app::GaussianSplattingGaussianTransform,

    /// The Gaussian splatting model transform.
    model_transform: app::GaussianSplattingModelTransform,

    /// The camera.
    camera: gs::Camera,

    /// The viewer size.
    viewer_size: Vec2,

    /// The Gaussian count.
    gaussian_count: usize,
}

impl egui_wgpu::CallbackTrait for SceneCallback {
    fn prepare(
        &self,
        _device: &eframe::wgpu::Device,
        queue: &eframe::wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut eframe::wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<eframe::wgpu::CommandBuffer> {
        let SceneResource { viewer } = callback_resources.get_mut().expect("scene resource");

        viewer.update_camera(queue, &self.camera, self.viewer_size.as_uvec2());
        viewer.update_model_transform(
            queue,
            self.model_transform.pos,
            self.model_transform.quat(),
            self.model_transform.scale,
        );
        viewer.update_gaussian_transform(
            queue,
            self.gaussian_transform.size,
            self.gaussian_transform.display_mode,
        );

        viewer
            .preprocessor
            .preprocess(egui_encoder, self.gaussian_count as u32);

        viewer
            .radix_sorter
            .sort(egui_encoder, &viewer.radix_sort_indirect_args_buffer);

        vec![]
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut eframe::wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let SceneResource { viewer } = callback_resources.get().expect("scene resource");

        viewer
            .renderer
            .render_with_pass(render_pass, &viewer.indirect_args_buffer);
    }
}
