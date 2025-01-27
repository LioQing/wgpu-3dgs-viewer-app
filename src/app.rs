use std::{
    io::BufRead,
    sync::mpsc::{Receiver, Sender},
};

use glam::*;
use wgpu_3dgs_viewer as gs;

use crate::tab;

/// The main application.
#[derive(Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct App {
    /// The tab manager.
    tab_manager: tab::Manager,

    /// The state of the application.
    #[serde(skip)]
    state: State,
}

impl App {
    /// Create a main application.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        if let Some(storage) = cc.storage {
            return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        }

        Self::default()
    }

    /// Get the document.
    ///
    /// This is only available on the web.
    #[cfg(target_arch = "wasm32")]
    pub fn get_document() -> web_sys::Document {
        web_sys::window()
            .expect("No window")
            .document()
            .expect("No document")
    }

    /// Get the canvas.
    ///
    /// This is only available on the web.
    #[cfg(target_arch = "wasm32")]
    pub fn get_canvas() -> web_sys::HtmlCanvasElement {
        use eframe::wasm_bindgen::JsCast as _;

        Self::get_document()
            .get_element_by_id("the_canvas_id")
            .expect("Failed to find the_canvas_id")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("the_canvas_id was not a HtmlCanvasElement")
    }

    /// Create the menu bar.
    fn menu_bar(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if !cfg!(target_arch = "wasm32") && ui.button("Quit").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });

            ui.menu_button("View", |ui| self.tab_manager.menu(ui));

            ui.menu_button("About", |ui| self.about(ui));

            ui.separator();

            egui::widgets::global_theme_preference_buttons(ui);

            if cfg!(debug_assertions) {
                ui.separator();
                egui::warn_if_debug_build(ui);
            }
        });
    }

    /// Show the about dialog.
    fn about(&mut self, ui: &mut egui::Ui) {
        ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;

                ui.label("Made with ");
                ui.strong("a sudden burst of motivation");
                ui.label(" by ");
                ui.hyperlink_to("Lio Qing", "https://lioqing.com");
                ui.label(" with ");
                ui.hyperlink_to("wgpu", "https://wgpu.rs");
                ui.label(", ");
                ui.hyperlink_to("egui", "https://github.com/emilk/egui");
                ui.label(" and ");
                ui.hyperlink_to(
                    "eframe",
                    "https://github.com/emilk/egui/tree/master/crates/eframe",
                );
                ui.label(". ");
                ui.hyperlink_to(
                    "[Source Code]",
                    "https://github.com/lioqing/wgpu-3dgs-viewer-app",
                );
            });
        });
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            self.menu_bar(ctx, ui);
        });

        egui::CentralPanel::default()
            .frame(egui::Frame::central_panel(&ctx.style()).inner_margin(0.))
            .show(ctx, |ui| {
                self.tab_manager.dock_area(ui, frame, &mut self.state);
            });

        ctx.request_repaint();
    }
}

/// The state of the main application.
#[derive(Debug, Default)]
pub struct State {
    pub gs: Loadable<GaussianSplatting, gs::Error>,
}

/// A loadable value.
#[derive(Debug)]
pub enum Loadable<T, E> {
    None {
        tx: Sender<Result<T, E>>,
        rx: Receiver<Result<T, E>>,
        err: Option<E>,
    },
    Loaded(T),
}

impl<T, E> Loadable<T, E> {
    /// Create an empty instance of the loadable value.
    pub fn none() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self::None { tx, rx, err: None }
    }

    /// Create an error instance of the loadable value.
    pub fn error(err: E) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self::None {
            tx,
            rx,
            err: Some(err),
        }
    }

    /// Create a loaded instance of the loadable value.
    pub fn loaded(value: T) -> Self {
        Self::Loaded(value)
    }
}

impl<T, E> Default for Loadable<T, E> {
    fn default() -> Self {
        Self::none()
    }
}

/// The Gaussian splatting model.
pub type GaussianSplatting = GaussianSplattingData;

/// The Gaussian splatting model data.
#[derive(Debug)]
pub struct GaussianSplattingData {
    pub file_name: String,
    pub camera: gs::Camera,
    pub gaussians: gs::Gaussians,
    pub transform: GaussianSplattingTransform,
}

impl GaussianSplattingData {
    /// Create a Gaussian splatting model from a PLY file.
    pub fn new(file_name: String, ply: &mut impl BufRead) -> Result<Self, gs::Error> {
        let transform = GaussianSplattingTransform::new();

        let gaussians = gs::Gaussians::read_ply(ply)?;

        let mut camera = gs::Camera::new(1e-4..1e4, 60f32.to_radians());
        camera.pos = gaussians
            .gaussians
            .iter()
            .map(|g| transform.quat() * g.pos)
            .sum::<Vec3>()
            / gaussians.gaussians.len() as f32;
        camera.pos.z += gaussians
            .gaussians
            .iter()
            .map(|g| (transform.quat() * g.pos).z - camera.pos.z)
            .fold(f32::INFINITY, |a, b| a.min(b));

        log::info!("Gaussian splatting model loaded");

        Ok(Self {
            file_name,
            camera,
            gaussians,
            transform,
        })
    }
}

/// The Gaussian splatting model transform.
#[derive(Debug, Clone)]
pub struct GaussianSplattingTransform {
    /// The position.
    pub pos: Vec3,

    /// The Euler rotation.
    pub rot: Vec3,

    /// The scale.
    pub scale: Vec3,
}

impl GaussianSplattingTransform {
    /// Create a new Gaussian splatting model transform.
    pub const fn new() -> Self {
        Self {
            pos: Vec3::ZERO,
            rot: Vec3::new(0.0, 0.0, std::f32::consts::PI),
            scale: Vec3::ONE,
        }
    }

    /// Get the rotation in quaternion.
    pub fn quat(&self) -> Quat {
        Quat::from_euler(EulerRot::ZYX, self.rot.z, self.rot.y, self.rot.x)
    }
}

impl Default for GaussianSplattingTransform {
    fn default() -> Self {
        Self::new()
    }
}
