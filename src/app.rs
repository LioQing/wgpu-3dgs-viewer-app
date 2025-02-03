use std::{
    io::BufRead,
    sync::mpsc::{Receiver, Sender},
};

use glam::*;
use strum::{EnumCount, EnumIter};
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
            ui.spacing_mut().item_spacing.y *= 2.0;

            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;

                ui.label("This viewer app is built for ");
                ui.add(
                    egui::Hyperlink::from_label_and_url(
                        "[3D Gaussian Splatting]",
                        "https://en.wikipedia.org/wiki/Gaussian_splatting",
                    )
                    .open_in_new_tab(true),
                );
                ui.label(". It supports the PLY file format from the ");
                ui.add(
                    egui::Hyperlink::from_label_and_url(
                        "[3D Gaussian Splatting for Real-Time Radiance Field Rendering]",
                        "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/",
                    )
                    .open_in_new_tab(true),
                );
                ui.label(" research paper.");
            });

            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;

                ui.label("Made by ");
                ui.hyperlink_to("[Lio Qing]", "https://lioqing.com");
                ui.label(" with ");
                ui.add(
                    egui::Hyperlink::from_label_and_url("[wgpu]", "https://wgpu.rs")
                        .open_in_new_tab(true),
                );
                ui.label(", ");
                ui.add(
                    egui::Hyperlink::from_label_and_url("[egui]", "https://github.com/emilk/egui")
                        .open_in_new_tab(true),
                );
                ui.label(" and ");
                ui.add(
                    egui::Hyperlink::from_label_and_url(
                        "[eframe]",
                        "https://github.com/emilk/egui/tree/master/crates/eframe",
                    )
                    .open_in_new_tab(true),
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
    /// The Gaussian splatting model, which can be loaded from a file.
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
#[derive(Debug)]
pub struct GaussianSplatting {
    /// The file name of the opened Gaussian splatting model.
    pub file_name: String,

    /// The camera to view the model.
    pub camera: Camera,

    /// The Gaussians parsed from the model.
    pub gaussians: gs::Gaussians,

    /// The model transform.
    pub model_transform: GaussianSplattingModelTransform,

    /// The Gaussian transform.
    pub gaussian_transform: GaussianSplattingGaussianTransform,

    /// The measurement of the Gaussian splatting.
    pub measurement: Measurement,
}

impl GaussianSplatting {
    /// Create a Gaussian splatting model from a PLY file.
    pub fn new(file_name: String, ply: &mut impl BufRead) -> Result<Self, gs::Error> {
        let measurement = Measurement::new();

        let gaussian_transform = GaussianSplattingGaussianTransform::new();

        let model_transform = GaussianSplattingModelTransform::new();

        let gaussians = gs::Gaussians::read_ply(ply)?;

        let camera = Camera::new(&gaussians, &model_transform);

        log::info!("Gaussian splatting model loaded");

        Ok(Self {
            file_name,
            camera,
            gaussians,
            model_transform,
            gaussian_transform,
            measurement,
        })
    }
}

/// The Gaussian splatting model transform.
#[derive(Debug, Clone)]
pub struct GaussianSplattingModelTransform {
    /// The position.
    pub pos: Vec3,

    /// The Euler rotation.
    pub rot: Vec3,

    /// The scale.
    pub scale: Vec3,
}

impl GaussianSplattingModelTransform {
    /// Create a new Gaussian splatting model transform.
    pub const fn new() -> Self {
        Self {
            pos: Vec3::ZERO,
            rot: Vec3::new(0.0, 0.0, 180.0),
            scale: Vec3::ONE,
        }
    }

    /// Get the rotation in quaternion.
    pub fn quat(&self) -> Quat {
        Quat::from_euler(
            EulerRot::ZYX,
            self.rot.z.to_radians(),
            self.rot.y.to_radians(),
            self.rot.x.to_radians(),
        )
    }
}

impl Default for GaussianSplattingModelTransform {
    fn default() -> Self {
        Self::new()
    }
}

/// The Gaussian splatting Gaussian transform.
#[derive(Debug, Clone)]
pub struct GaussianSplattingGaussianTransform {
    /// The size.
    pub size: f32,

    /// The display mode.
    pub display_mode: gs::GaussianDisplayMode,
}

impl GaussianSplattingGaussianTransform {
    /// Create a new Gaussian splatting Gaussian transform.
    pub const fn new() -> Self {
        Self {
            size: 1.0,
            display_mode: gs::GaussianDisplayMode::Splat,
        }
    }
}

impl Default for GaussianSplattingGaussianTransform {
    fn default() -> Self {
        Self::new()
    }
}

/// The camera to view the Gaussian splatting.
#[derive(Debug, Clone)]
pub struct Camera {
    /// The actual camera.
    pub camera: gs::Camera,

    /// The movement speed.
    pub speed: Vec2,

    /// The mouse sensitivity.
    pub sensitivity: f32,
}

impl Camera {
    /// Create a new camera.
    pub fn new(
        gaussians: &gs::Gaussians,
        model_transform: &GaussianSplattingModelTransform,
    ) -> Self {
        let mut camera = gs::Camera::new(1e-4..1e4, 60f32.to_radians());
        camera.pos = gaussians
            .gaussians
            .iter()
            .map(|g| model_transform.quat() * g.pos)
            .sum::<Vec3>()
            / gaussians.gaussians.len() as f32;
        camera.pos.z += gaussians
            .gaussians
            .iter()
            .map(|g| (model_transform.quat() * g.pos).z - camera.pos.z)
            .fold(f32::INFINITY, |a, b| a.min(b));

        Self {
            camera,
            speed: Vec2::ONE,
            sensitivity: 0.3,
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            camera: gs::Camera::new(1e-4..1e4, 60f32.to_radians()),
            speed: Vec2::ONE,
            sensitivity: 0.3,
        }
    }
}

/// The measurement of the Gaussian splatting.
#[derive(Debug, Default)]
pub struct Measurement {
    /// The measurement hits.
    pub hit_pairs: Vec<MeasurementHitPair>,

    /// The hit method.
    pub hit_method: MeasurementHitMethod,

    /// The current measurement action.
    pub action: Option<MeasurementAction>,
}

impl Measurement {
    /// Create a new measurement.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the visible hit pairs.
    pub fn visible_hit_pairs(&self) -> impl Iterator<Item = &MeasurementHitPair> + '_ {
        self.hit_pairs.iter().filter(|hit_pair| hit_pair.visible)
    }
}

/// The measurement action.
#[derive(Debug)]
pub enum MeasurementAction {
    /// Locating a hit.
    LocateHit {
        /// The index of the hit pair.
        hit_pair_index: usize,

        /// The index of the hit.
        ///
        /// Must be 0 or 1.
        hit_index: usize,

        /// The sender to send the result.
        tx: Sender<Vec3>,

        /// The receiver to receive the result.
        rx: Receiver<Vec3>,
    },
}

/// The measurement hit method.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, EnumCount, EnumIter)]
pub enum MeasurementHitMethod {
    /// The most alpha hit.
    #[default]
    MostAlpha,

    /// The closest hit.
    Closest,
}

/// The measurement hit pair.
#[derive(Debug, Clone)]
pub struct MeasurementHitPair {
    /// The label.
    pub label: String,

    /// Whether the hit pair is visible.
    pub visible: bool,

    /// The color of the hit pair.
    pub color: egui::Color32,

    /// The line width.
    pub line_width: f32,

    /// The hits.
    pub hits: [MeasurementHit; 2],
}

impl MeasurementHitPair {
    /// Create a new measurement hit pair.
    pub fn new(label: String) -> Self {
        Self {
            label,
            visible: true,
            color: egui::Color32::RED,
            line_width: 1.0,
            hits: [MeasurementHit::default(), MeasurementHit::default()],
        }
    }

    /// Ge the distance between the hits.
    pub fn distance(&self) -> f32 {
        (self.hits[0].pos - self.hits[1].pos).length()
    }
}

/// The measurement hit.
#[derive(Debug, Clone)]
pub struct MeasurementHit {
    /// The position of the hit.
    pub pos: Vec3,
}

impl Default for MeasurementHit {
    fn default() -> Self {
        Self { pos: Vec3::ZERO }
    }
}
