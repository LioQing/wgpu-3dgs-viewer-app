use std::{
    io::{BufRead, Cursor},
    ops::Range,
    sync::mpsc::{Receiver, Sender},
};

use glam::*;
use strum::{EnumCount, EnumIter};
use wgpu_3dgs_viewer as gs;

use crate::{tab, util};

/// The main application.
#[derive(Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct App {
    /// The tab manager.
    tab_manager: tab::Manager,

    /// The state of the application.
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
            .expect("window")
            .document()
            .expect("document")
    }

    /// Get the canvas.
    ///
    /// This is only available on the web.
    #[cfg(target_arch = "wasm32")]
    pub fn get_canvas() -> web_sys::HtmlCanvasElement {
        use eframe::wasm_bindgen::JsCast as _;

        Self::get_document()
            .get_element_by_id("the_canvas_id")
            .expect("the_canvas_id")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("the_canvas_id to be a HtmlCanvasElement")
    }

    /// Create the menu bar.
    fn menu_bar(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open model").clicked() {
                    self.state.gs = Loadable::unloaded();
                    let Loadable::Unloaded(unloaded) = &mut self.state.gs else {
                        unreachable!()
                    };

                    let tx = unloaded.tx.clone();
                    let ctx = ui.ctx().clone();
                    let compressions = self.state.compressions.clone();
                    let task = rfd::AsyncFileDialog::new()
                        .set_title("Open a PLY file")
                        .pick_file();

                    util::exec_task(async move {
                        if let Some(file) = task.await {
                            let mut reader = Cursor::new(file.read().await);
                            let gs = GaussianSplatting::new(
                                file.file_name(),
                                &compressions,
                                &mut reader,
                            )
                            .map_err(|e| e.to_string());

                            tx.send(gs).expect("send gs");
                            ctx.request_repaint();
                        }
                    });

                    ui.close_menu();
                }

                if ui
                    .add_enabled(self.state.gs.is_loaded(), egui::Button::new("Close model"))
                    .clicked()
                {
                    self.state.gs = Loadable::unloaded();
                    ui.close_menu();
                }

                ui.separator();

                ui.menu_button("Compression Settings", |ui| {
                    macro_rules! value {
                        ($ui: expr, $value: expr, $label: expr, $display: expr) => {
                            if $ui.selectable_label($value == $label, $display).clicked() {
                                $value = $label;
                            }
                        };
                    }

                    ui.menu_button("Spherical Harmonics", |ui| {
                        value!(
                            ui,
                            self.state.compressions.sh,
                            ShCompression::Single,
                            "Single Precision"
                        );
                        value!(
                            ui,
                            self.state.compressions.sh,
                            ShCompression::Half,
                            "Half Precision"
                        );
                        value!(
                            ui,
                            self.state.compressions.sh,
                            ShCompression::MinMaxNorm,
                            "Min-Max Normalization"
                        );
                        value!(
                            ui,
                            self.state.compressions.sh,
                            ShCompression::Remove,
                            "Remove"
                        );
                    });

                    ui.menu_button("Covariance 3D", |ui| {
                        value!(
                            ui,
                            self.state.compressions.cov3d,
                            Cov3dCompression::Single,
                            "Single Precision"
                        );
                        value!(
                            ui,
                            self.state.compressions.cov3d,
                            Cov3dCompression::Half,
                            "Half Precision"
                        );
                    });
                });

                if !cfg!(target_arch = "wasm32") {
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
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
#[derive(Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct State {
    /// The Gaussian splatting model, which can be loaded from a file.
    #[serde(skip)]
    pub gs: Loadable<GaussianSplatting, String>,

    /// The compression settings.
    pub compressions: Compressions,
}

/// The compression settings.
#[derive(Debug, Default, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Compressions {
    /// The spherical harmonics compression.
    pub sh: ShCompression,

    /// The covariance 3D compression.
    pub cov3d: Cov3dCompression,
}

/// The spherical harmonics compression settings.
#[derive(Debug, Default, Clone, Copy, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum ShCompression {
    /// No compression
    Single,
    /// Half precision
    Half,
    /// Min-max normalization
    #[default]
    MinMaxNorm,
    /// Remove SH completely
    Remove,
}

/// The covariance 3D compression settings.
#[derive(Debug, Default, Clone, Copy, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Cov3dCompression {
    /// No compression
    Single,
    /// Half precision
    #[default]
    Half,
}

/// An unloaded value.
#[derive(Debug)]
pub struct Unloaded<T, E> {
    pub tx: Sender<Result<T, E>>,
    pub rx: Receiver<Result<T, E>>,
    pub err: Option<E>,
}

/// A loadable value.
#[derive(Debug)]
pub enum Loadable<T, E> {
    Unloaded(Unloaded<T, E>),
    Loaded(T),
}

impl<T, E> Loadable<T, E> {
    /// Create an unloaded instance of the loadable value.
    pub fn unloaded() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self::Unloaded(Unloaded { tx, rx, err: None })
    }

    /// Create an error instance of the loadable value.
    pub fn error(err: E) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self::Unloaded(Unloaded {
            tx,
            rx,
            err: Some(err),
        })
    }

    /// Create a loaded instance of the loadable value.
    pub fn loaded(value: T) -> Self {
        Self::Loaded(value)
    }

    /// Check if the value is loaded.
    pub fn is_loaded(&self) -> bool {
        matches!(self, Self::Loaded(_))
    }

    /// Get the loaded value.
    ///
    /// # Panics
    ///
    /// Panics if the value is not loaded.
    pub fn unwrap(self) -> T {
        match self {
            Self::Loaded(value) => value,
            _ => panic!("value not loaded"),
        }
    }

    /// Converts from `&mut Loadable<T, E>` to `Loadable<&mut T, E>`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not loaded.
    pub fn as_mut(&mut self) -> Loadable<&mut T, E> {
        match self {
            Self::Loaded(value) => Loadable::loaded(value),
            _ => panic!("value not loaded"),
        }
    }
}

impl<T, E> Default for Loadable<T, E> {
    fn default() -> Self {
        Self::unloaded()
    }
}

/// The Gaussian splatting model.
#[derive(Debug)]
pub struct GaussianSplatting {
    /// The file name of the opened Gaussian splatting model.
    pub file_name: String,

    /// The original model size.
    pub model_size: usize,

    /// The compressed model size.
    pub compressed_size: usize,

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
    pub fn new(
        file_name: String,
        compressions: &Compressions,
        ply: &mut impl BufRead,
    ) -> Result<Self, gs::Error> {
        let measurement = Measurement::new();

        let gaussian_transform = GaussianSplattingGaussianTransform::new();

        let model_transform = GaussianSplattingModelTransform::new();

        let gaussians = gs::Gaussians::read_ply(ply)?;

        let camera = Camera::new(&gaussians, &model_transform);

        macro_rules! compressions_case {
            ($sh:ident, $cov3d:ident) => {
                Compressions {
                    sh: ShCompression::$sh,
                    cov3d: Cov3dCompression::$cov3d,
                }
            };
        }

        macro_rules! compressed_size {
            ($sh:ident, $cov3d:ident) => {
                paste::paste! {
                    std::mem::size_of::<gs::[<GaussianPodWithSh $sh Cov3d $cov3d Configs>]>()
                }
            };
        }

        let compressed_size = gaussians.gaussians.len()
            * match compressions {
                compressions_case!(Single, Single) => compressed_size!(Single, Single),
                compressions_case!(Single, Half) => compressed_size!(Single, Half),
                compressions_case!(Half, Single) => compressed_size!(Half, Single),
                compressions_case!(Half, Half) => compressed_size!(Half, Half),
                compressions_case!(MinMaxNorm, Single) => compressed_size!(MinMaxNorm, Single),
                compressions_case!(MinMaxNorm, Half) => compressed_size!(MinMaxNorm, Half),
                compressions_case!(Remove, Single) => compressed_size!(None, Single),
                compressions_case!(Remove, Half) => compressed_size!(None, Half),
            };

        let model_size = gaussians.gaussians.len() * std::mem::size_of::<gs::PlyGaussianPod>();

        log::info!("Gaussian splatting model loaded");

        Ok(Self {
            file_name,
            model_size,
            compressed_size,
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

    /// The spherical harmonics degree.
    pub sh_deg: gs::GaussianShDegree,

    /// Whether the SH0 is disabled.
    pub no_sh0: bool,
}

impl GaussianSplattingGaussianTransform {
    /// Create a new Gaussian splatting Gaussian transform.
    pub const fn new() -> Self {
        Self {
            size: 1.0,
            display_mode: gs::GaussianDisplayMode::Splat,
            sh_deg: gs::GaussianShDegree::new_unchecked(3),
            no_sh0: false,
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
    /// The control.
    pub control: CameraControl,

    /// The movement speed.
    pub speed: f32,

    /// The rotation sensitivity.
    pub sensitivity: f32,
}

impl Camera {
    /// Create a new camera.
    pub fn new(
        gaussians: &gs::Gaussians,
        model_transform: &GaussianSplattingModelTransform,
    ) -> Self {
        let target = gaussians
            .gaussians
            .iter()
            .map(|g| model_transform.quat() * g.pos)
            .sum::<Vec3>()
            / gaussians.gaussians.len() as f32;
        let pos = target
            + Vec3::Z
                * gaussians
                    .gaussians
                    .iter()
                    .map(|g| (model_transform.quat() * g.pos).z - target.z)
                    .fold(f32::INFINITY, |a, b| a.min(b));
        let control = CameraOrbitControl {
            target,
            pos,
            z: 0.1..1e4,
            vertical_fov: 60f32.to_radians(),
        };

        Self {
            control: CameraControl::Orbit(control),
            speed: 1.0,
            sensitivity: 0.3,
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            control: CameraControl::Orbit(CameraOrbitControl {
                target: Vec3::ZERO,
                pos: Vec3::ZERO,
                z: 0.1..1e4,
                vertical_fov: 60f32.to_radians(),
            }),
            speed: 1.0,
            sensitivity: 0.3,
        }
    }
}

/// The orbit camera control.
#[derive(Debug, Clone)]
pub struct CameraOrbitControl {
    /// The target.
    pub target: Vec3,

    /// The position.
    pub pos: Vec3,

    /// The z range of the camera.
    pub z: Range<f32>,

    /// The vertical FOV.
    pub vertical_fov: f32,
}

impl gs::CameraTrait for CameraOrbitControl {
    fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.pos, self.target, Vec3::Y)
    }

    fn projection(&self, aspect_ratio: f32) -> Mat4 {
        Mat4::perspective_rh(self.vertical_fov, aspect_ratio, self.z.start, self.z.end)
    }
}

/// The first person camera control.
pub type CameraFirstPersonControl = gs::Camera;

/// The camera control.
#[derive(Debug, Clone)]
pub enum CameraControl {
    /// The orbit.
    Orbit(CameraOrbitControl),

    /// The first person.
    FirstPerson(CameraFirstPersonControl),
}

impl CameraControl {
    /// Get the position mutably.
    pub fn pos_mut(&mut self) -> &mut Vec3 {
        match self {
            Self::FirstPerson(control) => &mut control.pos,
            Self::Orbit(control) => &mut control.pos,
        }
    }

    /// Convert into first person control.
    pub fn to_first_person(&self) -> CameraFirstPersonControl {
        match self {
            Self::FirstPerson(first_person) => first_person.clone(),
            Self::Orbit(orbit) => {
                let pos = orbit.pos;
                let direction = (orbit.target - pos).normalize();
                let mut control =
                    CameraFirstPersonControl::new(orbit.z.clone(), orbit.vertical_fov);
                control.pos = pos;
                control.yaw = direction.x.atan2(direction.z);
                control.pitch = direction.y.asin();
                control
            }
        }
    }

    /// Convert into orbit control.
    pub fn to_orbit(&self, arm_length: f32) -> CameraOrbitControl {
        match self {
            Self::FirstPerson(first_person) => {
                let pos = first_person.pos;
                let target = pos + first_person.get_forward() * arm_length;
                let z = first_person.z.start..first_person.z.end;
                let vertical_fov = first_person.vertical_fov;
                CameraOrbitControl {
                    target,
                    pos,
                    z,
                    vertical_fov,
                }
            }
            Self::Orbit(orbit) => orbit.clone(),
        }
    }
}

impl gs::CameraTrait for CameraControl {
    fn view(&self) -> Mat4 {
        match self {
            Self::FirstPerson(control) => control.view(),
            Self::Orbit(control) => control.view(),
        }
    }

    fn projection(&self, aspect_ratio: f32) -> Mat4 {
        match self {
            Self::FirstPerson(control) => control.projection(aspect_ratio),
            Self::Orbit(control) => control.projection(aspect_ratio),
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
