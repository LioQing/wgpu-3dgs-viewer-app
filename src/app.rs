use std::{
    collections::HashMap,
    io::{BufRead, Cursor},
    ops::Range,
    sync::mpsc::{Receiver, Sender},
};

use glam::*;
use strum::{Display, EnumCount, EnumIter, IntoEnumIterator};
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
                    let task = rfd::AsyncFileDialog::new()
                        .set_title("Open a PLY file")
                        .pick_file();
                    let compressions = self.state.compressions.clone();

                    util::exec_task(async move {
                        if let Some(file) = task.await {
                            let mut reader = Cursor::new(file.read().await);
                            let gs =
                                GaussianSplatting::new(file.file_name(), &mut reader, compressions)
                                    .map_err(|e| e.to_string());

                            tx.send(gs).expect("send gs");
                            ctx.request_repaint();
                        }
                    });

                    ui.close_menu();
                }

                if ui
                    .add_enabled(self.state.gs.is_loaded(), egui::Button::new("Close models"))
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
                        for sh in ShCompression::iter() {
                            value!(ui, self.state.compressions.sh, sh, sh.to_string().as_str());
                        }
                    });

                    ui.menu_button("Covariance 3D", |ui| {
                        for cov3d in Cov3dCompression::iter() {
                            value!(
                                ui,
                                self.state.compressions.cov3d,
                                cov3d,
                                cov3d.to_string().as_str()
                            );
                        }
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
                ui.label(" and ");
                ui.add(
                    egui::Hyperlink::from_label_and_url("[egui]", "https://github.com/emilk/egui")
                        .open_in_new_tab(true),
                );
                ui.label(". ");
            });

            ui.horizontal_wrapped(|ui| {
                ui.add(
                    egui::Hyperlink::from_label_and_url(
                        "[Source Code]",
                        "https://github.com/lioqing/wgpu-3dgs-viewer-app",
                    )
                    .open_in_new_tab(true),
                );
                ui.add(
                    egui::Hyperlink::from_label_and_url(
                        "[Native App]",
                        "https://github.com/LioQing/wgpu-3dgs-viewer-app/releases",
                    )
                    .open_in_new_tab(true),
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

impl Compressions {
    /// Calculate the compressed size.
    pub fn compressed_size(&self, gaussian_count: usize) -> usize {
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

        gaussian_count
            * match self {
                compressions_case!(Single, Single) => compressed_size!(Single, Single),
                compressions_case!(Single, Half) => compressed_size!(Single, Half),
                compressions_case!(Half, Single) => compressed_size!(Half, Single),
                compressions_case!(Half, Half) => compressed_size!(Half, Half),
                compressions_case!(Norm8, Single) => compressed_size!(Norm8, Single),
                compressions_case!(Norm8, Half) => compressed_size!(Norm8, Half),
                compressions_case!(Remove, Single) => compressed_size!(None, Single),
                compressions_case!(Remove, Half) => compressed_size!(None, Half),
            }
    }
}

/// The spherical harmonics compression settings.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, EnumIter, Display, serde::Deserialize, serde::Serialize,
)]
pub enum ShCompression {
    /// No compression
    #[strum(to_string = "Single Precision")]
    Single,
    /// Half precision
    #[strum(to_string = "Half Precision")]
    Half,
    /// 8 bit normalization
    #[default]
    #[strum(to_string = "8-bit Normalization")]
    Norm8,
    /// Remove SH completely
    #[strum(to_string = "Remove")]
    Remove,
}

/// The covariance 3D compression settings.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, EnumIter, Display, serde::Deserialize, serde::Serialize,
)]
pub enum Cov3dCompression {
    /// No compression
    #[strum(to_string = "Single Precision")]
    Single,
    /// Half precision
    #[default]
    #[strum(to_string = "Half Precision")]
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
}

impl<T, E> Default for Loadable<T, E> {
    fn default() -> Self {
        Self::unloaded()
    }
}

/// The scene commands.
///
/// This is for updating expensive scene data, scene will take this and update the resource.
///
/// For cheap data, they are updated in the scene tab by cloning the needed data from state.
#[derive(Debug)]
pub enum SceneCommand {
    /// Add a new model.
    AddModel(Result<GaussianSplattingModel, String>),

    /// Remove a model.
    RemoveModel(String),

    /// Update the measurement hit.
    UpdateMeasurementHit,
}

/// The Gaussian splatting model.
#[derive(Debug)]
pub struct GaussianSplatting {
    /// The camera to view the model.
    pub camera: Camera,

    /// The Gaussian models loaded.
    pub models: HashMap<String, GaussianSplattingModel>,

    /// The sender for scene to handle scene related updates.
    pub scene_tx: Sender<SceneCommand>,

    /// The receiver for scene to handle scene related updates.
    pub scene_rx: Receiver<SceneCommand>,

    /// The currently selected Gaussian model.
    pub selected_model_key: String,

    /// The Gaussian transform.
    pub gaussian_transform: GaussianSplattingGaussianTransform,

    /// The current action.
    pub action: Option<Action>,

    /// The measurement of the Gaussian splatting.
    pub measurement: Measurement,

    /// The selection of the Gaussian splatting.
    pub selection: Selection,

    /// The used compression settings.
    pub compressions: Compressions,
}

impl GaussianSplatting {
    /// Create a Gaussian splatting model from a PLY file.
    pub fn new(
        file_name: String,
        ply: &mut impl BufRead,
        compressions: Compressions,
    ) -> Result<Self, gs::Error> {
        let selection = Selection::new();

        let measurement = Measurement::new();

        let gaussian_transform = GaussianSplattingGaussianTransform::new();

        let model = GaussianSplattingModel::new(file_name, gs::Gaussians::read_ply(ply)?);

        let key = model.file_name.clone();

        let (scene_tx, scene_rx) = std::sync::mpsc::channel();

        let camera = Camera::new_with_model(&model);

        log::info!("Gaussian splatting model loaded");

        Ok(Self {
            camera,
            models: HashMap::from([(key.clone(), model)]),
            scene_tx,
            scene_rx,
            selected_model_key: key,
            gaussian_transform,
            action: None,
            measurement,
            selection,
            compressions,
        })
    }

    /// Get the currently selected model.
    pub fn selected_model(&self) -> &GaussianSplattingModel {
        self.models
            .get(&self.selected_model_key)
            .expect("selected model")
    }
}

/// The action.
#[derive(Debug)]
pub enum Action {
    /// Locating a hit for measurement.
    MeasurementLocateHit {
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

    /// Selecting.
    Selection,
}

/// The Gaussian splatting model.
#[derive(Debug)]
pub struct GaussianSplattingModel {
    /// The file name.
    pub file_name: String,

    /// The Gaussians.
    pub gaussians: gs::Gaussians,

    /// The transform.
    pub transform: GaussianSplattingModelTransform,

    /// The center of the bounding box.
    pub center: Vec3,

    /// Whether the model is visible.
    pub visible: bool,
}

impl GaussianSplattingModel {
    /// Create a new Gaussian splatting model.
    pub fn new(file_name: String, gaussians: gs::Gaussians) -> Self {
        let (min, max) = gaussians
            .gaussians
            .iter()
            .map(|g| g.pos)
            .fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(min, max), pos| {
                (min.min(pos), max.max(pos))
            });

        let center = (min + max) / 2.0;

        Self {
            file_name,
            gaussians,
            transform: GaussianSplattingModelTransform::new(),
            center,
            visible: true,
        }
    }

    /// Get the center in world space.
    pub fn world_center(&self) -> Vec3 {
        self.transform.quat() * (self.center * self.transform.scale) + self.transform.pos
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
    pub fn new() -> Self {
        Self {
            control: CameraControl::Orbit(CameraOrbitControl::new(
                Vec3::ZERO,
                Vec3::ZERO,
                0.1..1e4,
                60f32.to_radians(),
            )),
            speed: 1.0,
            sensitivity: 0.5,
        }
    }

    /// Create a new camera with an initial position based on a [`GaussianSplattingModel`].
    pub fn new_with_model(model: &GaussianSplattingModel) -> Self {
        let target = model
            .gaussians
            .gaussians
            .iter()
            .map(|g| model.transform.quat() * g.pos)
            .sum::<Vec3>()
            / model.gaussians.gaussians.len() as f32;
        let pos = target
            + Vec3::Z
                * model
                    .gaussians
                    .gaussians
                    .iter()
                    .map(|g| (model.transform.quat() * g.pos).z - target.z)
                    .fold(f32::INFINITY, |a, b| a.min(b));
        let control = CameraOrbitControl::new(target, pos, 0.1..1e4, 60f32.to_radians());

        Self {
            control: CameraControl::Orbit(control),
            speed: 1.0,
            sensitivity: 0.5,
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
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

impl CameraOrbitControl {
    /// Create a new camera.
    pub fn new(target: Vec3, pos: Vec3, z: Range<f32>, vertical_fov: f32) -> Self {
        Self {
            target,
            pos,
            z,
            vertical_fov,
        }
    }
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
    /// Get the position.
    pub fn pos(&self) -> Vec3 {
        match self {
            Self::FirstPerson(control) => control.pos,
            Self::Orbit(control) => control.pos,
        }
    }

    /// Get the position mutably.
    pub fn pos_mut(&mut self) -> &mut Vec3 {
        match self {
            Self::FirstPerson(control) => &mut control.pos,
            Self::Orbit(control) => &mut control.pos,
        }
    }

    /// Get the field of view in radian.
    pub fn vertical_fov(&self) -> f32 {
        match self {
            Self::FirstPerson(control) => control.vertical_fov,
            Self::Orbit(control) => control.vertical_fov,
        }
    }

    /// Get the field of view mutably in radian.
    pub fn vertical_fov_mut(&mut self) -> &mut f32 {
        match self {
            Self::FirstPerson(control) => &mut control.vertical_fov,
            Self::Orbit(control) => &mut control.vertical_fov,
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
}

impl Measurement {
    /// Create a new measurement.
    pub fn new() -> Self {
        Self::default()
    }
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

/// The selection.
#[derive(Debug)]
pub struct Selection {
    /// The selection method.
    pub method: SelectionMethod,

    /// The selection operation.
    pub operation: gs::QuerySelectionOp,

    /// Whether the selection is immediate.
    pub immediate: bool,

    /// The brush radius.
    pub brush_radius: u32,

    /// The highlight color.
    pub highlight_color: egui::Color32,

    /// The edit.
    pub edit: Option<SelectionEdit>,
}

impl Selection {
    /// Create a new selection.
    pub fn new() -> Self {
        Self {
            method: SelectionMethod::Rect,
            operation: gs::QuerySelectionOp::Set,
            immediate: false,
            brush_radius: 40,
            highlight_color: egui::Color32::from_rgba_unmultiplied(255, 0, 255, 127),
            edit: None,
        }
    }
}

impl Default for Selection {
    fn default() -> Self {
        Self::new()
    }
}

/// The selection method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectionMethod {
    /// The rectangle selection.
    Rect,

    /// The brush selection.
    Brush,
}

/// The selection color edit.
#[derive(Debug, Clone, Copy)]
pub enum SelectionColorEdit {
    /// HSV.
    Hsv(Vec3),

    /// Override RGB.
    OverrideColor(Vec3),
}

impl SelectionColorEdit {
    /// Create a new selection color edit.
    pub fn new() -> Self {
        Self::Hsv(Vec3::new(0.0, 1.0, 1.0))
    }
}

impl From<SelectionColorEdit> for Vec3 {
    fn from(val: SelectionColorEdit) -> Self {
        match val {
            SelectionColorEdit::Hsv(hsv) => hsv,
            SelectionColorEdit::OverrideColor(rgb) => rgb,
        }
    }
}

impl Default for SelectionColorEdit {
    fn default() -> Self {
        Self::new()
    }
}

/// The selection edit.
#[derive(Debug, Clone)]
pub struct SelectionEdit {
    /// Hidden.
    pub hidden: bool,

    /// The color.
    pub color: SelectionColorEdit,

    /// The contrast.
    pub contrast: f32,

    /// The exposure.
    pub exposure: f32,

    /// The gamma.
    pub gamma: f32,

    /// The alpha.
    pub alpha: f32,
}

impl SelectionEdit {
    /// Create a new selection edit.
    pub fn new() -> Self {
        Self {
            hidden: false,
            color: SelectionColorEdit::new(),
            contrast: 0.0,
            exposure: 0.0,
            gamma: 1.0,
            alpha: 1.0,
        }
    }

    /// To [`gs::GaussianEditPod`].
    pub fn to_pod(&self) -> gs::GaussianEditPod {
        let mut flag = gs::GaussianEditFlag::ENABLED;
        if self.hidden {
            flag |= gs::GaussianEditFlag::HIDDEN;
        }
        if matches!(self.color, SelectionColorEdit::OverrideColor(..)) {
            flag |= gs::GaussianEditFlag::OVERRIDE_COLOR;
        }

        gs::GaussianEditPod::new(
            flag,
            self.color.into(),
            self.contrast,
            self.exposure,
            self.gamma,
            self.alpha,
        )
    }
}

impl Default for SelectionEdit {
    fn default() -> Self {
        Self::new()
    }
}
