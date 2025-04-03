use std::{
    collections::HashMap,
    io::{BufRead, Cursor},
    ops::Range,
    sync::mpsc,
};

use glam::*;
use itertools::Itertools;
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
        #[cfg(target_arch = "wasm32")]
        Self::apple_silicon_crash_warning();

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

    /// Apple silicon crash warning.
    ///
    /// This is only available on macOS.
    #[cfg(target_arch = "wasm32")]
    pub fn apple_silicon_crash_warning() {
        let window = web_sys::window().expect("window");
        log::warn!(
            "TEST {}",
            window
                .navigator()
                .user_agent()
                .unwrap_or_default()
                .to_lowercase()
        );
        if window
            .navigator()
            .user_agent()
            .unwrap_or_default()
            .to_lowercase()
            .contains("mac")
        {
            window
                .alert_with_message(
                    "Apple Silicon has been known to crash when using this app due to bug in wgpu. \
                    Please do not use this app on Apple Silicon.",
                )
                .ok();
        }
    }

    /// Create the menu bar.
    fn menu_bar(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
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
                            let filename = match file.file_name().trim().is_empty() {
                                true => "Unnamed".to_string(),
                                false => file.file_name().trim().to_string(),
                            };
                            let reader = Cursor::new(file.read().await);
                            let gs = GaussianSplatting::new(filename, reader, compressions)
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

                if ui
                    .add_enabled(self.state.gs.is_loaded(), egui::Button::new("Export model"))
                    .clicked()
                {
                    let Loadable::Loaded(gs) = &mut self.state.gs else {
                        unreachable!()
                    };

                    gs.export_modal = Some(ExportModal::new(gs.models.len()));

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

            ui.separator();

            ui.add(
                egui::Hyperlink::from_label_and_url(
                    "[Native App]",
                    "https://github.com/LioQing/wgpu-3dgs-viewer-app/releases",
                )
                .open_in_new_tab(true),
            );
            ui.add(
                egui::Hyperlink::from_label_and_url(
                    "[Source Code]",
                    "https://github.com/lioqing/wgpu-3dgs-viewer-app",
                )
                .open_in_new_tab(true),
            );
            ui.add(
                egui::Hyperlink::from_label_and_url(
                    "[3DGS Models]",
                    "https://drive.google.com/drive/folders/1WXCpR3kshQt2jmOtuCBsHKfzt1IMqey2",
                )
                .open_in_new_tab(true),
            );

            if cfg!(debug_assertions) {
                ui.separator();
                egui::warn_if_debug_build(ui);
            }
        });

        if let Loadable::Loaded(gs) = &mut self.state.gs {
            if let Some(export_modal) = &mut gs.export_modal {
                macro_rules! case {
                    ($sh:ident, $cov3d:ident) => {
                        Compressions {
                            sh: ShCompression::$sh,
                            cov3d: Cov3dCompression::$cov3d,
                        }
                    };
                }

                macro_rules! ui {
                    ($sh:ident, $cov3d:ident) => {
                        paste::paste! {
                            if !export_modal.ui::<
                                gs::[<GaussianPodWithSh $sh Cov3d $cov3d Configs>]
                            >(ui, frame, &gs.models) {
                                gs.export_modal = None;
                            }
                        }
                    };
                }

                match &gs.compressions {
                    case!(Single, Single) => ui!(Single, Single),
                    case!(Single, Half) => ui!(Single, Half),
                    case!(Half, Single) => ui!(Half, Single),
                    case!(Half, Half) => ui!(Half, Half),
                    case!(Norm8, Single) => ui!(Norm8, Single),
                    case!(Norm8, Half) => ui!(Norm8, Half),
                    case!(Remove, Single) => ui!(None, Single),
                    case!(Remove, Half) => ui!(None, Half),
                }
            }
        }
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
        });
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            self.menu_bar(ctx, ui, frame);
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
    pub tx: mpsc::Sender<Result<T, E>>,
    pub rx: mpsc::Receiver<Result<T, E>>,
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
        let (tx, rx) = mpsc::channel();
        Self::Unloaded(Unloaded { tx, rx, err: None })
    }

    /// Create an error instance of the loadable value.
    pub fn error(err: E) -> Self {
        let (tx, rx) = mpsc::channel();
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
pub enum SceneCommand {
    /// Add a new model.
    AddModel {
        file_name: String,
        reader: Box<dyn BufRead + Send>,
    },

    /// Remove a model.
    RemoveModel(String),

    /// Update the measurement hit.
    UpdateMeasurementHit,

    /// Update mask.
    EvaluateMask(Option<GaussianSplattingMaskOp>),
}

impl std::fmt::Debug for SceneCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AddModel { .. } => write!(f, "AddModel"),
            Self::RemoveModel(_) => write!(f, "RemoveModel"),
            Self::UpdateMeasurementHit => write!(f, "UpdateMeasurementHit"),
            Self::EvaluateMask(_) => write!(f, "EvaluateMasking"),
        }
    }
}

/// The Gaussian splatting model.
#[derive(Debug)]
pub struct GaussianSplatting {
    /// The camera to view the model.
    pub camera: Camera,

    /// The Gaussian models loaded.
    pub models: HashMap<String, GaussianSplattingModel>,

    /// The Gaussian model loader receiver.
    pub model_loader: Option<(String, mpsc::Receiver<Result<gs::Gaussian, gs::Error>>)>,

    /// The sender for scene to handle scene related updates.
    pub scene_tx: mpsc::Sender<SceneCommand>,

    /// The receiver for scene to handle scene related updates.
    pub scene_rx: mpsc::Receiver<SceneCommand>,

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

    /// The export modal.
    pub export_modal: Option<ExportModal>,
}

impl GaussianSplatting {
    /// Create a Gaussian splatting model from a PLY file.
    pub fn new(
        file_name: String,
        ply: impl BufRead + Send + 'static,
        compressions: Compressions,
    ) -> Result<Self, gs::Error> {
        let selection = Selection::new();

        let measurement = Measurement::new();

        let gaussian_transform = GaussianSplattingGaussianTransform::new();

        let (count, gaussian_rx) = GaussianSplattingModel::init_load(ply)?;

        let model = GaussianSplattingModel::new(file_name, count);

        let key = model.file_name.clone();

        let (scene_tx, scene_rx) = mpsc::channel();

        let camera = Camera::new();

        log::info!("Gaussian splatting model loaded");

        Ok(Self {
            camera,
            models: HashMap::from([(key.clone(), model)]),
            model_loader: Some((key.clone(), gaussian_rx)),
            scene_tx,
            scene_rx,
            selected_model_key: key,
            gaussian_transform,
            action: None,
            measurement,
            selection,
            compressions,
            export_modal: None,
        })
    }

    /// Get the currently selected model.
    pub fn selected_model(&self) -> &GaussianSplattingModel {
        self.models
            .get(&self.selected_model_key)
            .expect("selected model")
    }
}

/// The download receiver of [`ExportStage::Edits`].
#[derive(Debug)]
pub enum ExportDownloadReceiver<T> {
    /// Waiting for download.
    Downloading(oneshot::Receiver<Vec<T>>),

    /// Downloaded.
    Downloaded(Vec<T>),
}

impl<T> ExportDownloadReceiver<T> {
    /// Create a new downloading receiver.
    pub fn new(rx: oneshot::Receiver<Vec<T>>) -> Self {
        Self::Downloading(rx)
    }

    /// Try to receive the downloaded data.
    pub fn try_recv(&mut self) {
        match self {
            Self::Downloading(rx) => {
                if let Ok(data) = rx.try_recv() {
                    *self = Self::Downloaded(data);
                }
            }
            Self::Downloaded(_) => {}
        }
    }
}

/// The stages of export.
#[derive(Debug)]
pub enum ExportStage {
    /// Waiting for edit and mask downloads.
    Downloads {
        edits: ExportDownloadReceiver<Vec<gs::GaussianEditPod>>,
        masks: ExportDownloadReceiver<Vec<u32>>,
    },

    /// Waiting for save location.
    Save {
        rx: oneshot::Receiver<Option<rfd::FileHandle>>,
        edits: Vec<Vec<gs::GaussianEditPod>>,
        masks: Vec<Vec<u32>>,
    },
}

/// The export modal.
#[derive(Debug)]
pub struct ExportModal {
    /// The export settings.
    pub settings: Vec<ExportSettings>,

    /// The receiver for the edits download.
    pub stage: Option<ExportStage>,
}

impl ExportModal {
    /// Create a new export modal.
    pub fn new(count: usize) -> Self {
        Self {
            settings: vec![ExportSettings::default(); count],
            stage: None,
        }
    }

    /// The ui.
    ///
    /// Returns whether the export modal should be kept alive.
    fn ui<G: gs::GaussianPod>(
        &mut self,
        ui: &mut egui::Ui,
        frame: &mut eframe::Frame,
        models: &HashMap<String, GaussianSplattingModel>,
    ) -> bool {
        let mut alive = true;

        let models_ordered = models
            .iter()
            .sorted_by_key(|(k, _)| (*k).clone())
            .collect::<Vec<_>>();

        egui::Modal::new(egui::Id::new("export_modal")).show(ui.ctx(), |ui| {
            ui.add(egui::Label::new(
                egui::RichText::new("Export model").heading(),
            ));
            ui.separator();

            ui.label("Please confirm the following models to export");
            ui.label("");

            let text_height = egui::TextStyle::Body
                .resolve(ui.style())
                .size
                .max(ui.spacing().interact_size.y);

            let available_height = ui.available_height();

            egui_extras::TableBuilder::new(ui)
                .striped(true)
                .resizable(true)
                .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                .columns(egui_extras::Column::auto(), 4)
                .min_scrolled_height(0.0)
                .max_scroll_height(available_height)
                .header(20.0, |mut header| {
                    header.col(|ui| {
                        let mut all_export = self.settings.iter().all(|s| s.export);
                        if ui.checkbox(&mut all_export, "").clicked() {
                            for s in &mut self.settings {
                                s.export = all_export;
                            }
                        }
                    });
                    header.col(|ui| {
                        ui.strong("File Name");
                    });
                    header.col(|ui| {
                        let mut all_edit = self.settings.iter().all(|s| s.edit);
                        if ui.checkbox(&mut all_edit, "").clicked() {
                            for s in &mut self.settings {
                                s.edit = all_edit;
                            }
                        }
                        ui.strong("Edit");
                    });
                    header.col(|ui| {
                        let mut all_mask = self.settings.iter().all(|s| s.mask);
                        if ui.checkbox(&mut all_mask, "").clicked() {
                            for s in &mut self.settings {
                                s.mask = all_mask;
                            }
                        }
                        ui.strong("Mask");
                    });
                })
                .body(|body| {
                    body.rows(text_height, models.len(), |mut row| {
                        let index = row.index();

                        let setting = &mut self.settings[index];
                        let (_, model) = &models_ordered[index];

                        row.col(|ui| {
                            ui.checkbox(&mut setting.export, "");
                        });

                        row.col(|ui| {
                            ui.label(&model.file_name);
                        });

                        row.col(|ui| {
                            ui.checkbox(&mut setting.edit, "");
                        });

                        row.col(|ui| {
                            ui.checkbox(&mut setting.mask, "");
                        });
                    });
                });
            ui.label("");

            ui.horizontal(|ui| {
                if ui.button("Confirm").clicked() {
                    let (edits_tx, edits_rx) = oneshot::channel();
                    let (masks_tx, masks_rx) = oneshot::channel();
                    self.stage = Some(ExportStage::Downloads {
                        edits: ExportDownloadReceiver::new(edits_rx),
                        masks: ExportDownloadReceiver::new(masks_rx),
                    });

                    let render_state = frame.wgpu_render_state().expect("render state");
                    let renderer = render_state.renderer.read();
                    let tab::scene::SceneResource::<G> { viewer, .. } =
                        renderer.callback_resources.get().expect("scene");
                    let viewer = viewer.lock().expect("viewer");

                    let device = render_state.device.clone();
                    let queue = render_state.queue.clone();
                    let (edit_buffers, mask_buffers): (Vec<_>, Vec<_>) = models_ordered
                        .iter()
                        .map(|(k, _)| {
                            let gaussian_buffers =
                                &viewer.models.get(*k).expect("model").gaussian_buffers;

                            (
                                gaussian_buffers.gaussians_edit_buffer.clone(),
                                gaussian_buffers.mask_buffer.clone(),
                            )
                        })
                        .unzip();

                    // Download edits
                    {
                        let device = device.clone();
                        let queue = queue.clone();
                        util::exec_task(async move {
                            let mut edits = Vec::with_capacity(edit_buffers.len());
                            for buffer in edit_buffers {
                                match buffer.download(&device, &queue).await {
                                    Ok(edit) => edits.push(edit),
                                    Err(e) => {
                                        log::error!("Download edit buffer: {e}");
                                        edits.push(Vec::new());
                                    }
                                }
                            }

                            edits_tx.send(edits).expect("send edits");
                        });
                    }

                    // Download masks
                    util::exec_task(async move {
                        let mut masks = Vec::with_capacity(mask_buffers.len());
                        for buffer in mask_buffers {
                            match buffer.download(&device, &queue).await {
                                Ok(mask) => masks.push(mask),
                                Err(e) => {
                                    log::error!("Download mask buffer: {e}");
                                    masks.push(Vec::new());
                                }
                            }
                        }

                        masks_tx.send(masks).expect("send masks");
                    });
                }

                if ui.button("Cancel").clicked() {
                    alive = false;
                }
            });
        });

        match &self.stage {
            Some(ExportStage::Downloads { .. }) => {
                let Some(ExportStage::Downloads { edits, masks }) = &mut self.stage else {
                    // Variant of stage has been matched
                    unreachable!()
                };

                edits.try_recv();
                masks.try_recv();

                if let (
                    ExportDownloadReceiver::Downloaded(edits),
                    ExportDownloadReceiver::Downloaded(..),
                ) = (edits, masks)
                {
                    let task = rfd::AsyncFileDialog::new()
                        .set_title("Save the exported models")
                        .set_file_name(match edits.len() {
                            1 if models_ordered[0].0.to_lowercase().ends_with(".ply") => {
                                models_ordered[0].0.clone()
                            }
                            1 => format!("{}.ply", models_ordered[0].0),
                            _ => "models.zip".to_string(),
                        })
                        .save_file();

                    let (tx, rx) = oneshot::channel();

                    let Some(ExportStage::Downloads {
                        edits: ExportDownloadReceiver::Downloaded(edits),
                        masks: ExportDownloadReceiver::Downloaded(masks),
                    }) = std::mem::take(&mut self.stage)
                    else {
                        // Variant of stage has been matched
                        unreachable!()
                    };

                    self.stage = Some(ExportStage::Save { rx, edits, masks });

                    util::exec_task(async move {
                        let file = task.await;
                        tx.send(file).expect("send file");
                    });
                }
            }
            Some(ExportStage::Save { rx, edits, masks }) => {
                if let Ok(Some(file)) = rx.try_recv() {
                    let mut cursor = Cursor::new(Vec::new());
                    self.export_models(
                        &mut cursor,
                        models_ordered.iter().map(|(_, m)| *m),
                        edits,
                        masks,
                    )
                    .expect("export models");

                    util::exec_task(async move {
                        if let Err(e) = file.write(cursor.into_inner().as_slice()).await {
                            log::error!("Save file: {e}");
                        }
                    });

                    alive = false;
                }
            }
            _ => {}
        }

        alive
    }

    /// Export the models.
    pub fn export_models<'a, W: std::io::Write + std::io::Seek>(
        &self,
        writer: &mut W,
        models: impl IntoIterator<Item = &'a GaussianSplattingModel>,
        edits: &[Vec<gs::GaussianEditPod>],
        masks: &[Vec<u32>],
    ) -> Result<(), String> {
        if edits.len() == 1 {
            return models
                .into_iter()
                .next()
                .expect("model")
                .gaussians
                .write_ply(
                    writer,
                    self.settings[0].edit.then_some(&edits[0]),
                    self.settings[0].mask.then_some(masks[0].iter().copied()),
                )
                .map_err(|e| e.to_string());
        }

        let mut zip = zip::ZipWriter::new(writer);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .large_file(true);

        itertools::multizip((
            models.into_iter(),
            self.settings.iter(),
            edits.iter(),
            masks.iter(),
        ))
        .filter(|(_, s, _, _)| s.export)
        .try_for_each(|(model, setting, edit, mask)| {
            zip.start_file(model.file_name.clone(), options)
                .map_err(|e| e.to_string())?;

            model
                .gaussians
                .write_ply(
                    &mut zip,
                    setting.edit.then_some(edit),
                    setting.mask.then_some(mask.iter().copied()),
                )
                .map_err(|e| e.to_string())
        })?;

        zip.finish().map_err(|e| e.to_string())?;

        Ok(())
    }
}

/// The export modal settings.
#[derive(Debug, Clone)]
pub struct ExportSettings {
    /// Export or not.
    pub export: bool,

    /// Apply the edit or not.
    pub edit: bool,

    /// Apply the mask or not.
    pub mask: bool,
}

impl ExportSettings {
    /// Create a new export settings.
    pub fn new() -> Self {
        Self {
            export: true,
            edit: true,
            mask: true,
        }
    }
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self::new()
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
        tx: mpsc::Sender<Vec3>,

        /// The receiver to receive the result.
        rx: mpsc::Receiver<Vec3>,
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

    /// The mask.
    pub mask: GaussianSplattingMask,

    /// The center of the bounding box.
    pub center: Vec3,

    /// Whether the model is visible.
    pub visible: bool,
}

impl GaussianSplattingModel {
    /// Create a new Gaussian splatting model.
    pub fn new(file_name: String, count: usize) -> Self {
        let gaussians = gs::Gaussians {
            gaussians: Vec::with_capacity(count),
        };

        Self {
            file_name,
            gaussians,
            transform: GaussianSplattingModelTransform::new(),
            mask: GaussianSplattingMask::new(),
            center: Vec3::ZERO,
            visible: true,
        }
    }

    /// Get the center in world space.
    pub fn world_center(&self) -> Vec3 {
        self.transform.quat() * (self.center * self.transform.scale) + self.transform.pos
    }

    /// Initialize loading a model.
    ///
    /// This starts a task and sends to the returned [`mpsc::Receiver`].
    ///
    /// Returns the number of Gaussians and the receiver.
    pub fn init_load(
        mut ply: impl BufRead + Send + 'static,
    ) -> Result<(usize, mpsc::Receiver<Result<gs::Gaussian, gs::Error>>), gs::Error> {
        let ply_header = gs::Gaussians::read_ply_header(&mut ply)?;
        let count = ply_header.count()?;

        let (tx, rx) = mpsc::channel();

        util::exec_task(async move {
            match gs::Gaussians::read_ply_gaussians(&mut ply, ply_header) {
                Ok(iter) => {
                    #[cfg(not(target_arch = "wasm32"))]
                    for g in iter {
                        if let Err(err) = tx.send(g.map(gs::Gaussian::from)) {
                            log::error!("Send error: {err}");
                            return;
                        }
                    }

                    #[cfg(target_arch = "wasm32")]
                    {
                        let chunks = iter.chunks(1000);
                        for chunk in &chunks {
                            for g in chunk {
                                if let Err(err) = tx.send(g.map(gs::Gaussian::from)) {
                                    log::error!("Send error: {err}");
                                    return;
                                }
                            }

                            gloo_timers::future::TimeoutFuture::new(0).await;
                        }
                    }
                }
                Err(err) => {
                    if let Err(err) = tx.send(Err(err)) {
                        log::error!("Send error: {err}");
                    }
                }
            }
        });

        Ok((count, rx))
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
            rot: Vec3::ZERO,
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
                Vec3::NEG_Z,
                0.1..1e4,
                60f32.to_radians(),
            )),
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

    /// Whether to show unedited.
    pub show_unedited: bool,
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
            show_unedited: false,
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

/// The mask.
#[derive(Debug, Clone)]
pub struct GaussianSplattingMask {
    /// The mask shapes.
    pub shapes: Vec<GaussianSplattingMaskShape>,

    /// The operation shape PODs.
    pub op_shape_pods: Vec<gs::MaskOpShapePod>,

    /// The operations code.
    pub op_code: String,
}

impl GaussianSplattingMask {
    /// Create a new Gaussian splatting mask.
    pub fn new() -> Self {
        Self {
            shapes: Vec::new(),
            op_shape_pods: Vec::new(),
            op_code: String::new(),
        }
    }

    /// Update the PODs.
    pub fn update_pods(&mut self) {
        self.op_shape_pods = self
            .shapes
            .iter()
            .map(|shape| shape.shape.to_mask_op_shape_pod())
            .collect();
    }
}

/// The mask shape.
#[derive(Debug, Clone)]
pub struct GaussianSplattingMaskShape {
    /// The shape.
    pub shape: gs::MaskShape,

    /// The euler rotation.
    pub rot: Vec3,

    /// Whether the shape is visible.
    pub visible: bool,
}

impl GaussianSplattingMaskShape {
    /// Create a new Gaussian splatting mask shape.
    pub fn new() -> Self {
        Self {
            shape: gs::MaskShape::new(gs::MaskShapeKind::Box),
            rot: Vec3::ZERO,
            visible: true,
        }
    }
}

impl Default for GaussianSplattingMaskShape {
    fn default() -> Self {
        Self::new()
    }
}

/// The syntax tree representing the mask operation.
#[derive(Debug, Clone)]
pub enum GaussianSplattingMaskOp {
    /// Union.
    Union(Box<GaussianSplattingMaskOp>, Box<GaussianSplattingMaskOp>),

    /// Intersection.
    Intersection(Box<GaussianSplattingMaskOp>, Box<GaussianSplattingMaskOp>),

    /// Difference.
    Difference(Box<GaussianSplattingMaskOp>, Box<GaussianSplattingMaskOp>),

    /// Symmetric difference.
    SymmetricDifference(Box<GaussianSplattingMaskOp>, Box<GaussianSplattingMaskOp>),

    /// Complement.
    Complement(Box<GaussianSplattingMaskOp>),

    /// Shape.
    Shape(usize),
}

impl GaussianSplattingMaskOp {
    /// Parse from a string.
    pub fn parse(input: &str) -> Result<Option<Self>, String> {
        use nom::{
            Finish, IResult, Parser,
            branch::alt,
            character::complete::{char, digit1, space0},
            combinator::{all_consuming, map, map_res},
            multi::separated_list1,
            sequence::{delimited, preceded},
        };

        // Parse a shape (just a number)
        fn parse_shape(input: &str) -> IResult<&str, GaussianSplattingMaskOp> {
            map(
                map_res(digit1, str::parse::<usize>),
                GaussianSplattingMaskOp::Shape,
            )
            .parse(input)
        }

        // Parse a term in parentheses
        fn parse_paren(input: &str) -> IResult<&str, GaussianSplattingMaskOp> {
            delimited(
                delimited(space0, char('('), space0),
                parse_expr,
                delimited(space0, char(')'), space0),
            )
            .parse(input)
        }

        // Parse a complement expression (!X)
        fn parse_complement(input: &str) -> IResult<&str, GaussianSplattingMaskOp> {
            map(
                preceded(delimited(space0, char('!'), space0), parse_factor),
                |op| GaussianSplattingMaskOp::Complement(Box::new(op)),
            )
            .parse(input)
        }

        // Parse a factor (shape, paren, or complement)
        fn parse_factor(input: &str) -> IResult<&str, GaussianSplattingMaskOp> {
            delimited(
                space0,
                alt((parse_shape, parse_paren, parse_complement)),
                space0,
            )
            .parse(input)
        }

        // Parse symmetric difference with ^ operator
        fn parse_symmetric_difference(input: &str) -> IResult<&str, GaussianSplattingMaskOp> {
            let (input, items) =
                separated_list1(delimited(space0, char('^'), space0), parse_factor).parse(input)?;

            let mut iter = items.into_iter();
            let initial = iter.next().unwrap();
            let result = iter.fold(initial, |acc, val| {
                GaussianSplattingMaskOp::SymmetricDifference(Box::new(acc), Box::new(val))
            });

            Ok((input, result))
        }

        // Parse difference with - operator
        fn parse_difference(input: &str) -> IResult<&str, GaussianSplattingMaskOp> {
            let (input, items) = separated_list1(
                delimited(space0, char('-'), space0),
                parse_symmetric_difference,
            )
            .parse(input)?;

            let mut iter = items.into_iter();
            let initial = iter.next().unwrap();
            let result = iter.fold(initial, |acc, val| {
                GaussianSplattingMaskOp::Difference(Box::new(acc), Box::new(val))
            });

            Ok((input, result))
        }

        // Parse intersection with & operator
        fn parse_intersection(input: &str) -> IResult<&str, GaussianSplattingMaskOp> {
            let (input, items) =
                separated_list1(delimited(space0, char('&'), space0), parse_difference)
                    .parse(input)?;

            let mut iter = items.into_iter();
            let initial = iter.next().unwrap();
            let result = iter.fold(initial, |acc, val| {
                GaussianSplattingMaskOp::Intersection(Box::new(acc), Box::new(val))
            });

            Ok((input, result))
        }

        // Parse union with | operator
        fn parse_union(input: &str) -> IResult<&str, GaussianSplattingMaskOp> {
            let (input, items) =
                separated_list1(delimited(space0, char('|'), space0), parse_intersection)
                    .parse(input)?;

            let mut iter = items.into_iter();
            let initial = iter.next().unwrap();
            let result = iter.fold(initial, |acc, val| {
                GaussianSplattingMaskOp::Union(Box::new(acc), Box::new(val))
            });

            Ok((input, result))
        }

        // Parse the main expression
        fn parse_expr(input: &str) -> IResult<&str, GaussianSplattingMaskOp> {
            parse_union(input)
        }

        if input.trim().is_empty() {
            return Ok(None);
        }

        // Apply the main parser and convert the result
        match all_consuming(parse_expr).parse(input.trim()).finish() {
            Ok((_, op)) => Ok(Some(op)),
            Err(e) => Err(format!("Failed to parse mask operation: {e}")),
        }
    }

    /// Validate shape indices.
    pub fn validate_shapes(&self, shape_count: usize) -> Result<(), usize> {
        match self {
            Self::Union(left, right) => {
                left.validate_shapes(shape_count)?;
                right.validate_shapes(shape_count)
            }
            Self::Intersection(left, right) => {
                left.validate_shapes(shape_count)?;
                right.validate_shapes(shape_count)
            }
            Self::Difference(left, right) => {
                left.validate_shapes(shape_count)?;
                right.validate_shapes(shape_count)
            }
            Self::SymmetricDifference(left, right) => {
                left.validate_shapes(shape_count)?;
                right.validate_shapes(shape_count)
            }
            Self::Complement(op) => op.validate_shapes(shape_count),
            Self::Shape(index) => {
                if *index >= shape_count {
                    Err(*index)
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Create a [`gs::MaskOpTree`] from the operation.
    pub fn to_tree<'a>(&self, shapes: &'a [gs::MaskOpShapePod]) -> gs::MaskOpTree<'a> {
        match self {
            Self::Union(left, right) => gs::MaskOpTree::Union(
                Box::new(left.to_tree(shapes)),
                Box::new(right.to_tree(shapes)),
            ),
            Self::Intersection(left, right) => gs::MaskOpTree::Intersection(
                Box::new(left.to_tree(shapes)),
                Box::new(right.to_tree(shapes)),
            ),
            Self::Difference(left, right) => gs::MaskOpTree::Difference(
                Box::new(left.to_tree(shapes)),
                Box::new(right.to_tree(shapes)),
            ),
            Self::SymmetricDifference(left, right) => gs::MaskOpTree::SymmetricDifference(
                Box::new(left.to_tree(shapes)),
                Box::new(right.to_tree(shapes)),
            ),
            Self::Complement(op) => gs::MaskOpTree::Complement(Box::new(op.to_tree(shapes))),
            Self::Shape(index) => gs::MaskOpTree::Shape(&shapes[*index]),
        }
    }
}
