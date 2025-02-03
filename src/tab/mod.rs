use egui::ahash::HashMap;
use strum::{EnumCount, EnumIter, IntoEnumIterator};

mod camera;
mod measurement;
mod scene;
mod transform;

use crate::app;
use camera::Camera;
use measurement::Measurement;
use scene::Scene;
use transform::Transform;

/// The type of tab.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    EnumCount,
    EnumIter,
    serde::Deserialize,
    serde::Serialize,
)]
pub enum Type {
    Scene,
    Transform,
    Camera,
    Measurement,
}

impl Type {
    /// Get the title of the tab for the menu.
    pub fn menu_title(&self) -> &'static str {
        match self {
            Self::Scene => "Scene",
            Self::Transform => "Transform",
            Self::Camera => "Camera",
            Self::Measurement => "Measurement",
        }
    }
}

/// The tab manager.
#[derive(serde::Deserialize, serde::Serialize)]
pub struct Manager {
    /// The dock state.
    dock_state: egui_dock::DockState<Type>,

    /// The tab states.
    #[serde(skip)]
    tabs: HashMap<Type, Box<dyn Tab>>,
}

impl Manager {
    /// Create a new tab manager.
    pub fn new() -> Self {
        let dock_state = egui_dock::DockState::new(vec![Type::Scene]);
        let tabs = [
            (
                Type::Scene,
                Box::new(Scene::create(&mut app::State::default())) as Box<dyn Tab>,
            ),
            (
                Type::Transform,
                Box::new(Transform::create(&mut app::State::default())) as Box<dyn Tab>,
            ),
        ]
        .into_iter()
        .collect();

        Self { dock_state, tabs }
    }

    /// The dock area for the tabs.
    pub fn dock_area(
        &mut self,
        ui: &mut egui::Ui,
        frame: &mut eframe::Frame,
        state: &mut app::State,
    ) {
        egui_dock::DockArea::new(&mut self.dock_state)
            .style(egui_dock::Style::from_egui(ui.style().as_ref()))
            .show_inside(
                ui,
                &mut Viewer {
                    tabs: &mut self.tabs,
                    frame,
                    state,
                },
            );
    }

    /// The menu for the tabs.
    pub fn menu(&mut self, ui: &mut egui::Ui) {
        let mut added = Vec::new();
        let mut removed = Vec::new();

        for tab in Type::iter() {
            let curr = self.dock_state.find_tab(&tab);
            let mut enabled = curr.is_some();

            ui.toggle_value(&mut enabled, tab.menu_title());
            if enabled && curr.is_none() {
                added.push(tab);
            } else if !enabled && curr.is_some() {
                removed.push(curr.unwrap());
            }
        }

        if !added.is_empty() || !removed.is_empty() {
            ui.close_menu();
        }
        if !added.is_empty() {
            self.dock_state.add_window(added);
        }
        if !removed.is_empty() {
            for i in removed {
                self.dock_state.remove_tab(i);
            }
        }
    }
}

impl std::fmt::Debug for Manager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Manager")
            .field("dock_state", &self.dock_state)
            .finish()
    }
}

impl Default for Manager {
    fn default() -> Self {
        Self::new()
    }
}

/// The tab trait.
pub trait Tab {
    /// Create a new tab.
    fn create(state: &mut app::State) -> Self
    where
        Self: Sized;

    /// Get the title of the tab viewer.
    fn title(&mut self, frame: &mut eframe::Frame, state: &mut app::State) -> egui::WidgetText;

    /// The user interface for the tab.
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame, state: &mut app::State);
}

/// The tab viewer.
struct Viewer<'a> {
    /// The tab states.
    tabs: &'a mut HashMap<Type, Box<dyn Tab>>,

    /// The current frame.
    frame: &'a mut eframe::Frame,

    /// The state of the application.
    state: &'a mut app::State,
}

impl Viewer<'_> {
    /// Make sure the tab is created.
    fn make_sure_created(&mut self, tab: Type) {
        self.tabs.entry(tab).or_insert_with(|| match tab {
            Type::Scene => Box::new(Scene::create(self.state)) as Box<dyn Tab>,
            Type::Transform => Box::new(Transform::create(self.state)) as Box<dyn Tab>,
            Type::Camera => Box::new(Camera::create(self.state)) as Box<dyn Tab>,
            Type::Measurement => Box::new(Measurement::create(self.state)) as Box<dyn Tab>,
        });
    }
}

impl egui_dock::TabViewer for Viewer<'_> {
    type Tab = Type;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        self.make_sure_created(*tab);
        self.tabs
            .get_mut(tab)
            .expect("tab")
            .title(self.frame, self.state)
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        self.make_sure_created(*tab);
        self.tabs
            .get_mut(tab)
            .expect("tab")
            .ui(ui, self.frame, self.state);
    }
}
