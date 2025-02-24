#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;

use eframe::{egui_wgpu, wgpu};

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 300.0])
            .with_min_inner_size([300.0, 220.0])
            .with_icon(
                eframe::icon_data::from_png_bytes(&include_bytes!("../assets/icon-256.png")[..])
                    .expect("Failed to load icon"),
            ),
        wgpu_options: wgpu_configuration(),
        ..Default::default()
    };

    eframe::run_native(
        "3D Gaussian Splatting Viewer",
        native_options,
        Box::new(|cc| Ok(Box::new(wgpu_3dgs_viewer_app::App::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let web_options = eframe::WebOptions {
        wgpu_options: wgpu_configuration(),
        ..Default::default()
    };

    wasm_bindgen_futures::spawn_local(async {
        let start_result = eframe::WebRunner::new()
            .start(
                wgpu_3dgs_viewer_app::App::get_canvas(),
                web_options,
                Box::new(|cc| Ok(Box::new(wgpu_3dgs_viewer_app::App::new(cc)))),
            )
            .await;

        // Remove the loading text and spinner:
        if let Some(loading_text) =
            wgpu_3dgs_viewer_app::App::get_document().get_element_by_id("loading_text")
        {
            match start_result {
                Ok(_) => {
                    loading_text.remove();
                }
                Err(e) => {
                    loading_text.set_inner_html(
                        "<p> The app has crashed. See the developer console for details. </p>",
                    );
                    panic!("Failed to start eframe: {e:?}");
                }
            }
        }
    });
}

fn wgpu_configuration() -> egui_wgpu::WgpuConfiguration {
    egui_wgpu::WgpuConfiguration {
        wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(egui_wgpu::WgpuSetupCreateNew {
            instance_descriptor: wgpu::InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                ..Default::default()
            },
            power_preference: wgpu::PowerPreference::HighPerformance,
            device_descriptor: Arc::new(|adapter| wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_limits: adapter.limits(),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    }
}
