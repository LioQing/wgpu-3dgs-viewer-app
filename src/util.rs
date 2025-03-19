use std::future::Future;

#[cfg(not(target_arch = "wasm32"))]
/// Execute a task on a background thread.
pub fn exec_task(f: impl Future<Output = ()> + std::marker::Send + 'static) {
    std::thread::spawn(move || futures::executor::block_on(f));
}

#[cfg(target_arch = "wasm32")]
/// Execute a task on a background thread.
pub fn exec_task(f: impl Future<Output = ()> + 'static) {
    wasm_bindgen_futures::spawn_local(f);
}

#[cfg(not(target_arch = "wasm32"))]
/// Execute a blocking task on a background thread.
pub fn exec_blocking_task(f: impl Future<Output = ()> + 'static) {
    futures::executor::block_on(f);
}

#[cfg(target_arch = "wasm32")]
/// Execute a blocking task on a background thread.
///
/// Note: this function does the same thing as [`exec_task`] on the web,
/// because browsers cannot run blocking code.
pub fn exec_blocking_task(f: impl Future<Output = ()> + 'static) {
    wasm_bindgen_futures::spawn_local(f);
}

/// A wrapper that allows the more idiomatic usage pattern: `ui.add(toggle(&mut my_bool))`
/// iOS-style toggle switch.
///
/// ## Example:
/// ``` ignore
/// ui.add(toggle(&mut my_bool));
/// ```
pub fn toggle(on: &mut bool) -> impl egui::Widget + '_ {
    move |ui: &mut egui::Ui| {
        let desired_size = ui.spacing().interact_size.y * egui::vec2(2.0, 1.0);
        let (rect, mut response) = ui.allocate_exact_size(desired_size, egui::Sense::click());
        if response.clicked() {
            *on = !*on;
            response.mark_changed();
        }
        response.widget_info(|| {
            egui::WidgetInfo::selected(egui::WidgetType::Checkbox, ui.is_enabled(), *on, "")
        });

        if ui.is_rect_visible(rect) {
            let how_on = ui.ctx().animate_bool_responsive(response.id, *on);
            let visuals = ui.style().interact_selectable(&response, *on);
            let rect = rect.expand(visuals.expansion);
            let radius = 0.5 * rect.height();
            ui.painter().rect(
                rect,
                radius,
                visuals.bg_fill,
                visuals.bg_stroke,
                egui::StrokeKind::Outside,
            );
            let circle_x = egui::lerp((rect.left() + radius)..=(rect.right() - radius), how_on);
            let center = egui::pos2(circle_x, rect.center().y);
            ui.painter()
                .circle(center, 0.75 * radius, visuals.bg_fill, visuals.fg_stroke);
        }

        response
    }
}

/// Convert a `usize` to a human-readable string representing storage size.
pub fn human_readable_size(size: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * KB;
    const GB: f64 = MB * KB;
    const TB: f64 = GB * KB;
    const PB: f64 = TB * KB;

    let (size, unit) = if size < KB as usize {
        (size as f64, "B")
    } else if size < MB as usize {
        (size as f64 / KB, "KB")
    } else if size < GB as usize {
        (size as f64 / MB, "MB")
    } else if size < TB as usize {
        (size as f64 / GB, "GB")
    } else if size < PB as usize {
        (size as f64 / TB, "TB")
    } else {
        (size as f64 / PB, "PB")
    };

    format!("{:.2} {}", size, unit)
}
