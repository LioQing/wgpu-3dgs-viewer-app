use std::future::Future;

#[cfg(not(target_arch = "wasm32"))]
pub fn exec_task(f: impl Future<Output = ()> + Send + 'static) {
    std::thread::spawn(move || futures::executor::block_on(f));
}

#[cfg(target_arch = "wasm32")]
pub fn exec_task(f: impl Future<Output = ()> + 'static) {
    wasm_bindgen_futures::spawn_local(f);
}
