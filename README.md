# [3D Gaussian Splatting Viewer App](https://lioqing.com/wgpu-3dgs-viewer-app/)

...written in Rust using [wgpu](https://wgpu.rs/) and [egui](https://www.egui.rs/).

[![Github Pages](https://github.com/LioQing/wgpu-3dgs-viewer-app/actions/workflows/pages.yml/badge.svg)](https://github.com/LioQing/wgpu-3dgs-viewer-app/actions/workflows/pages.yml) [![CI](https://github.com/LioQing/wgpu-3dgs-viewer-app/actions/workflows/rust.yml/badge.svg)](https://github.com/LioQing/wgpu-3dgs-viewer-app/actions/workflows/rust.yml)

## Getting started

This viewer app is built for [3D Gaussian Splatting](https://en.wikipedia.org/wiki/Gaussian_splatting). It supports the PLY file format from the [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) research paper.

The viewer implementation can be found at [wgpu-3dgs-viewer](https://github.com/LioQing/wgpu-3dgs-viewer). The app uses [wgpu](https://wgpu.rs/), the Rust implementation of [WebGPU](https://en.wikipedia.org/wiki/WebGPU), to render the model, so it supports most of the GPU backends.

> [!NOTE]
>
> To use the web version, you need a browser that supports WebGPU. Please refer to the [WebGPU Implementation Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status) for more information.
>
> You may also run the app natively, see [Native](#native) below.

### Running the app

#### Web

You can run the app on the web by visiting [https://lioqing.com/wgpu-3dgs-viewer-app/](https://lioqing.com/wgpu-3dgs-viewer-app/).

#### Native

You can run the app natively, by downloading from the latest release from the [releases page](https://github.com/LioQing/wgpu-3dgs-viewer-app/releases) depending on your operating system.

### Features

- Load large PLY files at very fast speed.
- Render multiple models at once.
- Edit the model's transform and colors.
- Point and click measurement tool.

### Demo

https://github.com/user-attachments/assets/63275b41-ba1e-4c71-aae0-c43e514eac0c

## Development

### Native

Make sure you are using the latest version of stable rust by running `rustup update`.

`cargo run --release`

On Linux you need to first run:

`sudo apt-get install libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev`

On Fedora Rawhide you need to run:

`dnf install clang clang-devel clang-tools-extra libxkbcommon-devel pkg-config openssl-devel libxcb-devel gtk3-devel atk fontconfig-devel`

### Web locally

We compile to [WASM](https://en.wikipedia.org/wiki/WebAssembly) and use [Trunk](https://trunkrs.dev/) to build for web target.

1. Install the required target with `rustup target add wasm32-unknown-unknown`.
2. Install Trunk with `cargo install --locked trunk`.
3. Run `trunk serve` to build and serve on `http://127.0.0.1:8080`. Trunk will rebuild automatically if you edit the project.
4. Open `http://127.0.0.1:8080/index.html#dev` in a browser. See the warning below.

> [!NOTE]
>
> `assets/sw.js` script will try to cache our app, and loads the cached version when it cannot connect to server allowing your app to work offline (like PWA).
> appending `#dev` to `index.html` will skip this caching, allowing us to load the latest builds during development.
