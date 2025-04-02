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
- Different Gaussian display modes.
- Render multiple models at once.
- Edit the model's transform and colors.
- Mask the model with composite shapes.

### Demo

The following demos are recorded on laptop with:

- CPU: AMD Ryzen 9 4900HS
- GPU: NVIDIA GeForce RTX 2060 with Max-Q Design
- RAM: 32 GB

#### Compression & Loading

Models are compressed and loaded while being streamed from the file at a very fast speed.

https://github.com/user-attachments/assets/9f3ba3de-e7b9-4a86-baae-98574a93a21f

#### Display Mode

Gaussians can be viewed as ellipses or points, with specified spherical harmonic degrees, and at different sizes.

https://github.com/user-attachments/assets/bfc36e2a-0939-4703-af47-3c958556770a

#### Multiple Models

Multiple models can be loaded to be viewed in the same scene.

https://github.com/user-attachments/assets/45e05ed1-ad93-44ea-9213-5218748cd26e

#### Selection & Editing

Gaussians can be selected using various tools and be used for color editing.

https://github.com/user-attachments/assets/5a2ef198-f8a3-43db-94ff-9e393041bf1e

#### Composite Mask

Masks can be used to select Gaussians to view, which can be composed using set operations.

https://github.com/user-attachments/assets/3ec2a95f-291b-4684-9023-94a57d8c3b2a

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
