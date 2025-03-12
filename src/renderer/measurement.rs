use eframe::wgpu::{self, util::DeviceExt};
use glam::*;
use wgpu_3dgs_viewer as gs;

use crate::app;

/// The renderer for measurement.
#[derive(Debug)]
pub struct Measurement {
    hit_pairs_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl Measurement {
    /// Create a new measurement renderer.
    pub fn new(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        camera: &gs::CameraBuffer,
    ) -> Self {
        log::debug!("Creating measurement hit pairs buffer");
        let hit_pairs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Measurement Hit Pairs Buffer"),
            size: std::mem::size_of::<Vec4>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        log::debug!("Creating measurement bind group layout");
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Measurement Bind Group Layout"),
            entries: &[
                // The hit pairs storage buffer.
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // The camera uniform buffer.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        log::debug!("Creating measurement bind group");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Measurement Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                // The hit pairs storage buffer.
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hit_pairs_buffer.as_entire_binding(),
                },
                // The camera uniform buffer.
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera.buffer().as_entire_binding(),
                },
            ],
        });

        log::debug!("Creating measurement pipeline");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Measurement Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Measurement Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shader/measurement.wgsl").into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Measurement Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vert_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("frag_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        log::info!("Measurement renderer created");

        Self {
            hit_pairs_buffer,
            bind_group_layout,
            bind_group,
            pipeline,
        }
    }

    /// Update the hit pairs buffer.
    pub fn update_hit_pairs(
        &mut self,
        device: &wgpu::Device,
        hit_pairs: &[app::MeasurementHitPair],
        camera: &gs::CameraBuffer,
    ) {
        self.hit_pairs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Measurement Hit Pairs Buffer"),
            contents: bytemuck::cast_slice(
                &hit_pairs
                    .iter()
                    .cloned()
                    .map(HitPair::from)
                    .collect::<Vec<_>>(),
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Measurement Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                // The hit pairs storage buffer.
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.hit_pairs_buffer.as_entire_binding(),
                },
                // The camera uniform buffer.
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera.buffer().as_entire_binding(),
                },
            ],
        });
    }

    /// Render the measurement.
    pub fn render(&self, render_pass: &mut wgpu::RenderPass, hit_pair_count: u32) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..6, 0..hit_pair_count);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct HitPair {
    hit_0: Vec3,
    color: U8Vec4,
    hit_1: Vec3,
    line_width: f32,
}

impl From<app::MeasurementHitPair> for HitPair {
    fn from(hit_pair: app::MeasurementHitPair) -> Self {
        Self {
            hit_0: hit_pair.hits[0].pos,
            color: U8Vec4::from_array(hit_pair.color.to_array()),
            hit_1: hit_pair.hits[1].pos,
            line_width: hit_pair.line_width,
        }
    }
}
