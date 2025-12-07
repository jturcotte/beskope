// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use crate::ui;
use crate::view::{AudioInputData, FFT_SIZE, RenderSurface, VERTEX_BUFFER_SIZE, ViewTransform};

use cgmath::{Matrix4, SquareMatrix};
use core::f64;
use num_complex::Complex;
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer, Observer, RingBuffer};
use rustfft::{Fft, FftPlanner};
use splines::Interpolation;
use std::collections::HashSet;
use std::{borrow::Cow, sync::Arc};
use tracing::instrument;
use wgpu::util::DeviceExt;
use wgpu::{BufferUsages, CommandEncoder, TextureView};

use super::{Vertex, View};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WaveformConfigUniform {
    fill_color: [f32; 4],
    stroke_color: [f32; 4],
}

pub struct CompressedView {
    render_surface: RenderSurface,
    is_left_channel: bool,
    view_transform: Option<ViewTransform>,
    wgpu_queue: Arc<wgpu::Queue>,
    y_value_buffer: wgpu::Buffer,
    y_value_offset_buffer: wgpu::Buffer,
    transform_buffer: wgpu::Buffer,
    waveform_config_buffer: wgpu::Buffer,
    fill_render_pipeline: wgpu::RenderPipeline,
    fill_vertex_buffer: wgpu::Buffer,
    stroke_render_pipeline: wgpu::RenderPipeline,
    stroke_vertex_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    fft_input_ringbuf: HeapRb<f32>,
    fft: Arc<dyn Fft<f32>>,
    fft_inout_buffer: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    y_value_write_offset: usize,
}

impl CompressedView {
    #[instrument(skip(device, queue))]
    pub fn new(
        device: &wgpu::Device,
        queue: &Arc<wgpu::Queue>,
        swapchain_format: wgpu::TextureFormat,
        render_surface: RenderSurface,
        is_left_channel: bool,
    ) -> CompressedView {
        // Load the shaders from disk
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("compressed.wgsl"))),
        });

        // Initialize the vertex buffers with a linear time curve
        let fill_vertices: Vec<Vertex> = (0..VERTEX_BUFFER_SIZE)
            .flat_map(|i| {
                let x = i as f32 / VERTEX_BUFFER_SIZE as f32 * 2.0 - 1.0;
                vec![
                    Vertex {
                        position: [x, -1.0],
                        waveform_index: i as u32,
                        should_offset: 0.0,
                    },
                    Vertex {
                        position: [x, 0.0],
                        waveform_index: i as u32,
                        should_offset: 1.0,
                    },
                ]
            })
            .collect();
        let stroke_vertices: Vec<Vertex> = (0..VERTEX_BUFFER_SIZE)
            .map(|i| Vertex {
                position: [i as f32 / VERTEX_BUFFER_SIZE as f32 * 2.0 - 1.0, 0.0],
                waveform_index: i as u32,
                should_offset: 1.0,
            })
            .collect();
        let y_values: Vec<f32> = vec![0.0; VERTEX_BUFFER_SIZE];

        let fill_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fill Vertex Buffer"),
            contents: bytemuck::cast_slice(&fill_vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let stroke_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Stroke Vertex Buffer"),
            contents: bytemuck::cast_slice(&stroke_vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let y_value_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Y Value Buffer"),
            contents: bytemuck::cast_slice(&y_values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Create the y_value_offset buffer
        let y_value_offset_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Y Value Offset Buffer"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // let transform_matrix_array: [[f32; 4]; 4] = transform_matrix.into();
        // Must be updated by apply_lazy_config_changes
        let identity: [[f32; 4]; 4] = Matrix4::<f32>::identity().into();
        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform Buffer"),
            // contents: bytemuck::cast_slice(&transform_matrix_array),
            contents: bytemuck::cast_slice(&identity),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let waveform_config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Waveform Config Buffer"),
            contents: bytemuck::cast_slice(&[WaveformConfigUniform {
                fill_color: [1.0, 1.0, 1.0, 1.0],
                stroke_color: [1.0, 1.0, 1.0, 1.0],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create the bind group layout and bind group for the uniform
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: y_value_offset_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y_value_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: transform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: waveform_config_buffer.as_entire_binding(),
                },
            ],
            label: Some("Bind Group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let fill_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_fill_main"),
                compilation_options: Default::default(),
                targets: &[Some(swapchain_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let stroke_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_stroke_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(swapchain_format.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let scratch_len = fft.get_inplace_scratch_len();

        CompressedView {
            render_surface,
            is_left_channel,
            view_transform: None,
            wgpu_queue: queue.clone(),
            y_value_buffer,
            y_value_offset_buffer,
            transform_buffer,
            waveform_config_buffer,
            fill_render_pipeline,
            fill_vertex_buffer,
            stroke_render_pipeline,
            stroke_vertex_buffer,
            bind_group,
            fft_input_ringbuf: HeapRb::<f32>::new(FFT_SIZE),
            fft,
            fft_inout_buffer: vec![Complex::default(); FFT_SIZE],
            fft_scratch: vec![Complex::default(); scratch_len],
            y_value_write_offset: 0,
        }
    }
}

impl View for CompressedView {
    fn render_surface(&self) -> RenderSurface {
        self.render_surface
    }

    #[instrument(skip(self))]
    fn apply_lazy_config_changes(
        &mut self,
        config: &ui::Configuration,
        config_changes: Option<&HashSet<usize>>,
        view_transform_change: Option<&ViewTransform>,
    ) {
        if config_changes.is_none_or(|c| {
            c.contains(&ui::COMPRESSED_FILL_COLOR) || c.contains(&ui::COMPRESSED_STROKE_COLOR)
        }) {
            let waveform_config = WaveformConfigUniform {
                fill_color: [
                    config.compressed.fill_color.red() as f32 / 255.0,
                    config.compressed.fill_color.green() as f32 / 255.0,
                    config.compressed.fill_color.blue() as f32 / 255.0,
                    config.compressed.fill_color.alpha() as f32 / 255.0,
                ],
                stroke_color: [
                    config.compressed.stroke_color.red() as f32 / 255.0,
                    config.compressed.stroke_color.green() as f32 / 255.0,
                    config.compressed.stroke_color.blue() as f32 / 255.0,
                    config.compressed.stroke_color.alpha() as f32 / 255.0,
                ],
            };

            self.wgpu_queue.write_buffer(
                &self.waveform_config_buffer,
                0,
                bytemuck::cast_slice(&[waveform_config]),
            );
        }

        if config_changes.is_none_or(|c| c.contains(&ui::COMPRESSED_TIME_CURVE_CONTROL_POINTS)) {
            let control_points_with_prefix_suffix_iter =
                std::iter::once((0.0, 0.0, Interpolation::Linear))
                    .chain(
                        config
                            .compressed
                            .time_curve
                            .control_points
                            .iter()
                            .map(|cp| {
                                (
                                    (cp.t / 3.0 + 1.0) * (VERTEX_BUFFER_SIZE - 1) as f32,
                                    cp.v,
                                    Interpolation::CatmullRom,
                                )
                            }),
                    )
                    .chain(std::iter::once((
                        (VERTEX_BUFFER_SIZE - 1) as f32,
                        1.0,
                        Interpolation::Linear,
                    )))
                    .chain(std::iter::once((
                        (VERTEX_BUFFER_SIZE) as f32,
                        1.0,
                        Interpolation::Linear,
                    )));

            let new_spline = splines::Spline::from_vec(
                control_points_with_prefix_suffix_iter
                    .map(|(x, y, interpolation)| splines::Key::new(x, y, interpolation))
                    .collect(),
            );

            let fill_vertices = (0..VERTEX_BUFFER_SIZE)
                .flat_map(|i| {
                    let x = new_spline.sample(i as f32).unwrap_or(0.0) * 2.0 - 1.0;
                    vec![
                        Vertex {
                            position: [x, -1.0],
                            waveform_index: i as u32,
                            should_offset: 0.0,
                        },
                        Vertex {
                            position: [x, 0.0],
                            waveform_index: i as u32,
                            should_offset: 1.0,
                        },
                    ]
                })
                .collect::<Vec<Vertex>>();

            let stroke_vertices = (0..VERTEX_BUFFER_SIZE)
                .map(|i| {
                    let x = new_spline.sample(i as f32).unwrap_or(0.0) * 2.0 - 1.0;
                    Vertex {
                        position: [x, 0.0],
                        waveform_index: i as u32,
                        should_offset: 1.0,
                    }
                })
                .collect::<Vec<Vertex>>();

            // Update fill_vertex_buffer
            self.wgpu_queue.write_buffer(
                &self.fill_vertex_buffer,
                0,
                bytemuck::cast_slice(&fill_vertices),
            );

            // Update stroke_vertex_buffer
            self.wgpu_queue.write_buffer(
                &self.stroke_vertex_buffer,
                0,
                bytemuck::cast_slice(&stroke_vertices),
            );
        }

        if view_transform_change.is_some()
            || config_changes.is_none_or(|c| c.contains(&ui::COMPRESSED_WIDTH_RATIO))
        {
            if view_transform_change.is_some() {
                self.view_transform = view_transform_change.cloned();
            }

            if let Some(view_transform) = self.view_transform {
                let window_to_scene_transform = view_transform
                    .get_window_to_scene_transform(config.compressed.width_ratio as f32);
                let transform_matrix: [[f32; 4]; 4] =
                    (window_to_scene_transform * view_transform.transform_matrix).into();

                self.wgpu_queue.write_buffer(
                    &self.transform_buffer,
                    0,
                    bytemuck::cast_slice(&transform_matrix),
                );
            }
        }
    }

    #[instrument(skip(self))]
    fn render(
        &self,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        _depth_texture_view: &TextureView,
        clear_color: Option<wgpu::Color>,
    ) {
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: clear_color
                            .map(wgpu::LoadOp::Clear)
                            .unwrap_or(wgpu::LoadOp::Load),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.fill_render_pipeline);
            render_pass.set_vertex_buffer(0, self.fill_vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..(VERTEX_BUFFER_SIZE * 2) as u32, 0..1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.stroke_render_pipeline);
            render_pass.set_vertex_buffer(0, self.stroke_vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..VERTEX_BUFFER_SIZE as u32, 0..1);
        }
    }

    #[instrument(skip(self, audio_input))]
    fn process_audio(&mut self, _timestamp: u32, audio_input: &AudioInputData) {
        let audio_sample_skip = if self.is_left_channel { 0 } else { 1 };

        // Keep track of the last FFT size samples.
        self.fft_input_ringbuf.push_iter_overwrite(
            audio_input
                .samples
                .iter()
                .skip(audio_sample_skip)
                .step_by(2)
                .copied(),
        );

        let phase_samples = if !self.fft_input_ringbuf.is_full() {
            0
        } else {
            // Run an FFT on the accumulated latest FFT length samples as a way to find the peak frequency
            // and align the end of our waveform at the end of the vertex attribute buffer so that the eye
            // isn't totally lost frame over frame.
            let (first, second) = self.fft_input_ringbuf.as_slices();
            self.fft_inout_buffer
                .iter_mut()
                .zip(first.iter().chain(second.iter()))
                .for_each(|(dst, &y)| *dst = Complex::new(y, 0.));

            self.fft
                .process_with_scratch(&mut self.fft_inout_buffer, &mut self.fft_scratch);

            // Skipping k=0 makes sense as it doesn't really capture oscillations, also skip frequencies low enough that
            // aligning to them would prevent the waveform from scrolling enough to be noticeable at 60Hz refresh and 44100Hz sampling rates.
            let k_to_skip: usize = (FFT_SIZE as f64 / (44100.0 / 60.0)).ceil() as usize;

            // Find the peak frequency
            let peak_frequency_index = self
                .fft_inout_buffer
                .iter()
                .take(self.fft_inout_buffer.len() / 2)
                .enumerate()
                .skip(k_to_skip)
                .max_by(|(_, a), (_, b)| {
                    a.norm()
                        .partial_cmp(&b.norm())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            fn phase_to_samples(phase: f32, k: usize, fft_size: usize) -> usize {
                // When e.g. k=2, the FFT identifies an oscillation that repeats 2 times in the FFT window.
                // To find the phase shift in samples, find where the phase in radians corresponds vs the FFT buffer size.
                ((phase + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * fft_size as f32
                    / k as f32) as usize
            }

            // To be able to perform the inverse FFT, each frequency bin also has a phase.
            // Use this phase to align the waveform to the end of the buffer.
            // This here is the sine phase shift in radians.
            let phase_shift = self.fft_inout_buffer[peak_frequency_index].arg();

            phase_to_samples(
                phase_shift,
                peak_frequency_index,
                self.fft_inout_buffer.len(),
            )
        };

        let data_iter = audio_input
            .samples
            .iter()
            .skip(audio_sample_skip)
            .step_by(2)
            .copied();
        let y_values: Vec<f32> = data_iter.collect();

        // First pass: write to the end of the buffer
        let first_pass_len = {
            let first_pass_len = VERTEX_BUFFER_SIZE - self.y_value_write_offset;
            let first_pass_data = &y_values[..first_pass_len.min(y_values.len())];
            self.wgpu_queue.write_buffer(
                &self.y_value_buffer,
                (self.y_value_write_offset * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
                bytemuck::cast_slice(first_pass_data),
            );

            first_pass_len
        };

        // Update the write offset, subtracting the phase so that the vertex shader aligns the
        // last peak frequency cycle with the end of the waveform.
        let aligned_write_offset =
            (VERTEX_BUFFER_SIZE + self.y_value_write_offset + y_values.len() - phase_samples)
                % VERTEX_BUFFER_SIZE;
        self.y_value_write_offset =
            (self.y_value_write_offset + y_values.len()) % VERTEX_BUFFER_SIZE;

        // Second pass: write to the beginning of the buffer
        if first_pass_len < y_values.len() {
            let second_pass_data = &y_values[first_pass_len..];
            self.wgpu_queue.write_buffer(
                &self.y_value_buffer,
                0,
                bytemuck::cast_slice(second_pass_data),
            );
        }

        // Tell the shader how to read the audio data ring buffer
        self.wgpu_queue.write_buffer(
            &self.y_value_offset_buffer,
            0,
            bytemuck::cast_slice(&[aligned_write_offset as u32]),
        );
    }
}
