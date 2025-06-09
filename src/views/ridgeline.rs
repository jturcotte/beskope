// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use crate::wlr_layers::PanelAnchorPosition;
use crate::{FFT_SIZE, RenderWindow, VERTEX_BUFFER_SIZE, ui};

use cgmath::{Matrix4, Rad, SquareMatrix, Vector3, Vector4};
use core::f64;
use num_complex::Complex;
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer, Observer, RingBuffer};
use rustfft::Fft;
use std::collections::HashSet;
use std::{borrow::Cow, sync::Arc};
use wgpu::util::DeviceExt;
use wgpu::{BufferUsages, CommandEncoder, TextureView};

use super::{Vertex, WaveformView, YValue};

const NUM_INSTANCES: usize = 30;
// FIXME: Don't hardcode the sampling rate here
const STRIDE_SIZE: usize = 48_000 / NUM_INSTANCES;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WaveformConfigUniform {
    fill_color: [f32; 4],
    highlight_color: [f32; 4],
    stroke_color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AudioSync {
    y_value_offsets: [u32; 32],
    progress: f32,
    num_instances: f32,
}

pub struct RidgelineWaveformView {
    render_window: RenderWindow,
    anchor_position: PanelAnchorPosition,
    channels: ui::RenderChannels,
    is_left_channel: bool,
    wgpu_queue: Arc<wgpu::Queue>,
    y_value_buffer: wgpu::Buffer,
    audio_sync: AudioSync,
    last_rotate_progress: f64,
    audio_sync_buffer: wgpu::Buffer,
    transform_buffer: wgpu::Buffer,
    waveform_config_buffer: wgpu::Buffer,
    fill_render_pipeline: wgpu::RenderPipeline,
    fill_vertex_buffer: wgpu::Buffer,
    stroke_render_pipeline: wgpu::RenderPipeline,
    stroke_vertex_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    fft_input_ringbuf: HeapRb<f32>,
    y_value_write_offset: usize,
    screen_width: f32,
    screen_height: f32,
    screen_size_dirty: bool,
}

impl RidgelineWaveformView {
    pub fn new(
        device: &wgpu::Device,
        queue: &Arc<wgpu::Queue>,
        swapchain_format: wgpu::TextureFormat,
        render_window: RenderWindow,
        anchor_position: PanelAnchorPosition,
        channels: ui::RenderChannels,
        is_left_channel: bool,
    ) -> RidgelineWaveformView {
        // Load the shaders from disk
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("ridgeline.wgsl"))),
        });

        // Sit the waveform at the bottom boundary of the -1..1, the shader will adjust it.
        let fill_vertices: Vec<Vertex> = (0..STRIDE_SIZE)
            .flat_map(|i| {
                let x = i as f32 / (STRIDE_SIZE - 1) as f32 * 2.0 - 1.0;
                // Point the last vertex to the last audio sample by default.
                let waveform_index = (VERTEX_BUFFER_SIZE - STRIDE_SIZE + i) as u32;
                vec![
                    Vertex {
                        position: [x, -1.0],
                        waveform_index,
                        should_offset: 0.0,
                    },
                    Vertex {
                        position: [x, 0.0],
                        waveform_index,
                        should_offset: 1.0,
                    },
                ]
            })
            .collect();
        let stroke_vertices: Vec<Vertex> = (0..STRIDE_SIZE)
            .map(|i| {
                // Point the last vertex to the last audio sample by default.
                let waveform_index = (VERTEX_BUFFER_SIZE - STRIDE_SIZE + i) as u32;
                Vertex {
                    position: [i as f32 / (STRIDE_SIZE - 1) as f32 * 2.0 - 1.0, 0.0],
                    waveform_index,
                    should_offset: 1.0,
                }
            })
            .collect();
        let y_values: Vec<YValue> = vec![YValue { y: 0.0 }; VERTEX_BUFFER_SIZE];

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
        let audio_sync_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Audio Sync Buffer"),
            size: (std::mem::size_of::<u32>() * 32 + 8) as wgpu::BufferAddress,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Must be updated by apply_lazy_config_changes
        let identity: [[f32; 4]; 4] = Matrix4::<f32>::identity().into();
        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transform Buffer"),
            contents: bytemuck::cast_slice(&identity),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let waveform_config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Waveform Config Buffer"),
            contents: bytemuck::cast_slice(&[WaveformConfigUniform {
                fill_color: [1.0, 1.0, 1.0, 1.0],
                highlight_color: [1.0, 1.0, 1.0, 1.0],
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    visibility: wgpu::ShaderStages::VERTEX,
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
                    resource: audio_sync_buffer.as_entire_binding(),
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
                entry_point: Some("vs_fill_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
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
                    entry_point: Some("vs_stroke_main"),
                    buffers: &[Vertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
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
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    // LessEqual is used to avoid z-fighting with the fill pipeline
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        RidgelineWaveformView {
            render_window,
            anchor_position,
            channels,
            is_left_channel,
            wgpu_queue: queue.clone(),
            y_value_buffer,
            audio_sync: AudioSync {
                y_value_offsets: [0; 32],
                progress: 0.0,
                num_instances: NUM_INSTANCES as f32,
            },
            last_rotate_progress: 0.0,
            audio_sync_buffer,
            transform_buffer,
            waveform_config_buffer,
            fill_render_pipeline,
            fill_vertex_buffer,
            stroke_render_pipeline,
            stroke_vertex_buffer,
            bind_group,
            fft_input_ringbuf: HeapRb::<f32>::new(FFT_SIZE),
            y_value_write_offset: 0,
            screen_width: 0.0,
            screen_height: 0.0,
            screen_size_dirty: true,
        }
    }

    fn get_transform_matrix(&self, panel_width: u32, horizon_offset: f32) -> [[f32; 4]; 4] {
        let panel_width = panel_width as f32;
        let screen_width = self.screen_width;
        let screen_height = self.screen_height;

        // This and the window_x/y assume that the surface is on screen edges.
        // Other panels positioned in-between could make the perspective transform incorrect
        // if they are large and this would require using the actual layer surface position
        // on the screen instead of using the anchor.
        let (window_width, window_height, half_screen, horizontal_offset, vertical_offset) =
            match self.anchor_position {
                PanelAnchorPosition::Top => (
                    screen_width,
                    panel_width,
                    screen_height / 2.0,
                    -horizon_offset,
                    0.0,
                ),
                PanelAnchorPosition::Bottom => (
                    screen_width,
                    panel_width,
                    screen_height / 2.0,
                    -horizon_offset,
                    0.0,
                ),
                PanelAnchorPosition::Left => (
                    panel_width,
                    screen_height,
                    screen_width / 2.0,
                    0.0,
                    -horizon_offset,
                ),
                PanelAnchorPosition::Right => (
                    panel_width,
                    screen_height,
                    screen_width / 2.0,
                    0.0,
                    -horizon_offset,
                ),
            };

        // Identity transform is a horizontal waveform scrolling from right to left.
        let rotation = Matrix4::from_angle_z(Rad(-std::f32::consts::FRAC_PI_2));
        let mirror_h = Matrix4::from_nonuniform_scale(-1.0, 1.0, 1.0);
        let mirror_v = Matrix4::from_nonuniform_scale(1.0, -1.0, 1.0);
        let half = Matrix4::from_nonuniform_scale(0.5, 1.0, 1.0);
        let translate_half_left = Matrix4::from_translation(Vector3::new(-1.0, 0.0, 0.0));
        let translate_half_right = Matrix4::from_translation(Vector3::new(1.0, 0.0, 0.0));
        let transform_left_top = match (
            self.render_window,
            self.anchor_position,
            self.channels,
            self.is_left_channel,
        ) {
            (RenderWindow::Primary, PanelAnchorPosition::Top, ui::RenderChannels::Single, _) => {
                Some((mirror_v, 0.0, 0.0))
            }
            (RenderWindow::Primary, PanelAnchorPosition::Top, ui::RenderChannels::Both, true) => {
                Some((mirror_v * half * translate_half_left, 0.0, 0.0))
            }
            (RenderWindow::Primary, PanelAnchorPosition::Top, ui::RenderChannels::Both, false) => {
                Some((mirror_v * half * translate_half_right * mirror_h, 0.0, 0.0))
            }
            (RenderWindow::Primary, PanelAnchorPosition::Bottom, ui::RenderChannels::Single, _) => {
                Some((Matrix4::identity(), 0.0, screen_height - window_height))
            }
            (
                RenderWindow::Primary,
                PanelAnchorPosition::Bottom,
                ui::RenderChannels::Both,
                true,
            ) => Some((
                half * translate_half_left,
                0.0,
                screen_height - window_height,
            )),
            (
                RenderWindow::Primary,
                PanelAnchorPosition::Bottom,
                ui::RenderChannels::Both,
                false,
            ) => Some((
                half * translate_half_right * mirror_h,
                0.0,
                screen_height - window_height,
            )),
            (RenderWindow::Primary, PanelAnchorPosition::Left, _, _) => {
                Some((rotation * mirror_h, 0.0, 0.0))
            }
            (RenderWindow::Secondary, PanelAnchorPosition::Right, _, _) => Some((
                rotation * mirror_v * mirror_h,
                screen_width - window_width,
                0.0,
            )),
            _ => None,
        };

        if let Some((transform_matrix, window_x, window_y)) = transform_left_top {
            let near_z = 0.0;
            let far_z = 1.0;

            let full_top = 1.0 + vertical_offset;
            let full_bottom = -1.0 + vertical_offset;
            let full_right = 1.0 + horizontal_offset;
            let full_left = -1.0 + horizontal_offset;

            // Window bounds in pixels
            let win_left_px = window_x;
            let win_right_px = window_x + window_width;
            let win_top_px = window_y;
            let win_bottom_px = window_y + window_height;

            // Map window pixel bounds to frustum bounds
            let left = full_left + (full_right - full_left) * (win_left_px / screen_width);
            let right = full_left + (full_right - full_left) * (win_right_px / screen_width);
            // Y axis: top is smaller y, bottom is larger y in screen coordinates
            let top = full_top - (full_top - full_bottom) * (win_top_px / screen_height);
            let bottom = full_top - (full_top - full_bottom) * (win_bottom_px / screen_height);

            // cgmath outputs z values in [-1, 1] for the near and far planes, but wgpu expects them in [0, 1].
            const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::from_cols(
                Vector4::new(1.0, 0.0, 0.0, 0.0),
                Vector4::new(0.0, 1.0, 0.0, 0.0),
                Vector4::new(0.0, 0.0, 0.5, 0.0),
                Vector4::new(0.0, 0.0, 0.5, 1.0),
            );

            const HALF_SCENE_EXTENT: f32 = 1.0;
            // Amplitude as reference point if the window takes half the screen
            // Must match the shader
            const DEFAULT_AMPLITUDE: f32 = 0.15;
            let compress_y = panel_width / half_screen;
            let wave_amplitude: f32 = DEFAULT_AMPLITUDE * compress_y;

            let half_scene_ratio_with_content = wave_amplitude * 2.0 / HALF_SCENE_EXTENT;
            let half_screen_pixels_with_content = half_screen * half_scene_ratio_with_content;

            let far_minus_near_z = far_z - near_z;

            // Instead of using a fixed camera position with a FOV, this keeps the near plane fixed
            // the scene but move the camera back to adjust the FOV so that the waveform history
            // fits the portion of the screen that the window is covering.
            //
            // This code calculates how far the camera should be moved along the Z-axis so that the
            // farthest point of the waveform’s history, when projected onto the near plane (the screen),
            // aligns exactly with the edge of the window.
            // The calculation is based on similar triangles formed by the camera, the near plane, and the far plane.
            //
            // Compute the ratio between the depth range (far_minus_near_z) and the visible width
            // of the panel minus the content width.
            let ratio_leftnear_leftfar_rightnear =
                far_minus_near_z / (panel_width - half_screen_pixels_with_content);
            // Then uses this ratio to determine how much to move the camera back, ensuring that the perspective
            // projection causes the farthest part of the waveform to touch the window’s edge.
            let move_camera_z = (half_screen
                - half_screen_pixels_with_content
                - (panel_width - half_screen_pixels_with_content))
                * ratio_leftnear_leftfar_rightnear
                - near_z;

            let move_scene_matrix =
                Matrix4::from_translation(Vector3::new(0.0, 0.0, -move_camera_z));

            let compress_y_matrix =
                // Move the waveform back to the edge
                Matrix4::from_translation(Vector3::new(0.0, -1.0, 0.0))
                // Waveform is centered at 0.0, scale it depending on the window size
                * Matrix4::from_nonuniform_scale(1.0, compress_y, 1.0);

            let perspective_matrix = OPENGL_TO_WGPU_MATRIX
                * cgmath::frustum(
                    left,
                    right,
                    bottom,
                    top,
                    near_z + move_camera_z,
                    far_z + move_camera_z,
                )
                * move_scene_matrix;

            let horizon_translate =
                Matrix4::from_translation(Vector3::new(horizontal_offset, vertical_offset, 0.0));
            // let transform_matrix = horizon_translate * transform_matrix;
            (perspective_matrix * horizon_translate * transform_matrix * compress_y_matrix).into()
        } else {
            // If the transform matrix is not set, return an identity matrix
            Matrix4::<f32>::identity().into()
        }
    }
}

impl WaveformView for RidgelineWaveformView {
    fn render_window(&self) -> RenderWindow {
        self.render_window
    }

    fn set_screen_size(&mut self, screen_width: u32, screen_height: u32) {
        self.screen_width = screen_width as f32;
        self.screen_height = screen_height as f32;
        self.screen_size_dirty = true;
    }

    fn apply_lazy_config_changes(
        &mut self,
        config: &ui::Configuration,
        config_changes: Option<&HashSet<usize>>,
    ) {
        if config_changes.is_none_or(|c| {
            c.contains(&ui::RIDGELINE_FILL_COLOR)
                || c.contains(&ui::RIDGELINE_HIGHLIGHT_COLOR)
                || c.contains(&ui::RIDGELINE_STROKE_COLOR)
        }) {
            let waveform_config = WaveformConfigUniform {
                fill_color: [
                    config.ridgeline.fill_color.red() as f32 / 255.0,
                    config.ridgeline.fill_color.green() as f32 / 255.0,
                    config.ridgeline.fill_color.blue() as f32 / 255.0,
                    config.ridgeline.fill_color.alpha() as f32 / 255.0,
                ],
                highlight_color: [
                    config.ridgeline.highlight_color.red() as f32 / 255.0,
                    config.ridgeline.highlight_color.green() as f32 / 255.0,
                    config.ridgeline.highlight_color.blue() as f32 / 255.0,
                    config.ridgeline.highlight_color.alpha() as f32 / 255.0,
                ],
                stroke_color: [
                    config.ridgeline.stroke_color.red() as f32 / 255.0,
                    config.ridgeline.stroke_color.green() as f32 / 255.0,
                    config.ridgeline.stroke_color.blue() as f32 / 255.0,
                    config.ridgeline.stroke_color.alpha() as f32 / 255.0,
                ],
            };

            self.wgpu_queue.write_buffer(
                &self.waveform_config_buffer,
                0,
                bytemuck::cast_slice(&[waveform_config]),
            );
        }

        if self.screen_size_dirty
            || config_changes.is_none_or(|c| {
                c.contains(&ui::RIDGELINE_WIDTH) || c.contains(&ui::RIDGELINE_HORIZON_OFFSET)
            })
        {
            let transform_matrix =
                self.get_transform_matrix(config.ridgeline.width, config.ridgeline.horizon_offset);
            self.wgpu_queue.write_buffer(
                &self.transform_buffer,
                0,
                bytemuck::cast_slice(&transform_matrix),
            );
            self.screen_size_dirty = false;
        }
    }

    fn render(
        &self,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        depth_texture_view: Option<&TextureView>,
    ) {
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_texture_view.expect("FIXME"),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.fill_render_pipeline);
            render_pass.set_vertex_buffer(0, self.fill_vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..(STRIDE_SIZE * 2) as u32, 0..NUM_INSTANCES as u32);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                // depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                //     view: &depth_texture_view,
                //     depth_ops: Some(wgpu::Operations {
                //         load: wgpu::LoadOp::Load,
                //         store: wgpu::StoreOp::Store,
                //     }),
                //     stencil_ops: None,
                // }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.stroke_render_pipeline);
            render_pass.set_vertex_buffer(0, self.stroke_vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..STRIDE_SIZE as u32, 0..NUM_INSTANCES as u32);
        }
    }

    fn process_audio(
        self: &mut Self,
        timestamp: u32,
        data: &[f32],
        fft: &dyn Fft<f32>,
        fft_inout_buffer: &mut [Complex<f32>],
        fft_scratch: &mut [Complex<f32>],
    ) {
        let audio_sample_skip = if self.is_left_channel { 0 } else { 1 };

        // Keep track of the last FFT size samples.
        self.fft_input_ringbuf
            .push_iter_overwrite(data.iter().skip(audio_sample_skip).step_by(2).copied());

        let phase_samples = if !self.fft_input_ringbuf.is_full() {
            0
        } else {
            // Run an FFT on the accumulated latest FFT length samples as a way to find the peak frequency
            // and align the end of our waveform at the end of the vertex attribute buffer so that the eye
            // isn't totally lost frame over frame.
            let (first, second) = self.fft_input_ringbuf.as_slices();
            fft_inout_buffer
                .iter_mut()
                .zip(first.iter().chain(second.iter()))
                .for_each(|(dst, &y)| *dst = Complex::new(y, 0.));

            fft.process_with_scratch(fft_inout_buffer, fft_scratch);

            // Skipping k=0 makes sense as it doesn't really capture oscillations, also skip frequencies low enough that
            // aligning to them would prevent the waveform from scrolling enough to be noticeable at 60Hz refresh and 44100Hz sampling rates.
            let k_to_skip: usize = (FFT_SIZE as f64 / (44100.0 / 60.0)).ceil() as usize;

            // Find the peak frequency
            let peak_frequency_index = fft_inout_buffer
                .iter()
                .take(fft_inout_buffer.len() / 2)
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
            let phase_shift = fft_inout_buffer[peak_frequency_index].arg();

            phase_to_samples(phase_shift, peak_frequency_index, fft_inout_buffer.len())
        };

        let data_iter = data.iter().skip(audio_sample_skip).step_by(2).copied();
        let y_values: Vec<YValue> = data_iter.map(|sample| YValue { y: sample }).collect();

        // First pass: write to the end of the buffer
        let first_pass_len = {
            let first_pass_len = VERTEX_BUFFER_SIZE - self.y_value_write_offset;
            let first_pass_data = &y_values[..first_pass_len.min(y_values.len())];
            self.wgpu_queue.write_buffer(
                &self.y_value_buffer,
                (self.y_value_write_offset * std::mem::size_of::<YValue>()) as wgpu::BufferAddress,
                bytemuck::cast_slice(first_pass_data),
            );

            first_pass_len
        };

        let aligned_write_offset =
            (VERTEX_BUFFER_SIZE + self.y_value_write_offset + y_values.len() - phase_samples)
                % VERTEX_BUFFER_SIZE;

        fn to_progress(t: u32) -> f64 {
            // t is in milliseconds, convert to seconds
            let t = t as f64 / 1000.0;
            // t is in seconds, convert to the number of sample strides
            t * NUM_INSTANCES as f64
        }

        // Convert the monotonic timestamp from the compositor to a
        let full_progress = to_progress(timestamp);
        // Wrap at 1.0 to get the progress per stride, 0.0 means start position, 1.0 means that
        // it's at the position of the next stride.
        // By using the modulo we keep the progress smooth across strides rotation.
        self.audio_sync.progress = (full_progress % 1.0) as f32;

        // When progress per stride passes 1.0, carry over the animation by moving this stride
        // to the next waveform stride instance in the shader.
        let wrapped = full_progress.floor() > self.last_rotate_progress.floor();
        if wrapped {
            self.audio_sync.y_value_offsets.rotate_right(1);
            self.last_rotate_progress = full_progress;
        }

        // Keep updating the latest stride even between rotations.
        // The front-most stride is animated while the rest are static
        // and move backwards.
        self.audio_sync.y_value_offsets[0] = aligned_write_offset as u32;

        // Update the GPU buffers with the audio sync state
        self.wgpu_queue.write_buffer(
            &self.audio_sync_buffer,
            0,
            bytemuck::cast_slice(&[self.audio_sync]),
        );

        // Update the write offset
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
    }
}
