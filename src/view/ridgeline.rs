// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use crate::ui;
use crate::view::models::ValuesRange;
use crate::view::{AudioInputData, AudioModel, RenderSurface, ViewTransform};

use cgmath::{Matrix4, SquareMatrix, Vector3, Vector4};
use std::collections::HashSet;
use std::{borrow::Cow, sync::Arc};
use tracing::instrument;
use wgpu::util::DeviceExt;
use wgpu::{BufferUsages, CommandEncoder, TextureView};

use super::{Vertex, View};

const NUM_INSTANCES: usize = 30;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WaveformConfigUniform {
    fill_color: [f32; 4],
    highlight_color: [f32; 4],
    stroke_color: [f32; 4],
    apply_highlight_to_front_instance: f32,
    _padding: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AudioSync {
    y_value_offsets: [u32; 32],
    progress: f32,
    num_instances: f32,
}

pub struct RidgelineView<M: AudioModel> {
    render_surface: RenderSurface,
    is_left_channel: bool,
    style: ui::Style,
    view_transform: Option<ViewTransform>,
    wgpu_queue: Arc<wgpu::Queue>,
    y_value_buffer: wgpu::Buffer,
    audio_sync: AudioSync,
    audio_sync_buffer: wgpu::Buffer,
    transform_buffer: wgpu::Buffer,
    waveform_config_buffer: wgpu::Buffer,
    fill_render_pipeline: wgpu::RenderPipeline,
    fill_vertex_buffer: wgpu::Buffer,
    stroke_render_pipeline: wgpu::RenderPipeline,
    stroke_vertex_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    model: M,
    stride_len: usize,
    y_values_buffer_size: usize,
    write_offset: usize,
    virtual_stride_index: f64,
}

impl<M: AudioModel> RidgelineView<M> {
    #[instrument(skip(model))]
    pub fn new(
        device: &wgpu::Device,
        queue: &Arc<wgpu::Queue>,
        swapchain_format: wgpu::TextureFormat,
        render_surface: RenderSurface,
        is_left_channel: bool,
        style: ui::Style,
        model: M,
    ) -> RidgelineView<M> {
        // Load the shaders from disk
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("ridgeline.wgsl"))),
        });

        let stride_len = model.stride_len();

        let fill_vertices: Vec<Vertex> = (0..stride_len)
            .flat_map(|i| {
                let x = i as f32 / (stride_len - 1) as f32 * 2.0 - 1.0;
                let waveform_index = i as u32; // index within stride; shader adds instance offset
                vec![
                    // Base fill vertex.
                    // Extend well into the edge of the screen so that the perspective won't reveal gaps.
                    Vertex {
                        position: [x, -16.0],
                        waveform_index,
                        should_offset: 0.0,
                    },
                    // Top fill vertex, should_offset==1.0 means that it will be offset by audio data in the shader.
                    Vertex {
                        position: [x, 0.0],
                        waveform_index,
                        should_offset: 1.0,
                    },
                ]
            })
            .collect();
        let stroke_vertices: Vec<Vertex> = (0..stride_len)
            .map(|i| {
                let waveform_index = i as u32;
                // Stroke only has one vertex per sample
                Vertex {
                    position: [i as f32 / (stride_len - 1) as f32 * 2.0 - 1.0, 0.0],
                    waveform_index,
                    should_offset: 1.0,
                }
            })
            .collect();

        let y_values_buffer_size = stride_len.saturating_mul(NUM_INSTANCES);
        let y_values: Vec<f32> = vec![0.0; y_values_buffer_size];

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
                apply_highlight_to_front_instance: 1.0,
                _padding: Default::default(),
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
                targets: &[Some(wgpu::ColorTargetState {
                    format: swapchain_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
                    targets: &[Some(wgpu::ColorTargetState {
                        format: swapchain_format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
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

        // Initialize audio_sync offsets; offsets will be populated as data arrives.
        let audio_sync = AudioSync {
            y_value_offsets: [0; 32],
            progress: 0.0,
            num_instances: NUM_INSTANCES as f32,
        };

        RidgelineView {
            render_surface,
            is_left_channel,
            style,
            view_transform: None,
            wgpu_queue: queue.clone(),
            y_value_buffer,
            audio_sync,
            audio_sync_buffer,
            transform_buffer,
            waveform_config_buffer,
            fill_render_pipeline,
            fill_vertex_buffer,
            stroke_render_pipeline,
            stroke_vertex_buffer,
            bind_group,
            model,
            stride_len,
            y_values_buffer_size,
            write_offset: 0,
            virtual_stride_index: 0.0,
        }
    }

    #[instrument(skip(self))]
    fn get_transform_matrix(&self, panel_width_ratio: f32, horizon_offset: f32) -> [[f32; 4]; 4] {
        if let Some(view_transform) = self.view_transform {
            let screen_width = view_transform.scene_width;
            let screen_height = view_transform.scene_height;

            let (window_x, window_y, window_width, window_height) =
                view_transform.get_window_coords(panel_width_ratio);
            let window_to_scene_transform =
                view_transform.get_window_to_scene_transform(panel_width_ratio);

            let near_z = 0.0;
            let far_z = 1.0;

            let (horizontal_offset, vertical_offset) = if view_transform.is_vertical {
                (0.0, -horizon_offset)
            } else {
                (-horizon_offset, 0.0)
            };
            let (panel_width, half_screen) = if view_transform.is_vertical {
                (window_width, view_transform.scene_width / 2.0)
            } else {
                (window_height, view_transform.scene_height / 2.0)
            };

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

            // Where the half screen is bound to by the geometry.
            const HALF_SCENE_EXTENT: f32 = 1.0;
            // How much a vertex offset by 1.0 in y_value affects the height in scene units.
            const AMPLITUDE_SCALING: f32 = 0.15;
            // The perspective would fill the screen, but the first instance must be compressed
            // relatively to the screen to fit where the window frustum will point to.
            let compress_y = panel_width / half_screen;
            // A -1.0 to 1.0 y_value range will take twice AMPLITUDE_SCALING in scene units
            // and then will be compressed according to the panel width.
            let wave_amplitude_max: f32 = AMPLITUDE_SCALING * 2.0 * compress_y;

            let half_scene_ratio_with_content = wave_amplitude_max / HALF_SCENE_EXTENT;
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
                // Move the waveform to the edge
                Matrix4::from_translation(Vector3::new(0.0, -1.0, 0.0))
                // Waveform is centered at 0.0, scale it depending on the window size
                * Matrix4::from_nonuniform_scale(1.0, compress_y, 1.0)
                // Depending on the model's range, center and scale the y_values accordingly
                * match self.model.values_range() {
                    ValuesRange::NegativeOneToOne => {
                        // Scale down the y_values range to the amplitude that we want them to have in the scene
                        Matrix4::from_nonuniform_scale(1.0, AMPLITUDE_SCALING, 1.0)
                            // Move 0.0 of the y_value to the center before scaling
                            * Matrix4::from_translation(Vector3::new(0.0, 1.0, 0.0))
                    }
                    ValuesRange::ZeroToOne => {
                        // Use half the scaling vs NegativeOneToOne (same scale, but half the value range), else peaks look too thin.
                        Matrix4::from_nonuniform_scale(1.0, AMPLITUDE_SCALING, 1.0)
                    }

                };

            // The horizon offset is applied to both the frustum and the geometry, but not to the perspective matrix which keeps the horizon
            // centered where it was.
            let horizon_translate =
                Matrix4::from_translation(Vector3::new(horizontal_offset, vertical_offset, 0.0));

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

            (window_to_scene_transform
                * perspective_matrix
                * horizon_translate
                * view_transform.transform_matrix
                * compress_y_matrix)
                .into()
        } else {
            // If the view transform is not set yet, return an identity matrix
            Matrix4::<f32>::identity().into()
        }
    }
}

impl<M: AudioModel> View for RidgelineView<M> {
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
        // Select the appropriate ridgeline config based on the view's style
        let ridgeline_config = match self.style {
            ui::Style::Ridgeline => &config.ridgeline,
            ui::Style::RidgelineFrequency => &config.ridgeline_frequency,
            _ => &config.ridgeline, // Fallback (shouldn't happen)
        };

        if config_changes.is_none_or(|c| {
            c.contains(&ui::RIDGELINE_FILL_COLOR)
                || c.contains(&ui::RIDGELINE_HIGHLIGHT_COLOR)
                || c.contains(&ui::RIDGELINE_STROKE_COLOR)
        }) {
            let waveform_config = WaveformConfigUniform {
                fill_color: [
                    ridgeline_config.fill_color.red() as f32 / 255.0,
                    ridgeline_config.fill_color.green() as f32 / 255.0,
                    ridgeline_config.fill_color.blue() as f32 / 255.0,
                    ridgeline_config.fill_color.alpha() as f32 / 255.0,
                ],
                highlight_color: [
                    ridgeline_config.highlight_color.red() as f32 / 255.0,
                    ridgeline_config.highlight_color.green() as f32 / 255.0,
                    ridgeline_config.highlight_color.blue() as f32 / 255.0,
                    ridgeline_config.highlight_color.alpha() as f32 / 255.0,
                ],
                stroke_color: [
                    ridgeline_config.stroke_color.red() as f32 / 255.0,
                    ridgeline_config.stroke_color.green() as f32 / 255.0,
                    ridgeline_config.stroke_color.blue() as f32 / 255.0,
                    ridgeline_config.stroke_color.alpha() as f32 / 255.0,
                ],
                // Highlighting front instance values doesn't look good for -1..1 ranges
                // since it uses abs(y) to mix colors.
                apply_highlight_to_front_instance: if self.model.values_range()
                    == ValuesRange::ZeroToOne
                {
                    1.0
                } else {
                    0.0
                },
                _padding: Default::default(),
            };

            self.wgpu_queue.write_buffer(
                &self.waveform_config_buffer,
                0,
                bytemuck::cast_slice(&[waveform_config]),
            );
        }

        if view_transform_change.is_some()
            || config_changes.is_none_or(|c| {
                c.contains(&ui::RIDGELINE_WIDTH_RATIO) || c.contains(&ui::RIDGELINE_HORIZON_OFFSET)
            })
        {
            if view_transform_change.is_some() {
                self.view_transform = view_transform_change.cloned();
            }

            // Compute panel width in pixels from configured ratio and current scene size
            let transform_matrix = self.get_transform_matrix(
                ridgeline_config.width_ratio,
                ridgeline_config.horizon_offset,
            );
            self.wgpu_queue.write_buffer(
                &self.transform_buffer,
                0,
                bytemuck::cast_slice(&transform_matrix),
            );
        }
    }

    #[instrument(skip(self))]
    fn render(
        &self,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        depth_texture_view: &TextureView,
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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        // When clearing the color attachment, also clear the depth texture
                        load: clear_color
                            .map(|_| wgpu::LoadOp::Clear(1.0))
                            .unwrap_or(wgpu::LoadOp::Load),
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
            render_pass.draw(0..(self.stride_len * 2) as u32, 0..NUM_INSTANCES as u32);
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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.stroke_render_pipeline);
            render_pass.set_vertex_buffer(0, self.stroke_vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..self.stride_len as u32, 0..NUM_INSTANCES as u32);
        }
    }

    #[instrument(skip(self, audio_input))]
    fn process_audio(&mut self, timestamp: u32, audio_input: &AudioInputData) {
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

        // Check if we've crossed a rotation boundary (time-based, not sample-based).
        let current_floor = full_progress.floor();
        let previous_floor = self.virtual_stride_index.floor();
        let wraps = (current_floor - previous_floor) as i32;

        // On rotation: advance write_offset to start a new stride in the ring buffer.
        if wraps > 0 {
            // Rotate right: shift historical offsets to make room for new frozen stride.
            let offsets = &mut self.audio_sync.y_value_offsets[..NUM_INSTANCES];
            let effective_wraps = (wraps as usize).min(NUM_INSTANCES);
            if effective_wraps > 0 {
                offsets.rotate_right(effective_wraps);
            }
            // Advance write_offset by one stride to start writing the new front stride.
            self.write_offset = (self.write_offset + self.stride_len) % self.y_values_buffer_size;
            self.virtual_stride_index = full_progress;

            // Instance 0 always points to write_offset (the actively updating front stride).
            // Historical instances (1+) point to successively older frozen strides via rotated offsets.
            self.audio_sync.y_value_offsets[0] = self.write_offset as u32;
        }

        // Process audio with callback to write values directly to GPU buffer
        let write_offset = self.write_offset;
        let wgpu_queue = &self.wgpu_queue;
        let y_value_buffer = &self.y_value_buffer;

        let audio_sample_skip = if self.is_left_channel { 0 } else { 1 };
        let channel_samples = audio_input
            .samples
            .iter()
            .skip(audio_sample_skip)
            .step_by(2)
            .copied();
        let cqt = if self.is_left_channel {
            audio_input.cqt_left.clone()
        } else {
            audio_input.cqt_right.clone()
        };

        self.model.process_audio(channel_samples, cqt, |values| {
            // Since the write_offset is aligned, we can write in one chunk.
            assert!(write_offset + values.len() <= self.y_values_buffer_size);
            wgpu_queue.write_buffer(
                y_value_buffer,
                (write_offset * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
                bytemuck::cast_slice(values),
            );
        });

        self.wgpu_queue.write_buffer(
            &self.audio_sync_buffer,
            0,
            bytemuck::cast_slice(&[self.audio_sync]),
        );
    }
}
