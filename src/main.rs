use cgmath::{Matrix4, Rad, SquareMatrix, Vector3, Vector4};
use core::f64;
use num_complex::Complex;
use ringbuf::storage::Heap;
use ringbuf::traits::{Consumer, Observer, RingBuffer, Split};
use ringbuf::wrap::caching::Caching;
use ringbuf::{HeapRb, SharedRb};
use rustfft::{Fft, FftDirection, FftPlanner};
use signal_hook::iterator::Signals;
use slint::{ComponentHandle, Global, Model};
use splines::Interpolation;
use std::rc::Rc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use std::{borrow::Cow, sync::Arc};
use wayland_client::{Connection, QueueHandle};
use wgpu::util::DeviceExt;
use wgpu::{BufferUsages, CommandEncoder, TextureFormat, TextureView};
use wlr_layers::{PanelAnchorPosition, WlrLayerApplicationHandler, WlrWaylandEventHandler};

mod audio;
mod config;
mod ui;
mod wlr_layers;

const VERTEX_BUFFER_SIZE: usize = 44100 * 3;
const STRIDE_SIZE: usize = 1500;
const FFT_SIZE: usize = 2048;
const NUM_CHANNELS: usize = 2;

pub trait WgpuSurface {
    fn adapter(&self) -> &wgpu::Adapter;
    fn device(&self) -> &wgpu::Device;
    fn surface(&self) -> &wgpu::Surface<'static>;
    fn queue(&self) -> &Arc<wgpu::Queue>;
    fn surface_id(&self) -> u32;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    waveform_index: u32,
    should_offset: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct YValue {
    y: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AudioSync {
    y_value_offsets: [u32; 32],
    progress: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WaveformConfigUniform {
    fill_color: [f32; 4],
    stroke_color: [f32; 4],
}

struct WaveformWindow {
    wgpu: Rc<dyn WgpuSurface>,
    depth_texture: Option<wgpu::Texture>,
    config: wgpu::SurfaceConfiguration,
    swapchain_format: TextureFormat,
    render_window: RenderWindow,
    must_reconfigure: bool,
    last_fps_dump_time: Instant,
    frame_count: u32,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum RenderWindow {
    Primary,
    Secondary,
}

impl WaveformWindow {
    fn new(
        wgpu: Rc<dyn WgpuSurface>,
        width: u32,
        height: u32,
        render_window: RenderWindow,
    ) -> WaveformWindow {
        let swapchain_capabilities = wgpu.surface().get_capabilities(wgpu.adapter());
        let swapchain_format = swapchain_capabilities.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 1,
            alpha_mode: wgpu::CompositeAlphaMode::PreMultiplied,
            view_formats: vec![],
        };

        WaveformWindow {
            wgpu,
            config,
            depth_texture: None,
            swapchain_format,
            render_window,
            must_reconfigure: true,
            last_fps_dump_time: Instant::now(),
            frame_count: 0,
        }
    }

    fn reconfigure(&mut self, width: u32, height: u32) {
        // Reconfigure the surface with the new size
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.must_reconfigure = true;
    }

    fn render(
        &mut self,
        left_waveform_view: &mut Option<Box<dyn WaveformView>>,
        right_waveform_view: &mut Option<Box<dyn WaveformView>>,
    ) {
        if self.must_reconfigure {
            println!(
                "Reconfiguring {:?} window with config: {:?}",
                self.render_window, self.config,
            );
            self.wgpu
                .surface()
                .configure(self.wgpu.device(), &self.config);

            // Create the depth texture
            self.depth_texture =
                Some(self.wgpu.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("Depth Texture"),
                    size: wgpu::Extent3d {
                        width: self.config.width,
                        height: self.config.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                }));

            self.must_reconfigure = false;
        }

        let frame = self
            .wgpu
            .surface()
            .get_current_texture()
            .expect("Failed to acquire next swap chain texture");
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture_view = self
            .depth_texture
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .wgpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        if let Some(waveform_view) = left_waveform_view {
            if waveform_view.render_window() == self.render_window {
                waveform_view.update_config();
                waveform_view.render(&mut encoder, &view, &depth_texture_view);
            }
        }
        if let Some(waveform_view) = right_waveform_view {
            if waveform_view.render_window() == self.render_window {
                waveform_view.update_config();
                waveform_view.render(&mut encoder, &view, &depth_texture_view);
            }
        }
        self.wgpu.queue().submit(Some(encoder.finish()));
        frame.present();

        let now = Instant::now();
        self.frame_count += 1;
        if now.duration_since(self.last_fps_dump_time) >= Duration::from_secs(1) {
            println!("{:?}:\t{} fps", self.render_window, self.frame_count);
            self.frame_count = 0;
            self.last_fps_dump_time = now;
        }
    }
}

trait WaveformView {
    fn render_window(&self) -> RenderWindow;
    fn update_transform(&mut self, panel_width: u32, screen_width: u32, screen_height: u32);
    fn set_waveform_config(&mut self, config: ui::WaveformConfig);
    // FIXME: Remove? How will the view-specific config be abstracted?
    fn set_time_curve_control_points(&mut self, _points: Vec<ui::ControlPoint>);
    fn update_config(&mut self);
    fn process_audio(
        self: &mut Self,
        timestamp: u32,
        data: &[f32],
        fft: &dyn Fft<f32>,
        fft_inout_buffer: &mut [Complex<f32>],
        fft_scratch: &mut [Complex<f32>],
    );
    fn render(
        &self,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        depth_texture_view: &TextureView,
    );
}

struct CompressedWaveformView {
    render_window: RenderWindow,
    is_left_channel: bool,
    wgpu_queue: Arc<wgpu::Queue>,
    y_value_buffer: wgpu::Buffer,
    y_value_offset_buffer: wgpu::Buffer,
    waveform_config_buffer: wgpu::Buffer,
    fill_render_pipeline: wgpu::RenderPipeline,
    fill_vertex_buffer: wgpu::Buffer,
    stroke_render_pipeline: wgpu::RenderPipeline,
    stroke_vertex_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    fft_input_ringbuf: HeapRb<f32>,
    y_value_write_offset: usize,
    waveform_config: WaveformConfigUniform,
    waveform_config_dirty: bool,
    time_curve_control_points_dirty: bool,
    time_curve_control_points: Vec<ui::ControlPoint>,
}

impl CompressedWaveformView {
    fn new(
        window: &WaveformWindow,
        render_window: RenderWindow,
        anchor_position: PanelAnchorPosition,
        channels: ui::RenderChannels,
        is_left_channel: bool,
    ) -> CompressedWaveformView {
        // Load the shaders from disk
        let shader = window
            .wgpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("linear.wgsl"))),
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
        let y_values: Vec<YValue> = vec![YValue { y: 0.0 }; VERTEX_BUFFER_SIZE];

        let fill_vertex_buffer =
            window
                .wgpu
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Fill Vertex Buffer"),
                    contents: bytemuck::cast_slice(&fill_vertices),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });

        let stroke_vertex_buffer =
            window
                .wgpu
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Stroke Vertex Buffer"),
                    contents: bytemuck::cast_slice(&stroke_vertices),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });

        let y_value_buffer =
            window
                .wgpu
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Y Value Buffer"),
                    contents: bytemuck::cast_slice(&y_values),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        // Create the y_value_offset buffer
        let y_value_offset_buffer = window.wgpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Y Value Offset Buffer"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Identity transform is a horizontal waveform scrolling from right to left.
        let rotation = Matrix4::from_angle_z(Rad(-std::f32::consts::FRAC_PI_2));
        let mirror_h = Matrix4::from_nonuniform_scale(-1.0, 1.0, 1.0);
        let mirror_v = Matrix4::from_nonuniform_scale(1.0, -1.0, 1.0);
        let half = Matrix4::from_nonuniform_scale(0.5, 1.0, 1.0);
        let translate_half_left = Matrix4::from_translation(Vector3::new(-1.0, 0.0, 0.0));
        let translate_half_right = Matrix4::from_translation(Vector3::new(1.0, 0.0, 0.0));
        let transform_matrix = match (render_window, anchor_position, channels, is_left_channel) {
            (RenderWindow::Primary, PanelAnchorPosition::Top, ui::RenderChannels::Single, _) => {
                mirror_v
            }
            (RenderWindow::Primary, PanelAnchorPosition::Top, ui::RenderChannels::Both, true) => {
                mirror_v * half * translate_half_left
            }
            (RenderWindow::Primary, PanelAnchorPosition::Top, ui::RenderChannels::Both, false) => {
                mirror_v * half * translate_half_right * mirror_h
            }
            (RenderWindow::Primary, PanelAnchorPosition::Bottom, ui::RenderChannels::Single, _) => {
                Matrix4::identity()
            }
            (
                RenderWindow::Primary,
                PanelAnchorPosition::Bottom,
                ui::RenderChannels::Both,
                true,
            ) => half * translate_half_left,
            (
                RenderWindow::Primary,
                PanelAnchorPosition::Bottom,
                ui::RenderChannels::Both,
                false,
            ) => half * translate_half_right * mirror_h,
            (RenderWindow::Primary, PanelAnchorPosition::Left, _, _) => rotation * mirror_h,
            (RenderWindow::Secondary, PanelAnchorPosition::Right, _, _) => {
                rotation * mirror_v * mirror_h
            }
            _ => unreachable!(),
        };

        let transform_matrix_array: [[f32; 4]; 4] = transform_matrix.into();
        let transform_buffer =
            window
                .wgpu
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Transform Buffer"),
                    contents: bytemuck::cast_slice(&transform_matrix_array),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let waveform_config_buffer =
            window
                .wgpu
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Waveform Config Buffer"),
                    contents: bytemuck::cast_slice(&[WaveformConfigUniform {
                        fill_color: [1.0, 1.0, 1.0, 1.0],
                        stroke_color: [1.0, 1.0, 1.0, 1.0],
                    }]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        // Create the bind group layout and bind group for the uniform
        let bind_group_layout =
            window
                .wgpu
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let bind_group = window
            .wgpu
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
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

        let pipeline_layout =
            window
                .wgpu
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let fill_render_pipeline =
            window
                .wgpu
                .device()
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                        targets: &[Some(window.swapchain_format.into())],
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
            window
                .wgpu
                .device()
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                        targets: &[Some(window.swapchain_format.into())],
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

        CompressedWaveformView {
            render_window,
            is_left_channel,
            wgpu_queue: window.wgpu.queue().clone(),
            y_value_buffer,
            y_value_offset_buffer,
            waveform_config_buffer,
            fill_render_pipeline,
            fill_vertex_buffer,
            stroke_render_pipeline,
            stroke_vertex_buffer,
            bind_group,
            fft_input_ringbuf: HeapRb::<f32>::new(FFT_SIZE),
            y_value_write_offset: 0,
            waveform_config: WaveformConfigUniform {
                fill_color: [1.0, 1.0, 1.0, 1.0],
                stroke_color: [1.0, 1.0, 1.0, 1.0],
            },
            waveform_config_dirty: true,
            time_curve_control_points_dirty: false,
            time_curve_control_points: vec![],
        }
    }
}

impl WaveformView for CompressedWaveformView {
    fn render_window(&self) -> RenderWindow {
        self.render_window
    }

    fn update_transform(&mut self, _panel_width: u32, _screen_width: u32, _screen_height: u32) {}

    fn set_waveform_config(&mut self, config: ui::WaveformConfig) {
        self.waveform_config.fill_color = [
            config.fill_color.red() as f32 / 255.0,
            config.fill_color.green() as f32 / 255.0,
            config.fill_color.blue() as f32 / 255.0,
            config.fill_color.alpha() as f32 / 255.0,
        ];
        self.waveform_config.stroke_color = [
            config.stroke_color.red() as f32 / 255.0,
            config.stroke_color.green() as f32 / 255.0,
            config.stroke_color.blue() as f32 / 255.0,
            config.stroke_color.alpha() as f32 / 255.0,
        ];
        self.waveform_config_dirty = true;
    }

    fn update_config(&mut self) {
        if self.waveform_config_dirty {
            self.wgpu_queue.write_buffer(
                &self.waveform_config_buffer,
                0,
                bytemuck::cast_slice(&[self.waveform_config]),
            );
            self.waveform_config_dirty = false;
        }

        if self.time_curve_control_points_dirty {
            let control_points_with_prefix_suffix_iter =
                std::iter::once((0.0, 0.0, Interpolation::Linear))
                    .chain(self.time_curve_control_points.iter().map(|cp| {
                        (
                            (cp.t / 3.0 + 1.0) * (VERTEX_BUFFER_SIZE - 1) as f32,
                            cp.v,
                            Interpolation::CatmullRom,
                        )
                    }))
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

            self.time_curve_control_points_dirty = false;
        }
    }

    fn render(
        &self,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        _depth_texture_view: &TextureView,
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
                    view: &view,
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

    fn process_audio(
        self: &mut Self,
        _timestamp: u32,
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

    fn set_time_curve_control_points(&mut self, points: Vec<ui::ControlPoint>) {
        self.time_curve_control_points = points;
        self.time_curve_control_points_dirty = true;
    }
}

struct RidgelineWaveformView {
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
    transform_matrix: [[f32; 4]; 4],
    transform_matrix_dirty: bool,
    waveform_config: WaveformConfigUniform,
    waveform_config_dirty: bool,
}

impl RidgelineWaveformView {
    fn new(
        window: &WaveformWindow,
        render_window: RenderWindow,
        anchor_position: PanelAnchorPosition,
        channels: ui::RenderChannels,
        is_left_channel: bool,
    ) -> RidgelineWaveformView {
        // Load the shaders from disk
        let shader = window
            .wgpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
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

        let fill_vertex_buffer =
            window
                .wgpu
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Fill Vertex Buffer"),
                    contents: bytemuck::cast_slice(&fill_vertices),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });

        let stroke_vertex_buffer =
            window
                .wgpu
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Stroke Vertex Buffer"),
                    contents: bytemuck::cast_slice(&stroke_vertices),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });

        let y_value_buffer =
            window
                .wgpu
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Y Value Buffer"),
                    contents: bytemuck::cast_slice(&y_values),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        // Create the y_value_offset buffer
        let audio_sync_buffer = window.wgpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Audio Sync Buffer"),
            size: (std::mem::size_of::<u32>() * 32 + 4) as wgpu::BufferAddress,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Must be updated by set_panel_width
        let identity: [[f32; 4]; 4] = Matrix4::<f32>::identity().into();
        let transform_buffer =
            window
                .wgpu
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Transform Buffer"),
                    contents: bytemuck::cast_slice(&identity),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let waveform_config_buffer =
            window
                .wgpu
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Waveform Config Buffer"),
                    contents: bytemuck::cast_slice(&[WaveformConfigUniform {
                        fill_color: [1.0, 1.0, 1.0, 1.0],
                        stroke_color: [1.0, 1.0, 1.0, 1.0],
                    }]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        // Create the bind group layout and bind group for the uniform
        let bind_group_layout =
            window
                .wgpu
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let bind_group = window
            .wgpu
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
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

        let pipeline_layout =
            window
                .wgpu
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let fill_render_pipeline =
            window
                .wgpu
                .device()
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                        targets: &[Some(window.swapchain_format.into())],
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
            window
                .wgpu
                .device()
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                        targets: &[Some(window.swapchain_format.into())],
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
            wgpu_queue: window.wgpu.queue().clone(),
            y_value_buffer,
            audio_sync: AudioSync {
                y_value_offsets: [0; 32],
                progress: 0.0,
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
            transform_matrix: Default::default(),
            transform_matrix_dirty: false,
            waveform_config: WaveformConfigUniform {
                fill_color: [1.0, 1.0, 1.0, 1.0],
                stroke_color: [1.0, 1.0, 1.0, 1.0],
            },
            waveform_config_dirty: false,
        }
    }
}

impl WaveformView for RidgelineWaveformView {
    fn render_window(&self) -> RenderWindow {
        self.render_window
    }

    fn update_transform(&mut self, panel_width: u32, screen_width: u32, screen_height: u32) {
        let panel_width = panel_width as f32;
        let screen_width = screen_width as f32;
        let screen_height = screen_height as f32;

        // This and the window_x/y assume that the surface is on screen edges.
        // Other panels positioned in-between could make the perspective transform incorrect
        // if they are large and this would require using the actual layer surface position
        // on the screen instead of using the anchor.
        let (window_width, window_height, half_screen) = match self.anchor_position {
            PanelAnchorPosition::Top => (screen_width, panel_width, screen_height / 2.0),
            PanelAnchorPosition::Bottom => (screen_width, panel_width, screen_height / 2.0),
            PanelAnchorPosition::Left => (panel_width, screen_height, screen_width / 2.0),
            PanelAnchorPosition::Right => (panel_width, screen_height, screen_width / 2.0),
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

            let full_top = 1.0;
            let full_bottom = -1.0;
            let full_right = 1.0;
            let full_left = -1.0;

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

            self.transform_matrix =
                (perspective_matrix * transform_matrix * compress_y_matrix).into();
            self.transform_matrix_dirty = true;
        }
    }

    fn set_waveform_config(&mut self, config: ui::WaveformConfig) {
        self.waveform_config.fill_color = [
            config.fill_color.red() as f32 / 255.0,
            config.fill_color.green() as f32 / 255.0,
            config.fill_color.blue() as f32 / 255.0,
            config.fill_color.alpha() as f32 / 255.0,
        ];
        self.waveform_config.stroke_color = [
            config.stroke_color.red() as f32 / 255.0,
            config.stroke_color.green() as f32 / 255.0,
            config.stroke_color.blue() as f32 / 255.0,
            config.stroke_color.alpha() as f32 / 255.0,
        ];
        self.waveform_config_dirty = true;
    }

    fn update_config(&mut self) {
        if self.transform_matrix_dirty {
            self.wgpu_queue.write_buffer(
                &self.transform_buffer,
                0,
                bytemuck::cast_slice(&self.transform_matrix),
            );
            self.transform_matrix_dirty = false;
        }

        if self.waveform_config_dirty {
            self.wgpu_queue.write_buffer(
                &self.waveform_config_buffer,
                0,
                bytemuck::cast_slice(&[self.waveform_config]),
            );
            self.waveform_config_dirty = false;
        }
    }

    fn render(
        &self,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        depth_texture_view: &TextureView,
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
                    view: &depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
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
            render_pass.draw(
                0..(STRIDE_SIZE * 2) as u32,
                0..32, // Render 32 instances
            );
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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_texture_view,
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
            render_pass.draw(
                0..STRIDE_SIZE as u32,
                0..32, // Render 32 instances
            );
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
            t * 48000.0 / STRIDE_SIZE as f64
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
            self.audio_sync.y_value_offsets.rotate_left(1);
            self.last_rotate_progress = full_progress;
        }

        // Keep updating the latest stride even between rotations.
        // The front-most stride is animated while the rest are static
        // and move backwards.
        self.audio_sync.y_value_offsets[31] = aligned_write_offset as u32;

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

    fn set_time_curve_control_points(&mut self, _points: Vec<ui::ControlPoint>) {}
}

struct ApplicationState {
    // The actual state is in an Option because its initialization is delayed to after
    // the even loop starts running.
    windowed_state: Option<WindowedApplicationState>,
    animation_stopped: Arc<AtomicBool>,
    request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
}

impl ApplicationState {
    fn new(request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>) -> ApplicationState {
        ApplicationState {
            windowed_state: None,
            animation_stopped: Arc::new(AtomicBool::new(false)),
            request_redraw_callback,
        }
    }
}

struct WindowedApplicationState {
    primary_waveform_window: Option<(WaveformWindow, PanelAnchorPosition)>,
    secondary_waveform_window: Option<(WaveformWindow, PanelAnchorPosition)>,
    last_non_zero_sample_age: usize,
    animation_stopped: Arc<AtomicBool>,
    left_waveform_view: Option<Box<dyn WaveformView>>,
    right_waveform_view: Option<Box<dyn WaveformView>>,
    screen_size: (u32, u32),
    style: ui::Style,
    channels: ui::RenderChannels,
    panel_width: u32,
    waveform_config: ui::WaveformConfig,
    time_curve_control_points: Vec<ui::ControlPoint>,
    audio_input_ringbuf_cons: Caching<Arc<SharedRb<Heap<f32>>>, false, true>,
    fft: Arc<dyn Fft<f32>>,
    fft_inout_buffer: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

impl WindowedApplicationState {
    fn new(
        animation_stopped: Arc<AtomicBool>,
        request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
    ) -> WindowedApplicationState {
        let (audio_input_ringbuf_prod, audio_input_ringbuf_cons) =
            HeapRb::<f32>::new(44100 * NUM_CHANNELS).split();

        std::thread::spawn({
            let animation_stopped = animation_stopped.clone();
            move || {
                audio::initialize_audio_capture(
                    audio_input_ringbuf_prod,
                    animation_stopped,
                    request_redraw_callback,
                );
            }
        });

        let fft = FftPlanner::new().plan_fft(FFT_SIZE, FftDirection::Forward);
        let scratch_len = fft.get_inplace_scratch_len();
        WindowedApplicationState {
            primary_waveform_window: None,
            secondary_waveform_window: None,
            last_non_zero_sample_age: 0,
            animation_stopped,
            left_waveform_view: None,
            right_waveform_view: None,
            screen_size: (1, 1),
            panel_width: 0,
            style: Default::default(),
            channels: Default::default(),
            waveform_config: Default::default(),
            time_curve_control_points: vec![],
            audio_input_ringbuf_cons,
            fft,
            fft_inout_buffer: vec![Complex::default(); FFT_SIZE],
            fft_scratch: vec![Complex::default(); scratch_len],
        }
    }

    fn create_waveform_view(
        &self,
        style: ui::Style,
        render_window: RenderWindow,
        is_left_channel: bool,
    ) -> Box<dyn WaveformView> {
        let (window, anchor_position) = match render_window {
            RenderWindow::Primary => self.primary_waveform_window.as_ref().unwrap(),
            RenderWindow::Secondary => self.secondary_waveform_window.as_ref().unwrap(),
        };
        let mut view: Box<dyn WaveformView> = match style {
            ui::Style::Ridgeline => Box::new(RidgelineWaveformView::new(
                &window,
                render_window,
                *anchor_position,
                self.channels,
                is_left_channel,
            )),
            ui::Style::Compressed => Box::new(CompressedWaveformView::new(
                &window,
                render_window,
                *anchor_position,
                self.channels,
                is_left_channel,
            )),
        };
        view.update_transform(self.panel_width, self.screen_size.0, self.screen_size.1);
        view.set_waveform_config(self.waveform_config.clone());
        view.set_time_curve_control_points(self.time_curve_control_points.clone());
        view
    }

    fn configure_primary_wgpu_surface(
        &mut self,
        wgpu_surface: Rc<dyn WgpuSurface>,
        anchor_position: PanelAnchorPosition,
        width: u32,
        height: u32,
    ) {
        let window =
            WaveformWindow::new(wgpu_surface.clone(), width, height, RenderWindow::Primary);
        self.primary_waveform_window = Some((window, anchor_position));

        self.left_waveform_view =
            Some(self.create_waveform_view(self.style, RenderWindow::Primary, true));

        if self.channels == ui::RenderChannels::Both {
            self.right_waveform_view =
                Some(self.create_waveform_view(self.style, RenderWindow::Primary, false));
        }
    }

    fn configure_secondary_wgpu_surface(
        &mut self,
        wgpu_surface: Rc<dyn WgpuSurface>,
        anchor_position: PanelAnchorPosition,
        width: u32,
        height: u32,
    ) {
        let window =
            WaveformWindow::new(wgpu_surface.clone(), width, height, RenderWindow::Secondary);
        self.secondary_waveform_window = Some((window, anchor_position));

        self.right_waveform_view =
            Some(self.create_waveform_view(self.style, RenderWindow::Secondary, false));
    }

    fn process_audio(&mut self, timestamp: u32) {
        let data: Vec<f32> = self.audio_input_ringbuf_cons.pop_iter().collect();
        if data.iter().all(|&x| x == 0.0) {
            self.last_non_zero_sample_age += data.len();
            if self.last_non_zero_sample_age > VERTEX_BUFFER_SIZE * 2 {
                // Stop requesting new frames and let the audio thread know if they
                // should wake us up once non-zero samples are available.
                self.animation_stopped.store(true, Ordering::Relaxed);
            }
        } else {
            self.last_non_zero_sample_age = 0;
        }

        if let Some(left_waveform_window) = &mut self.left_waveform_view {
            left_waveform_window.process_audio(
                timestamp,
                &data,
                self.fft.as_ref(),
                &mut self.fft_inout_buffer,
                &mut self.fft_scratch,
            );
        }
        if let Some(right_waveform_window) = &mut self.right_waveform_view {
            right_waveform_window.process_audio(
                timestamp,
                &data,
                self.fft.as_ref(),
                &mut self.fft_inout_buffer,
                &mut self.fft_scratch,
            );
        }
    }
    fn render(&mut self, surface_id: u32) {
        if let Some((window, _)) = self.primary_waveform_window.as_mut() {
            if window.wgpu.surface_id() == surface_id {
                window.render(&mut self.left_waveform_view, &mut self.right_waveform_view);
            }
        }

        if let Some((window, _)) = self.secondary_waveform_window.as_mut() {
            if window.wgpu.surface_id() == surface_id {
                window.render(&mut self.left_waveform_view, &mut self.right_waveform_view);
            }
        }
    }

    fn set_screen_size(&mut self, width: u32, height: u32) {
        self.screen_size.0 = width;
        self.screen_size.1 = height;
        if let Some(view) = &mut self.left_waveform_view {
            view.update_transform(self.panel_width, width, height);
        }
        if let Some(view) = &mut self.right_waveform_view {
            view.update_transform(self.panel_width, width, height);
        }
    }

    fn recreate_views(&mut self) {
        if self.primary_waveform_window.is_some() {
            self.left_waveform_view =
                Some(self.create_waveform_view(self.style, RenderWindow::Primary, true));

            if self.channels == ui::RenderChannels::Both {
                self.right_waveform_view =
                    Some(self.create_waveform_view(self.style, RenderWindow::Primary, false));
            } else {
                self.right_waveform_view = None;
            }
        }
        if self.secondary_waveform_window.is_some() {
            self.right_waveform_view =
                Some(self.create_waveform_view(self.style, RenderWindow::Secondary, false));
        }
    }

    fn set_style(&mut self, style: ui::Style) {
        self.style = style;
        self.recreate_views();
    }
    fn set_channels(&mut self, channels: ui::RenderChannels) {
        self.channels = channels;
        self.recreate_views();
    }
    fn set_panel_width(&mut self, width: u32) {
        self.panel_width = width;
        if let Some(view) = &mut self.left_waveform_view {
            view.update_transform(width, self.screen_size.0, self.screen_size.1);
        }
        if let Some(view) = &mut self.right_waveform_view {
            view.update_transform(width, self.screen_size.0, self.screen_size.1);
        }
    }
    fn set_waveform_config(&mut self, config: ui::WaveformConfig) {
        self.waveform_config = config.clone();
        if let Some(view) = &mut self.left_waveform_view {
            view.set_waveform_config(config.clone());
        }
        if let Some(view) = &mut self.right_waveform_view {
            view.set_waveform_config(config);
        }
    }

    fn set_time_curve_control_points(&mut self, points: Vec<ui::ControlPoint>) {
        self.time_curve_control_points = points.clone();
        if let Some(view) = &mut self.left_waveform_view {
            view.set_time_curve_control_points(points.clone());
        }
        if let Some(view) = &mut self.right_waveform_view {
            view.set_time_curve_control_points(points);
        }
    }
}

impl WlrLayerApplicationHandler for ApplicationState {
    fn initialize_app_state(&mut self) {
        self.windowed_state = Some(WindowedApplicationState::new(
            self.animation_stopped.clone(),
            self.request_redraw_callback.clone(),
        ));
    }

    fn configure_primary_wgpu_surface(
        &mut self,
        wgpu_surface: Rc<dyn WgpuSurface>,
        anchor_position: PanelAnchorPosition,
        width: u32,
        height: u32,
    ) {
        self.windowed_state
            .as_mut()
            .unwrap()
            .configure_primary_wgpu_surface(wgpu_surface, anchor_position, width, height);
    }

    fn configure_secondary_wgpu_surface(
        &mut self,
        wgpu_surface: Rc<dyn WgpuSurface>,
        anchor_position: PanelAnchorPosition,
        width: u32,
        height: u32,
    ) {
        self.windowed_state
            .as_mut()
            .unwrap()
            .configure_secondary_wgpu_surface(wgpu_surface, anchor_position, width, height);
    }

    fn set_screen_size(&mut self, width: u32, height: u32) {
        self.windowed_state
            .as_mut()
            .unwrap()
            .set_screen_size(width, height);
    }

    fn primary_resized(&mut self, width: u32, height: u32) {
        if let Some((waveform_window, _)) = self
            .windowed_state
            .as_mut()
            .unwrap()
            .primary_waveform_window
            .as_mut()
        {
            waveform_window.reconfigure(width, height);
        }
    }
    fn secondary_resized(&mut self, width: u32, height: u32) {
        if let Some((waveform_window, _)) = self
            .windowed_state
            .as_mut()
            .unwrap()
            .secondary_waveform_window
            .as_mut()
        {
            waveform_window.reconfigure(width, height);
        }
    }
    fn process_audio(&mut self, timestamp: u32) {
        self.windowed_state
            .as_mut()
            .unwrap()
            .process_audio(timestamp);
    }
    fn render(&mut self, surface_id: u32) {
        self.windowed_state.as_mut().unwrap().render(surface_id);
    }
}

enum UiMessage {
    ApplicationStateCallback(Box<dyn FnOnce(&mut WindowedApplicationState) + Send>),
    WlrWaylandEventHandlerCallback(
        Box<
            dyn FnOnce(
                    &mut WlrWaylandEventHandler,
                    &Connection,
                    &QueueHandle<WlrWaylandEventHandler>,
                ) + Send,
        >,
    ),
}

pub fn main() {
    let (ui_msg_tx, ui_msg_rx) = mpsc::channel::<UiMessage>();
    let request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>> =
        Arc::new(Mutex::new(Arc::new(|| {})));

    let send_ui_msg = {
        let request_redraw_callback = request_redraw_callback.clone();
        move |msg| {
            ui_msg_tx.send(msg).unwrap();
            request_redraw_callback.lock().unwrap()();
        }
    };

    let config_window = ui::init(send_ui_msg.clone());
    let configuration = ui::Configuration::get(&config_window);

    if let Err(e) = config::load_configuration(&configuration) {
        eprintln!("Failed to load configuration: {}", e);
    }

    // Apply the initial waveform and time curve configuration explicitly since it won't be passed in as parameter like the panel config
    {
        let channels = configuration.get_panel_channels();
        let waveform = configuration.get_waveform();

        let control_points = configuration.get_time_curve_control_points();
        let updated_points: Vec<_> = (0..control_points.row_count())
            .map(|i| control_points.row_data(i).unwrap())
            .collect();

        send_ui_msg(UiMessage::ApplicationStateCallback(Box::new(move |ws| {
            ws.set_channels(channels);
            ws.set_waveform_config(waveform);
            ws.set_time_curve_control_points(updated_points);
        })));
    }

    let panel_config = wlr_layers::PanelConfig {
        layout: configuration.get_panel_layout(),
        layer: configuration.get_panel_layer(),
        width: configuration.get_panel_width() as u32,
        exclusive_ratio: configuration.get_panel_exclusive_ratio(),
    };

    // Spawn the wlr panel rendering in a separate thread, this is supported with wayland
    std::thread::spawn(move || {
        let app_state = ApplicationState::new(request_redraw_callback.clone());

        let mut layers_even_queue = wlr_layers::WlrWaylandEventLoop::new(
            app_state,
            ui_msg_rx,
            panel_config,
            request_redraw_callback,
        );
        layers_even_queue.run_event_loop();
    });

    // The panels don't accept using input, so allow showing the config window again through SIGUSR1.
    std::thread::spawn({
        let mut signals = Signals::new(&[signal_hook::consts::SIGUSR1]).unwrap();
        let window_weak = config_window.as_weak();
        move || {
            for sig in signals.forever() {
                if sig == signal_hook::consts::SIGUSR1 {
                    window_weak
                        .upgrade_in_event_loop(|window| {
                            window.show().unwrap();
                        })
                        .unwrap();
                }
            }
        }
    });

    // Tie the main thread to the config window, since winit needs to be there on some platforms.
    slint::run_event_loop_until_quit().unwrap();
}
