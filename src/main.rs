use cgmath::{Matrix4, Rad, SquareMatrix};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use num_complex::Complex;
use ringbuf::storage::Heap;
use ringbuf::traits::{Consumer, Observer, Producer, RingBuffer, Split};
use ringbuf::wrap::caching::Caching;
use ringbuf::{HeapRb, SharedRb};
use rustfft::{Fft, FftDirection, FftPlanner};
use slint::Color;
use splines::Interpolation;
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use std::{borrow::Cow, sync::Arc};
use wgpu::util::DeviceExt;
use wgpu::BufferUsages;
use winit::platform::wayland::EventLoopBuilderExtWayland;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};
use wlr_layers::{PanelAnchorPosition, WlrLayerApplicationHandler};

mod wlr_layers;

const VERTEX_BUFFER_SIZE: usize = 44100 * 3;
const FFT_SIZE: usize = 2048;
const NUM_CHANNELS: usize = 2;

pub trait WgpuSurface {
    fn adapter(&self) -> &wgpu::Adapter;
    fn device(&self) -> &wgpu::Device;
    fn surface(&self) -> &wgpu::Surface<'static>;
    fn queue(&self) -> &Arc<wgpu::Queue>;
    fn request_redraw_callback(&self) -> &Arc<dyn Fn() + Send + Sync>;
}

pub struct WinitWgpuSurface {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    surface: wgpu::Surface<'static>,
    queue: Arc<wgpu::Queue>,
    request_redraw_callback: Arc<dyn Fn() + Send + Sync>,

    _window: Arc<Window>,
}

impl WinitWgpuSurface {
    fn new(window: Arc<Window>) -> WinitWgpuSurface {
        let mut size = window.inner_size();
        size.width = size.width.max(1);
        size.height = size.height.max(1);
        println!("Window size: {:?}", size);

        let instance = wgpu::Instance::default();

        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
            &instance,
            Some(&surface),
        ))
        .expect("Failed to find an appropriate adapter");

        // Create the logical device and command queue
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        ))
        .expect("Failed to create device");

        WinitWgpuSurface {
            adapter,
            device,
            surface,
            queue: Arc::new(queue),
            request_redraw_callback: {
                let window = window.clone();
                Arc::new(move || window.request_redraw())
            },
            _window: window,
        }
    }
}
impl WgpuSurface for WinitWgpuSurface {
    fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }
    fn device(&self) -> &wgpu::Device {
        &self.device
    }
    fn surface(&self) -> &wgpu::Surface<'static> {
        &self.surface
    }
    fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }
    fn request_redraw_callback(&self) -> &Arc<dyn Fn() + Send + Sync> {
        &self.request_redraw_callback
    }
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
struct WaveformConfig {
    fill_color: [f32; 4],
    stroke_color: [f32; 4],
}

struct WaveformWindow {
    wgpu_surface: Box<dyn WgpuSurface>,
    _shader: wgpu::ShaderModule,
    _pipeline_layout: wgpu::PipelineLayout,
    y_value_buffer: wgpu::Buffer,
    y_value_offset_buffer: wgpu::Buffer,
    waveform_config_buffer: wgpu::Buffer,
    fill_render_pipeline: wgpu::RenderPipeline,
    fill_vertex_buffer: wgpu::Buffer,
    stroke_render_pipeline: wgpu::RenderPipeline,
    stroke_vertex_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    config: wgpu::SurfaceConfiguration,
    audio_sample_skip: usize,
    fft_input_ringbuf: HeapRb<f32>,
    y_value_write_offset: usize,
    waveform_config: WaveformConfig,
    waveform_config_dirty: bool,
}

impl WaveformWindow {
    fn new(
        wgpu: Box<dyn WgpuSurface>,
        audio_sample_skip: usize,
        transform_matrix: [[f32; 4]; 4],
    ) -> WaveformWindow {
        // Load the shaders from disk
        let shader = wgpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
            });

        let pipeline_layout =
            wgpu.device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Pipeline Layout"),
                    bind_group_layouts: &[],
                    push_constant_ranges: &[],
                });

        let swapchain_capabilities = wgpu.surface().get_capabilities(wgpu.adapter());
        let swapchain_format = swapchain_capabilities.formats[0];

        // Define control points for the spline used to make the waveform look more compressed for older samples.
        let cps = vec![
            (0.0, 0.0, Interpolation::Linear),
            (0.0, 0.0, Interpolation::CatmullRom),
            (
                VERTEX_BUFFER_SIZE as f32 - 44100.0,
                0.15,
                Interpolation::CatmullRom,
            ),
            (
                VERTEX_BUFFER_SIZE as f32 - 44100.0 / 5.0,
                0.3,
                Interpolation::CatmullRom,
            ),
            (
                VERTEX_BUFFER_SIZE as f32 - 44100.0 / 30.0,
                0.5,
                Interpolation::CatmullRom,
            ),
            (
                VERTEX_BUFFER_SIZE as f32 - 44100.0 / 60.0,
                0.6,
                Interpolation::CatmullRom,
            ),
            (VERTEX_BUFFER_SIZE as f32, 1.0, Interpolation::Linear),
            (VERTEX_BUFFER_SIZE as f32, 1.0, Interpolation::Linear),
        ];

        // Create the spline
        let spline = splines::Spline::from_vec(
            cps.iter()
                .map(|(x, y, interpolation)| splines::Key::new(*x, *y, *interpolation))
                .collect(),
        );

        // Sample the spline for every waveform vertex so that the position of older samples are closer together
        let fill_vertices: Vec<Vertex> = (0..VERTEX_BUFFER_SIZE)
            .flat_map(|i| {
                let x = spline.sample(i as f32).unwrap() * 2.0 - 1.0;
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
                position: [spline.sample(i as f32).unwrap() * 2.0 - 1.0, 0.0],
                waveform_index: i as u32,
                should_offset: 1.0,
            })
            .collect();
        let y_values: Vec<YValue> = vec![YValue { y: 0.0 }; VERTEX_BUFFER_SIZE];

        let fill_vertex_buffer =
            wgpu.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Fill Vertex Buffer"),
                    contents: bytemuck::cast_slice(&fill_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        let stroke_vertex_buffer =
            wgpu.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Stroke Vertex Buffer"),
                    contents: bytemuck::cast_slice(&stroke_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        let y_value_buffer = wgpu
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Y Value Buffer"),
                contents: bytemuck::cast_slice(&y_values),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create the y_value_offset buffer
        let y_value_offset_buffer = wgpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Y Value Offset Buffer"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let transform_buffer =
            wgpu.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Transform Buffer"),
                    contents: bytemuck::cast_slice(&transform_matrix),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let waveform_config_buffer =
            wgpu.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Waveform Config Buffer"),
                    contents: bytemuck::cast_slice(&[WaveformConfig {
                        fill_color: [1.0, 1.0, 1.0, 1.0],
                        stroke_color: [1.0, 1.0, 1.0, 1.0],
                    }]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        // Create the bind group layout and bind group for the uniform
        let bind_group_layout =
            wgpu.device()
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

        let bind_group = wgpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
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

        let fill_render_pipeline =
            wgpu.device()
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&wgpu.device().create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: Some("Pipeline Layout"),
                            bind_group_layouts: &[&bind_group_layout],
                            push_constant_ranges: &[],
                        },
                    )),
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
            wgpu.device()
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&wgpu.device().create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: Some("Pipeline Layout"),
                            bind_group_layouts: &[&bind_group_layout],
                            push_constant_ranges: &[],
                        },
                    )),
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

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            // FIXME: Init wgpu on configure?
            width: 800,
            height: 600,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 1,
            alpha_mode: wgpu::CompositeAlphaMode::PreMultiplied,
            view_formats: vec![],
        };
        wgpu.surface().configure(wgpu.device(), &config);

        WaveformWindow {
            wgpu_surface: wgpu,
            _shader: shader,
            _pipeline_layout: pipeline_layout,
            y_value_buffer,
            y_value_offset_buffer,
            waveform_config_buffer,
            fill_render_pipeline,
            fill_vertex_buffer,
            stroke_render_pipeline,
            stroke_vertex_buffer,
            bind_group,
            config,
            audio_sample_skip,
            fft_input_ringbuf: HeapRb::<f32>::new(FFT_SIZE),
            y_value_write_offset: 0,
            waveform_config: WaveformConfig {
                fill_color: [1.0, 1.0, 1.0, 1.0],
                stroke_color: [1.0, 1.0, 1.0, 1.0],
            },
            waveform_config_dirty: true,
        }
    }

    fn set_fill_color(&mut self, fill_color: Color) {
        self.waveform_config.fill_color = [
            fill_color.red() as f32 / 255.0,
            fill_color.green() as f32 / 255.0,
            fill_color.blue() as f32 / 255.0,
            fill_color.alpha() as f32 / 255.0,
        ];
        self.waveform_config_dirty = true;
    }

    fn set_stroke_color(&mut self, stroke_color: Color) {
        self.waveform_config.stroke_color = [
            stroke_color.red() as f32 / 255.0,
            stroke_color.green() as f32 / 255.0,
            stroke_color.blue() as f32 / 255.0,
            stroke_color.alpha() as f32 / 255.0,
        ];
        self.waveform_config_dirty = true;
    }

    fn reconfigure(&mut self, width: u32, height: u32) {
        // Reconfigure the surface with the new size
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.wgpu_surface
            .surface()
            .configure(self.wgpu_surface.device(), &self.config);
        // On macos the window needs to be redrawn manually after resizing
        (self.wgpu_surface.request_redraw_callback())();
    }

    fn update_config(&mut self) {
        if self.waveform_config_dirty {
            self.wgpu_surface.queue().write_buffer(
                &self.waveform_config_buffer,
                0,
                bytemuck::cast_slice(&[self.waveform_config]),
            );
            self.waveform_config_dirty = false;
        }
    }

    fn render(&self) {
        let frame = self
            .wgpu_surface
            .surface()
            .get_current_texture()
            .expect("Failed to acquire next swap chain texture");
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .wgpu_surface
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
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
                        // Last pass
                        store: wgpu::StoreOp::Discard,
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

        self.wgpu_surface.queue().submit(Some(encoder.finish()));
        frame.present();
    }

    pub fn process_audio(
        self: &mut Self,
        data: &[f32],
        fft: &dyn Fft<f32>,
        fft_inout_buffer: &mut [Complex<f32>],
        fft_scratch: &mut [Complex<f32>],
    ) {
        // Keep track of the last FFT size samples.
        self.fft_input_ringbuf
            .push_iter_overwrite(data.iter().skip(self.audio_sample_skip).step_by(2).copied());

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

        let data_iter = data.iter().skip(self.audio_sample_skip).step_by(2).copied();
        let y_values: Vec<YValue> = data_iter.map(|sample| YValue { y: sample }).collect();

        // First pass: write to the end of the buffer
        let first_pass_len = {
            let first_pass_len = VERTEX_BUFFER_SIZE - self.y_value_write_offset;
            let first_pass_data = &y_values[..first_pass_len.min(y_values.len())];
            self.wgpu_surface.queue().write_buffer(
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
            self.wgpu_surface.queue().write_buffer(
                &self.y_value_buffer,
                0,
                bytemuck::cast_slice(second_pass_data),
            );
        }

        // Tell the shader how to read the audio data ring buffer
        self.wgpu_surface.queue().write_buffer(
            &self.y_value_offset_buffer,
            0,
            bytemuck::cast_slice(&[aligned_write_offset as u32]),
        );
    }
}

struct ApplicationState {
    // See https://docs.rs/winit/latest/winit/changelog/v0_30/index.html#removed
    // for the recommended practice regarding Window creation (from which everything depends)
    // in winit >= 0.30.0.
    // The actual state is in an Option because its initialization is now delayed to after
    // the even loop starts running.
    windowed_state: Option<WindowedApplicationState>,
    process_audio: Arc<Mutex<bool>>,
}

impl ApplicationState {
    fn new() -> ApplicationState {
        ApplicationState {
            windowed_state: None,
            process_audio: Arc::new(Mutex::new(true)),
        }
    }

    fn create_winit_window_and_wgpu_surface(event_loop: &ActiveEventLoop) -> Box<WinitWgpuSurface> {
        #[allow(unused_mut)]
        let mut attributes = Window::default_attributes();
        let window = Arc::new(event_loop.create_window(attributes).unwrap());

        Box::new(WinitWgpuSurface::new(window))
    }
}

struct WindowedApplicationState {
    left_waveform_window: Option<WaveformWindow>,
    right_waveform_window: Option<WaveformWindow>,
    left_request_redraw: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
    right_request_redraw: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
    audio_input_ringbuf_cons: Caching<Arc<SharedRb<Heap<f32>>>, false, true>,
    _stream: Option<cpal::Stream>,
    last_fps_dump_time: Instant,
    frame_count: u32,
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
    fn new(process_audio: Arc<Mutex<bool>>) -> WindowedApplicationState {
        let (mut audio_input_ringbuf_prod, audio_input_ringbuf_cons) =
            HeapRb::<f32>::new(44100 * NUM_CHANNELS).split();

        let left_request_redraw: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>> =
            Arc::new(Mutex::new(Arc::new(|| {})));
        let right_request_redraw: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>> =
            Arc::new(Mutex::new(Arc::new(|| {})));
        let request_redraw = {
            let left_request_redraw = left_request_redraw.clone();
            let right_request_redraw = right_request_redraw.clone();
            move || {
                left_request_redraw.lock().unwrap()();
                right_request_redraw.lock().unwrap()();
            }
        };

        let arc_process_audio = process_audio.clone();

        // List all cpal input sources
        let host = cpal::default_host();
        // let devices = host.input_devices().unwrap();
        // for device in devices {
        //     println!("Input device: {}", device.name().unwrap());
        // }

        // Print all input configs for the default input device
        let stream = if let Some(default_input_device) = host.default_input_device() {
            println!(
                "Default input device: {}",
                default_input_device.name().unwrap()
            );
            // let configs = default_input_device.supported_input_configs().unwrap();
            // for config in configs {
            //     println!("Supported input config: {:?}", config);
            // }

            // Modify the data_callback to update the vertex buffer directly and trigger a redraw
            let default_config = default_input_device.default_input_config().unwrap();
            println!("Default input config: {:?}", default_config);
            let stream: cpal::Stream = default_input_device
                .build_input_stream(
                    &cpal::StreamConfig {
                        buffer_size: cpal::BufferSize::Fixed(512),
                        ..default_config.into()
                    },
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        let process_audio = *arc_process_audio.lock().unwrap();
                        if process_audio {
                            audio_input_ringbuf_prod.push_slice(data);

                            request_redraw();
                        }
                    },
                    move |err| {
                        eprintln!("Error: {}", err);
                    },
                    None,
                )
                .unwrap();
            stream.play().unwrap();
            Some(stream)
        } else {
            None
        };

        let fft = FftPlanner::new().plan_fft(FFT_SIZE, FftDirection::Forward);
        let scratch_len = fft.get_inplace_scratch_len();
        WindowedApplicationState {
            left_waveform_window: None,
            right_waveform_window: None,
            left_request_redraw,
            right_request_redraw,
            audio_input_ringbuf_cons,
            _stream: stream,
            last_fps_dump_time: Instant::now(),
            frame_count: 0,

            fft,
            fft_inout_buffer: vec![Complex::default(); FFT_SIZE],
            fft_scratch: vec![Complex::default(); scratch_len],
        }
    }

    fn create_waveform_window(
        wgpu_surface: Box<dyn WgpuSurface>,
        anchor_position: PanelAnchorPosition,
    ) -> WaveformWindow {
        // Identity transform is a horizontal waveform scrolling from right to left.
        let rotation = Matrix4::from_angle_z(Rad(-std::f32::consts::FRAC_PI_2));
        let mirror_h = Matrix4::from_nonuniform_scale(-1.0, 1.0, 1.0);
        let mirror_v = Matrix4::from_nonuniform_scale(1.0, -1.0, 1.0);
        let transform_matrix = match anchor_position {
            PanelAnchorPosition::Top => mirror_v,
            PanelAnchorPosition::Bottom => Matrix4::identity(),
            PanelAnchorPosition::Left => rotation * mirror_h,
            PanelAnchorPosition::Right => rotation * mirror_v * mirror_h,
        };
        WaveformWindow::new(wgpu_surface, 0, transform_matrix.into())
    }

    fn configure_left_wgpu_surface(
        &mut self,
        wgpu_surface: Box<dyn WgpuSurface>,
        anchor_position: PanelAnchorPosition,
    ) {
        let waveform_window = Self::create_waveform_window(wgpu_surface, anchor_position);

        *self.left_request_redraw.lock().unwrap() = waveform_window
            .wgpu_surface
            .request_redraw_callback()
            .clone();
        self.left_waveform_window = Some(waveform_window);
    }

    fn configure_right_wgpu_surface(
        &mut self,
        wgpu_surface: Box<dyn WgpuSurface>,
        anchor_position: PanelAnchorPosition,
    ) {
        let waveform_window = Self::create_waveform_window(wgpu_surface, anchor_position);

        *self.right_request_redraw.lock().unwrap() = waveform_window
            .wgpu_surface
            .request_redraw_callback()
            .clone();
        self.right_waveform_window = Some(waveform_window);
    }

    fn render(&mut self) {
        let data: Vec<f32> = self.audio_input_ringbuf_cons.pop_iter().collect();

        if let Some(left_waveform_window) = &mut self.left_waveform_window {
            left_waveform_window.update_config();
            left_waveform_window.process_audio(
                &data,
                self.fft.as_ref(),
                &mut self.fft_inout_buffer,
                &mut self.fft_scratch,
            );
            left_waveform_window.render();
        }
        if let Some(right_waveform_window) = &mut self.right_waveform_window {
            right_waveform_window.update_config();
            right_waveform_window.process_audio(
                &data,
                self.fft.as_ref(),
                &mut self.fft_inout_buffer,
                &mut self.fft_scratch,
            );
            right_waveform_window.render();
        }

        // FPS calculation
        let now = Instant::now();
        self.frame_count += 1;
        if now.duration_since(self.last_fps_dump_time) >= Duration::from_secs(1) {
            println!("FPS: {}", self.frame_count);
            self.frame_count = 0;
            self.last_fps_dump_time = now;
        }
    }
}

impl WlrLayerApplicationHandler for ApplicationState {
    fn initialize_app_state(&mut self) {
        self.windowed_state = Some(WindowedApplicationState::new(self.process_audio.clone()));
    }

    fn configure_left_wgpu_surface(
        &mut self,
        wgpu_surface: Box<dyn WgpuSurface>,
        anchor_position: PanelAnchorPosition,
    ) {
        self.windowed_state
            .as_mut()
            .unwrap()
            .configure_left_wgpu_surface(wgpu_surface, anchor_position);
    }

    fn configure_right_wgpu_surface(
        &mut self,
        wgpu_surface: Box<dyn WgpuSurface>,
        anchor_position: PanelAnchorPosition,
    ) {
        self.windowed_state
            .as_mut()
            .unwrap()
            .configure_right_wgpu_surface(wgpu_surface, anchor_position);
    }

    fn left_resized(&mut self, width: u32, height: u32) {
        if let Some(waveform_window) = self
            .windowed_state
            .as_mut()
            .unwrap()
            .left_waveform_window
            .as_mut()
        {
            waveform_window.reconfigure(width, height);
        }
    }
    fn right_resized(&mut self, width: u32, height: u32) {
        if let Some(waveform_window) = self
            .windowed_state
            .as_mut()
            .unwrap()
            .right_waveform_window
            .as_mut()
        {
            waveform_window.reconfigure(width, height);
        }
    }

    fn render(&mut self) {
        self.windowed_state.as_mut().unwrap().render();
    }
}

impl ApplicationHandler for ApplicationState {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.windowed_state.is_none() {
            self.windowed_state = Some(WindowedApplicationState::new(self.process_audio.clone()));
            self.windowed_state
                .as_mut()
                .unwrap()
                .configure_left_wgpu_surface(
                    Self::create_winit_window_and_wgpu_surface(event_loop),
                    PanelAnchorPosition::Bottom,
                );
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(state) = self.windowed_state.as_mut() {
            match event {
                WindowEvent::Resized(new_size) => {
                    // FIXME: Check the window ID
                    state
                        .left_waveform_window
                        .as_mut()
                        .or(state.right_waveform_window.as_mut())
                        .unwrap()
                        .reconfigure(new_size.width, new_size.height);
                    // state.right_waveform_window.reconfigure(new_size);
                }
                WindowEvent::RedrawRequested => {
                    state.render();
                }
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::KeyboardInput { event, .. } => {
                    if let winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space) =
                        event.physical_key
                    {
                        if event.state == winit::event::ElementState::Pressed {
                            let mut process_audio = self.process_audio.lock().unwrap();
                            *process_audio = !*process_audio;
                            (state
                                .left_waveform_window
                                .as_ref()
                                .unwrap()
                                .wgpu_surface
                                .request_redraw_callback())();
                            // (state.right_waveform_window.wgpu_surface.request_redraw)();
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

slint::include_modules!();

enum UiMessage {
    ApplicationStateCallback(Box<dyn FnOnce(&mut WindowedApplicationState) + Send>),
    SetPanelLayout(PanelConfiguration),
    SetPanelLayer(PanelConfiguration),
    SetPanelWidth(PanelConfiguration),
    SetPanelExclusiveRatio(PanelConfiguration),
}

pub fn main() {
    let (ui_msg_tx, ui_msg_rx) = mpsc::channel::<UiMessage>();
    let window = ConfigurationWindow::new().unwrap();
    let configuration = Configuration::get(&window);
    let panel_configuration = configuration.get_panel();

    let request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>> =
        Arc::new(Mutex::new(Arc::new(|| {})));
    std::thread::spawn({
        let request_redraw_callback = request_redraw_callback.clone();
        move || {
            let mut app_state = ApplicationState::new();

            let mut layers_even_queue = wlr_layers::WlrWaylandEventLoop::new(
                app_state,
                ui_msg_rx,
                panel_configuration,
                request_redraw_callback,
            );
            layers_even_queue.run_event_loop();

            // // This won't work on macOS, but by then let's hope we can render using wgpu directly in a Slint window.
            // let event_loop = EventLoop::builder().with_any_thread(true).build().unwrap();
            // FIXME: Handle ui_msg_rx
            // event_loop.run_app(&mut loop_state).unwrap();
        }
    });

    let send_ui_msg = {
        move |msg| {
            ui_msg_tx.send(msg).unwrap();
            request_redraw_callback.lock().unwrap()();
        }
    };
    configuration.on_changed({
        let send = send_ui_msg.clone();
        let window = window.as_weak();
        move || {
            let window = window.upgrade().unwrap();
            let configuration = Configuration::get(&window);
            let fill_color = configuration.get_fill_color();
            let stroke_color = configuration.get_stroke_color();
            send(UiMessage::ApplicationStateCallback(Box::new(
                move |state| {
                    if let Some(left_waveform_window) = &mut state.left_waveform_window {
                        left_waveform_window.set_fill_color(fill_color);
                        left_waveform_window.set_stroke_color(stroke_color);
                    }
                    if let Some(right_waveform_window) = &mut state.right_waveform_window {
                        right_waveform_window.set_fill_color(fill_color);
                        right_waveform_window.set_stroke_color(stroke_color);
                    }
                },
            )));
        }
    });

    configuration.on_panel_layout_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::SetPanelLayout(config));
        }
    });
    configuration.on_panel_layer_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::SetPanelLayer(config));
        }
    });
    configuration.on_panel_width_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::SetPanelWidth(config));
        }
    });
    configuration.on_panel_exclusive_ratio_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::SetPanelExclusiveRatio(config));
        }
    });

    // Apply the initial configuration from the UI
    configuration.invoke_changed();

    window.run().unwrap();
}
