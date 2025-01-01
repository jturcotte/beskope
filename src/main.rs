use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use num_complex::Complex;
use ringbuf::traits::{Consumer, Observer, RingBuffer};
use ringbuf::HeapRb;
use rustfft::{Fft, FftDirection, FftPlanner};
use splines::Interpolation;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use std::{borrow::Cow, sync::Arc};
use wgpu::util::DeviceExt;
use wgpu::BufferUsages;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const VERTEX_BUFFER_SIZE: usize = 44100 * 3;

struct LoopState {
    // See https://docs.rs/winit/latest/winit/changelog/v0_30/index.html#removed
    // for the recommended practice regarding Window creation (from which everything depends)
    // in winit >= 0.30.0.
    // The actual state is in an Option because its initialization is now delayed to after
    // the even loop starts running.
    state: Option<InitializedLoopState>,
    last_fps_dump_time: Instant,
    frame_count: u32,
    process_audio: Arc<Mutex<bool>>,
}

impl LoopState {
    fn new() -> LoopState {
        LoopState {
            state: None,
            last_fps_dump_time: Instant::now(),
            frame_count: 0,
            process_audio: Arc::new(Mutex::new(true)),
        }
    }
}

struct InitializedLoopState {
    window: Arc<Window>,
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: Arc<wgpu::Queue>,
    _shader: wgpu::ShaderModule,
    _pipeline_layout: wgpu::PipelineLayout,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    _y_value_buffer: Arc<wgpu::Buffer>,
    y_value_write_offset: Arc<Mutex<usize>>,
    y_value_offset_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    config: wgpu::SurfaceConfiguration,
    _stream: Option<cpal::Stream>,
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

impl InitializedLoopState {
    async fn new(
        event_loop: &ActiveEventLoop,
        process_audio: Arc<Mutex<bool>>,
    ) -> InitializedLoopState {
        #[allow(unused_mut)]
        let mut attributes = Window::default_attributes();
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = web_sys::window()
                .unwrap()
                .document()
                .unwrap()
                .get_element_by_id("canvas")
                .unwrap()
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .unwrap();
            builder = builder.with_canvas(Some(canvas));
        }
        let window = Arc::new(event_loop.create_window(attributes).unwrap());

        let mut size = window.inner_size();
        size.width = size.width.max(1);
        size.height = size.height.max(1);
        println!("Window size: {:?}", size);

        let instance = wgpu::Instance::default();

        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, Some(&surface))
            .await
            .expect("Failed to find an appropriate adapter");

        // Create the logical device and command queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                    required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        // Load the shaders from disk
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        // Define control points for the spline used to make the waveform look more compressed for older samples.
        let cps = vec![
            (0.0, 0.0, Interpolation::Linear),
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
                Interpolation::Linear,
            ),
            (VERTEX_BUFFER_SIZE as f32, 1.0, Interpolation::Linear),
        ];

        // Create the spline
        let spline = splines::Spline::from_vec(
            cps.iter()
                .map(|(x, y, interpolation)| splines::Key::new(*x, *y, *interpolation))
                .collect(),
        );

        // Sample the spline for every waveform vertex so that the position of older samples are closer together
        let vertices: Vec<Vertex> = (0..VERTEX_BUFFER_SIZE)
            .map(|i| Vertex {
                position: [spline.sample(i as f32).unwrap() * 2.0 - 1.0, 0.0],
                waveform_index: i as u32,
                should_offset: 1.0,
            })
            .collect();
        let y_values: Vec<YValue> = vec![YValue { y: 0.0 }; VERTEX_BUFFER_SIZE];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
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
            ],
            label: Some("Bind Group"),
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 1,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let arc_queue = Arc::new(queue);
        let arc_queue_clone = arc_queue.clone();
        let arc_y_value_buffer = Arc::new(y_value_buffer);
        let arc_y_value_buffer_clone = arc_y_value_buffer.clone();
        let window_clone = window.clone();
        let y_value_write_offset = Arc::new(Mutex::new(0));
        let y_value_write_offset_clone = y_value_write_offset.clone();
        let arc_process_audio = process_audio.clone();
        let audio_ringbuf = Arc::new(Mutex::new(HeapRb::<f32>::new(1024)));
        let audio_ringbuf_clone = audio_ringbuf.clone();
        let mut fft_input_ringbuf = HeapRb::<f32>::new(1024);
        let mut audio_ringbuf_local_unused = HeapRb::<f32>::new(1024);

        let fft: Arc<dyn Fft<f32>> =
            FftPlanner::new().plan_fft(fft_input_ringbuf.capacity().get(), FftDirection::Forward);
        let mut fft_inout_buffer = vec![Complex::default(); fft.len()];
        let mut fft_scratch = vec![Complex::default(); fft.get_inplace_scratch_len()];

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
                            // Keep track of the last FFT size samples.
                            // Append them in reverse because phase_to_samples currently finds the samples phase from the beginning of the input buffer.
                            fft_input_ringbuf
                                .push_iter_overwrite(data.iter().rev().step_by(2).copied());

                            let phase_samples = if !fft_input_ringbuf.is_full() {
                                0
                            } else {
                                // Run an FFT on the accumulated latest FFT length samples as a way to find the peak frequency
                                // and align the end of our waveform at the end of the vertex attribute buffer so that the eye
                                // isn't totally lost frame over frame.
                                // FIXME: It's wasteful to run this on every audio callback, but it keeps things a bit simpler for now.
                                let (first, second) = fft_input_ringbuf.as_slices();
                                fft_inout_buffer
                                    .iter_mut()
                                    .zip(first.iter().chain(second.iter()))
                                    .for_each(|(dst, &y)| *dst = Complex::new(y, 0.));

                                fft.process_with_scratch(&mut fft_inout_buffer, &mut fft_scratch);

                                // Find the peak frequency
                                let peak_frequency_index = fft_inout_buffer
                                    .iter()
                                    .take(fft_inout_buffer.len() / 2)
                                    .enumerate()
                                    // Skipping k=0 makes sense as it doesn't really capture oscillations, but until where does it make sense?
                                    .skip(4)
                                    .max_by(|(_, a), (_, b)| {
                                        a.norm()
                                            .partial_cmp(&b.norm())
                                            .unwrap_or(std::cmp::Ordering::Equal)
                                    })
                                    .map(|(i, _)| i)
                                    .unwrap_or(0);

                                fn phase_to_samples(
                                    phase: f32,
                                    k: usize,
                                    fft_size: usize,
                                ) -> usize {
                                    // When e.g. k=2, the FFT identifies an oscillation that repeats 2 times in the FFT window.
                                    // To find the phase shift in samples, find where the phase in radians corresponds vs the FFT buffer size.
                                    ((phase + std::f32::consts::PI) / (2.0 * std::f32::consts::PI)
                                        * fft_size as f32
                                        / k as f32) as usize
                                }

                                // To be able to perform the inverse FFT, each frequency bin also has a phase.
                                // Use this phase to align the waveform to the end of the buffer.
                                // This here is the sine phase shift in radians.
                                let phase_shift = -fft_inout_buffer[peak_frequency_index].arg();

                                phase_to_samples(
                                    phase_shift,
                                    peak_frequency_index,
                                    fft_inout_buffer.len(),
                                )
                            };

                            let mut data_iter = data.iter().step_by(2).copied();
                            let data_iter_len = data_iter.len();
                            let unused_data_iter_len =
                                data_iter_len + audio_ringbuf_local_unused.occupied_len();
                            let y_values: Vec<YValue> = if phase_samples > unused_data_iter_len {
                                Vec::new()
                            } else {
                                audio_ringbuf_local_unused
                                    .pop_iter()
                                    .chain(data_iter.by_ref())
                                    .take(unused_data_iter_len - phase_samples)
                                    .map(|sample| YValue { y: sample })
                                    .collect()
                            };
                            audio_ringbuf_local_unused.push_iter_overwrite(data_iter);

                            // First pass: write to the end of the buffer
                            let first_pass_len = {
                                let mut y_value_write_offset_lock =
                                    y_value_write_offset_clone.lock().unwrap();
                                let first_pass_len =
                                    VERTEX_BUFFER_SIZE - *y_value_write_offset_lock;
                                let first_pass_data =
                                    &y_values[..first_pass_len.min(y_values.len())];
                                arc_queue_clone.write_buffer(
                                    &arc_y_value_buffer_clone,
                                    (*y_value_write_offset_lock * std::mem::size_of::<YValue>())
                                        as wgpu::BufferAddress,
                                    bytemuck::cast_slice(first_pass_data),
                                );

                                // Update the write offset
                                *y_value_write_offset_lock = (*y_value_write_offset_lock
                                    + y_values.len())
                                    % VERTEX_BUFFER_SIZE;

                                // Overwrite the latest data's ring buffer with the latest audio data to
                                // let the render thread process it before rendering.
                                audio_ringbuf_clone
                                    .lock()
                                    .unwrap()
                                    .push_iter_overwrite(data.iter().step_by(2).copied());

                                first_pass_len
                            };

                            // Second pass: write to the beginning of the buffer
                            if first_pass_len < y_values.len() {
                                let second_pass_data = &y_values[first_pass_len..];
                                arc_queue_clone.write_buffer(
                                    &arc_y_value_buffer_clone,
                                    0,
                                    bytemuck::cast_slice(second_pass_data),
                                );
                            }

                            window_clone.request_redraw();
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

        InitializedLoopState {
            window,
            _instance: instance,
            surface,
            _adapter: adapter,
            device,
            queue: arc_queue,
            _shader: shader,
            _pipeline_layout: pipeline_layout,
            render_pipeline,
            vertex_buffer,
            _y_value_buffer: arc_y_value_buffer,
            y_value_write_offset,
            y_value_offset_buffer,
            bind_group,
            config,
            _stream: stream,
        }
    }
}

impl ApplicationHandler for LoopState {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.state = Some(pollster::block_on(InitializedLoopState::new(
                    event_loop,
                    self.process_audio.clone(),
                )));
            }
            #[cfg(target_arch = "wasm32")]
            {
                self.state = Some(wasm_bindgen_futures::spawn_local(async move {
                    InitializedLoopState::new(event_loop, self.process_audio.clone()).await
                }));
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(state) = self.state.as_mut() {
            match event {
                WindowEvent::Resized(new_size) => {
                    // Reconfigure the surface with the new size
                    state.config.width = new_size.width.max(1);
                    state.config.height = new_size.height.max(1);
                    state.surface.configure(&state.device, &state.config);
                    // On macos the window needs to be redrawn manually after resizing
                    state.window.request_redraw();
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let frame = state
                        .surface
                        .get_current_texture()
                        .expect("Failed to acquire next swap chain texture");
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    let mut encoder = state
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                    // Tell the shader how to read the audio data ring buffer
                    let last_y_value_write_offset = *state.y_value_write_offset.lock().unwrap();
                    state.queue.write_buffer(
                        &state.y_value_offset_buffer,
                        0,
                        bytemuck::cast_slice(&[last_y_value_write_offset as u32]),
                    );

                    {
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: None,
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color {
                                            r: 0.6,
                                            g: 0.6,
                                            b: 0.6,
                                            a: 1.0,
                                        }),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                        render_pass.set_pipeline(&state.render_pipeline);
                        render_pass.set_vertex_buffer(0, state.vertex_buffer.slice(..));
                        render_pass.set_bind_group(0, &state.bind_group, &[]);
                        render_pass.draw(0..VERTEX_BUFFER_SIZE as u32, 0..1);
                    }

                    state.queue.submit(Some(encoder.finish()));
                    frame.present();

                    // FPS calculation
                    self.frame_count += 1;
                    if now.duration_since(self.last_fps_dump_time) >= Duration::from_secs(1) {
                        println!("FPS: {}", self.frame_count);
                        self.frame_count = 0;
                        self.last_fps_dump_time = now;
                    }

                    state.window.request_redraw();
                }
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::KeyboardInput { event, .. } => {
                    if let winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space) =
                        event.physical_key
                    {
                        if event.state == winit::event::ElementState::Pressed {
                            let mut process_audio = self.process_audio.lock().unwrap();
                            *process_audio = !*process_audio;
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

pub fn main() {
    env_logger::init();

    let mut loop_state = LoopState::new();
    let event_loop = EventLoop::new().unwrap();

    log::info!("Entering event loop...");
    event_loop.run_app(&mut loop_state).unwrap();
}
