use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
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

const AUDIO_BUFFER_SIZE: usize = 512;
const VERTEX_BUFFER_SIZE: usize = AUDIO_BUFFER_SIZE * 128;

struct LoopState {
    // See https://docs.rs/winit/latest/winit/changelog/v0_30/index.html#removed
    // for the recommended practice regarding Window creation (from which everything depends)
    // in winit >= 0.30.0.
    // The actual state is in an Option because its initialization is now delayed to after
    // the even loop starts running.
    state: Option<InitializedLoopState>,
    last_frame_time: Instant,
    last_fps_dump_time: Instant,
    frame_count: u32,
    process_audio: Arc<Mutex<bool>>,
}

impl LoopState {
    fn new() -> LoopState {
        LoopState {
            state: None,
            last_frame_time: Instant::now(),
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
    vertex_buffer: Arc<wgpu::Buffer>,
    _y_value_buffer: Arc<wgpu::Buffer>,
    y_value_offset: Arc<Mutex<usize>>,
    y_value_offset_buffer: Arc<wgpu::Buffer>,
    bind_group: wgpu::BindGroup,
    config: wgpu::SurfaceConfiguration,
    _stream: Option<cpal::Stream>,
    audio_sample_rate: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
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
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
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

        let vertices: Vec<Vertex> = (0..VERTEX_BUFFER_SIZE)
            .map(|i| Vertex {
                position: [i as f32 / VERTEX_BUFFER_SIZE as f32 * 2.0 - 1.0, 0.0],
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
        let arc_vertex_buffer = Arc::new(vertex_buffer);
        let arc_y_value_buffer = Arc::new(y_value_buffer);
        let arc_y_value_buffer_clone = arc_y_value_buffer.clone();
        let window_clone = window.clone();
        let y_value_offset = Arc::new(Mutex::new(0));
        let y_value_offset_clone = y_value_offset.clone();
        let arc_y_value_offset_buffer = Arc::new(y_value_offset_buffer);
        let arc_process_audio = process_audio.clone();

        // List all cpal input sources
        let host = cpal::default_host();
        // let devices = host.input_devices().unwrap();
        // for device in devices {
        //     println!("Input device: {}", device.name().unwrap());
        // }

        // Print all input configs for the default input device
        let (stream, audio_sample_rate) = if let Some(default_input_device) =
            host.default_input_device()
        {
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
            let audio_sample_rate = default_config.sample_rate().0 as f32;
            let stream: cpal::Stream = default_input_device
                .build_input_stream(
                    &cpal::StreamConfig {
                        buffer_size: cpal::BufferSize::Fixed(AUDIO_BUFFER_SIZE as u32),
                        ..default_config.into()
                    },
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        let process_audio = arc_process_audio.lock().unwrap();
                        if *process_audio {
                            let y_values: Vec<YValue> =
                                data.iter().map(|&sample| YValue { y: sample }).collect();
                            // FIXME: This is the "read" offset, but the sample callback and the rendering won't always
                            // be in sync. So we'd also need a separate "write" offset here to avoid visibly cutting the sound wave
                            // where we'll be writing.
                            let y_value_offset = y_value_offset_clone.lock().unwrap();

                            // First pass: write to the end of the buffer
                            let first_pass_len = VERTEX_BUFFER_SIZE - *y_value_offset;
                            let first_pass_data = &y_values[..first_pass_len.min(y_values.len())];
                            arc_queue_clone.write_buffer(
                                &arc_y_value_buffer_clone,
                                (*y_value_offset * std::mem::size_of::<YValue>())
                                    as wgpu::BufferAddress,
                                bytemuck::cast_slice(first_pass_data),
                            );

                            // Second pass: write to the beginning of the buffer
                            if first_pass_data.len() < y_values.len() {
                                let second_pass_data = &y_values[first_pass_data.len()..];
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
            (Some(stream), audio_sample_rate)
        } else {
            (None, 0.0)
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
            vertex_buffer: arc_vertex_buffer,
            _y_value_buffer: arc_y_value_buffer,
            y_value_offset,
            y_value_offset_buffer: arc_y_value_offset_buffer,
            bind_group,
            config,
            _stream: stream,
            audio_sample_rate,
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
                    if *self.process_audio.lock().unwrap() {
                        let frame_duration = now.duration_since(self.last_frame_time);
                        self.last_frame_time = now;

                        // Scroll how the y_values are applied onto vertices based on the time between frames.
                        let mut y_value_offset = state.y_value_offset.lock().unwrap();
                        *y_value_offset = (*y_value_offset
                            + (state.audio_sample_rate
                                * (frame_duration.as_micros() as f32 / 1000000.0))
                                as usize)
                            % VERTEX_BUFFER_SIZE;
                        state.queue.write_buffer(
                            &state.y_value_offset_buffer,
                            0,
                            bytemuck::cast_slice(&[*y_value_offset as u32]),
                        );
                    }

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
                    {
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: None,
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color {
                                            r: 0.9,
                                            g: 0.9,
                                            b: 0.9,
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
