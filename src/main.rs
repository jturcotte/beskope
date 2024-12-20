use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::time::{Duration, Instant};
use std::{borrow::Cow, sync::Arc};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

struct LoopState {
    // See https://docs.rs/winit/latest/winit/changelog/v0_30/index.html#removed
    // for the recommended practice regarding Window creation (from which everything depends)
    // in winit >= 0.30.0.
    // The actual state is in an Option because its initialization is now delayed to after
    // the even loop starts running.
    state: Option<InitializedLoopState>,
    last_frame_time: Instant,
    frame_count: u32,
}

impl LoopState {
    fn new() -> LoopState {
        LoopState {
            state: None,
            last_frame_time: Instant::now(),
            frame_count: 0,
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
    config: wgpu::SurfaceConfiguration,
    stream: Option<cpal::Stream>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
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
    async fn new(event_loop: &ActiveEventLoop) -> InitializedLoopState {
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
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
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

        let vertex_buffer: wgpu::Buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(
                    &[Vertex {
                        position: [0.0, 0.0],
                    }; 1024],
                ),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(swapchain_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
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

        let config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        surface.configure(&device, &config);

        let arc_queue = Arc::new(queue);
        let arc_queue_clone = arc_queue.clone();
        let arc_vertex_buffer = Arc::new(vertex_buffer);
        let arc_vertex_buffer_clone = arc_vertex_buffer.clone();
        let window_clone = window.clone();

        // List all cpal input sources
        let host = cpal::default_host();
        let devices = host.input_devices().unwrap();
        for device in devices {
            println!("Input device: {}", device.name().unwrap());
        }

        // Print all input configs for the default input device
        let stream = if let Some(default_input_device) = host.default_input_device() {
            println!(
                "Default input device: {}",
                default_input_device.name().unwrap()
            );
            let configs = default_input_device.supported_input_configs().unwrap();
            for config in configs {
                println!("Supported input config: {:?}", config);
            }

            // Modify the data_callback to update the vertex buffer directly and trigger a redraw
            let default_config = default_input_device.default_input_config().unwrap();
            let stream: cpal::Stream = default_input_device
                .build_input_stream(
                    &default_config.into(),
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        let vertices: Vec<Vertex> = data
                            .iter()
                            .enumerate()
                            .map(|(i, &sample)| {
                                let position = [i as f32 / 1024.0 * 2.0 - 1.0, sample];
                                // println!("Sample: {:?}", position);
                                Vertex { position }
                            })
                            .take(1024)
                            .collect();
                        arc_queue_clone.write_buffer(
                            &arc_vertex_buffer_clone,
                            0,
                            bytemuck::cast_slice(&vertices),
                        );
                        window_clone.request_redraw();
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
            vertex_buffer: arc_vertex_buffer,
            config,
            stream,
        }
    }
}

impl ApplicationHandler for LoopState {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.state = Some(pollster::block_on(InitializedLoopState::new(event_loop)));
            }
            #[cfg(target_arch = "wasm32")]
            {
                self.state = Some(wasm_bindgen_futures::spawn_local(async move {
                    InitializedLoopState::new(event_loop).await
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
                        render_pass.draw(0..1024 /* vertices.len() as u32 */, 0..1);
                    }

                    state.queue.submit(Some(encoder.finish()));
                    frame.present();

                    // FPS calculation
                    self.frame_count += 1;
                    let now = Instant::now();
                    if now.duration_since(self.last_frame_time) >= Duration::from_secs(1) {
                        println!("FPS: {}", self.frame_count);
                        self.frame_count = 0;
                        self.last_frame_time = now;
                    }
                }
                WindowEvent::CloseRequested => event_loop.exit(),
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
