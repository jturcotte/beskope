// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use clap::Parser;
use interprocess::local_socket::{GenericNamespaced, ListenerOptions, prelude::*};
use num_complex::Complex;
use ringbuf::storage::Heap;
use ringbuf::traits::{Consumer, Split};
use ringbuf::wrap::caching::Caching;
use ringbuf::{HeapRb, SharedRb};
use rustfft::{Fft, FftDirection, FftPlanner};
use slint::ComponentHandle;
use std::collections::HashSet;
use std::io::prelude::*;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use views::WaveformView;
use wayland_client::{Connection, QueueHandle};
// use wgpu::TextureFormat;
use wlr_layers::{PanelAnchorPosition, WlrWaylandEventHandler};

use crate::ui::RIDGELINE_WIDTH;
use crate::wlr_layers::WlrCanvasContext;

mod audio;
mod ui;
mod views;
mod wlr_layers;

const VERTEX_BUFFER_SIZE: usize = 44100 * 3;
const FFT_SIZE: usize = 2048;
const NUM_CHANNELS: usize = 2;

pub trait WgpuSurface {
    // fn adapter(&self) -> &wgpu::Adapter;
    fn device(&self) -> &wgpu::Device;
    // fn surface_texture(&self) -> &wgpu::SurfaceTexture;
    fn queue(&self) -> &Arc<wgpu::Queue>;
    fn surface_id(&self) -> u32;
    fn swapchain_format(&self) -> Option<wgpu::TextureFormat>;
}

struct WaveformWindow {
    surface_id: u32,
    wgpu: Rc<dyn WgpuSurface>,
    depth_texture: Option<wgpu::Texture>,
    // config: wgpu::SurfaceConfiguration,
    // swapchain_format: TextureFormat,
    render_window: RenderWindow,
    // FIXME: Remove
    must_reconfigure: bool,
    last_configure_size: (u32, u32),
    last_fps_dump_time: Instant,
    frame_count: u32,
    fps_callback: Box<dyn Fn(u32)>,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum RenderWindow {
    Primary,
    Secondary,
}

impl WaveformWindow {
    fn new(
        wgpu: &Rc<dyn WgpuSurface>,
        render_window: RenderWindow,
        fps_callback: Box<dyn Fn(u32)>,
    ) -> WaveformWindow {
        // let swapchain_capabilities = wgpu.surface().get_capabilities(wgpu.adapter());
        // let swapchain_format = swapchain_capabilities.formats[0];

        // let config = wgpu::SurfaceConfiguration {
        //     usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        //     format: swapchain_format,
        //     width: width.max(1),
        //     height: height.max(1),
        //     present_mode: wgpu::PresentMode::AutoVsync,
        //     desired_maximum_frame_latency: 1,
        //     alpha_mode: wgpu::CompositeAlphaMode::PreMultiplied,
        //     view_formats: vec![],
        // };

        WaveformWindow {
            surface_id: wgpu.surface_id(),
            wgpu: wgpu.clone(),
            // config,
            depth_texture: None,
            // swapchain_format: wgpu
            //     .swapchain_format()
            //     .expect("WaveformWindow::new should be on configured surface"),
            // swapchain_format,
            render_window,
            must_reconfigure: true,
            // Force reconfiguration of the depth texture on the first render
            last_configure_size: (u32::MAX, u32::MAX),
            last_fps_dump_time: Instant::now(),
            frame_count: 0,
            fps_callback,
        }
    }

    fn surface_id(&self) -> u32 {
        self.surface_id
    }

    // FIXME: Pass as a flag to render?
    fn reconfigure(&mut self) {
        // Reconfigure the surface with the new size
        // self.config.width = width.max(1);
        // self.config.height = height.max(1);
        self.must_reconfigure = true;
    }

    fn render(
        &mut self,
        wgpu: &Rc<dyn WgpuSurface>,
        surface_texture: &wgpu::SurfaceTexture,
        left_waveform_view: &mut Option<Box<dyn WaveformView>>,
        right_waveform_view: &mut Option<Box<dyn WaveformView>>,
    ) {
        if self.last_configure_size
            != (
                surface_texture.texture.width(),
                surface_texture.texture.height(),
            )
        {
            // wgpu.surface().configure(wgpu.device(), &self.config);

            // Create the depth texture
            self.depth_texture = Some(wgpu.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width: surface_texture.texture.width(),
                    height: surface_texture.texture.height(),
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

        // let frame = wgpu
        //     .surface()
        //     .get_current_texture()
        //     .expect("Failed to acquire next swap chain texture");
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture_view = self
            .depth_texture
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = wgpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Clear the depth texture before rendering the views
        let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Clear Depth Pass"),
            color_attachments: &[],
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

        if let Some(waveform_view) = left_waveform_view {
            if waveform_view.render_window() == self.render_window {
                waveform_view.render(&mut encoder, &view, &depth_texture_view);
            }
        }
        if let Some(waveform_view) = right_waveform_view {
            if waveform_view.render_window() == self.render_window {
                waveform_view.render(&mut encoder, &view, &depth_texture_view);
            }
        }
        wgpu.queue().submit(Some(encoder.finish()));
        // frame.present();

        let now = Instant::now();
        self.frame_count += 1;
        if now.duration_since(self.last_fps_dump_time) >= Duration::from_secs(1) {
            (self.fps_callback)(self.frame_count);
            self.frame_count = 0;
            self.last_fps_dump_time = now;
        }
    }
}

struct ApplicationState {
    pub config: ui::Configuration,
    pub lazy_config_changes: HashSet<usize>,
    primary_waveform_window: Option<(WaveformWindow, PanelAnchorPosition)>,
    secondary_waveform_window: Option<(WaveformWindow, PanelAnchorPosition)>,
    last_non_zero_sample_age: usize,
    animation_stopped: Arc<AtomicBool>,
    pub left_waveform_view: Option<Box<dyn WaveformView>>,
    right_waveform_view: Option<Box<dyn WaveformView>>,
    screen_size: (u32, u32),
    audio_input_ringbuf_cons: Option<Caching<Arc<SharedRb<Heap<f32>>>, false, true>>,
    fft: Option<Arc<dyn Fft<f32>>>,
    fft_inout_buffer: Option<Vec<Complex<f32>>>,
    fft_scratch: Option<Vec<Complex<f32>>>,
    request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
    config_window: slint::Weak<ui::ConfigurationWindow>,
}

impl ApplicationState {
    fn new(
        config: ui::Configuration,
        request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
        config_window: slint::Weak<ui::ConfigurationWindow>,
    ) -> ApplicationState {
        ApplicationState {
            config,
            lazy_config_changes: HashSet::new(),
            primary_waveform_window: None,
            secondary_waveform_window: None,
            last_non_zero_sample_age: 0,
            animation_stopped: Arc::new(AtomicBool::new(false)),
            left_waveform_view: None,
            right_waveform_view: None,
            screen_size: (1, 1),
            audio_input_ringbuf_cons: None,
            fft: None,
            fft_inout_buffer: None,
            fft_scratch: None,
            request_redraw_callback,
            config_window,
        }
    }

    pub fn reload_configuration(&mut self) {
        let config = match ui::Configuration::load() {
            Ok(config) => config,
            Err(e) => {
                eprintln!("Failed to load configuration, will use default: {}", e);
                ui::Configuration::default()
            }
        };
        self.config = config;
    }

    fn initialize_audio_and_fft(&mut self) {
        let (audio_input_ringbuf_prod, audio_input_ringbuf_cons) =
            HeapRb::<f32>::new(44100 * NUM_CHANNELS).split();

        std::thread::spawn({
            let animation_stopped = self.animation_stopped.clone();
            let request_redraw_callback = self.request_redraw_callback.clone();
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

        self.audio_input_ringbuf_cons = Some(audio_input_ringbuf_cons);
        self.fft = Some(fft);
        self.fft_inout_buffer = Some(vec![Complex::default(); FFT_SIZE]);
        self.fft_scratch = Some(vec![Complex::default(); scratch_len]);
    }

    fn create_waveform_view(
        &self,
        wgpu: &Rc<dyn WgpuSurface>,
        style: ui::Style,
        render_window: RenderWindow,
        is_left_channel: bool,
    ) -> Box<dyn WaveformView> {
        let (_, anchor_position) = match render_window {
            RenderWindow::Primary => self.primary_waveform_window.as_ref().unwrap(),
            RenderWindow::Secondary => self.secondary_waveform_window.as_ref().unwrap(),
        };
        let device = wgpu.device();
        let queue = wgpu.queue();
        let swapchain_format = wgpu.swapchain_format().unwrap();
        let mut view: Box<dyn WaveformView> = match style {
            ui::Style::Ridgeline => Box::new(views::RidgelineWaveformView::new(
                device,
                queue,
                swapchain_format,
                render_window,
                *anchor_position,
                self.config.general.channels,
                is_left_channel,
            )),
            ui::Style::Compressed => Box::new(views::CompressedWaveformView::new(
                device,
                queue,
                swapchain_format,
                render_window,
                *anchor_position,
                self.config.general.channels,
                is_left_channel,
            )),
        };
        view.set_screen_size(self.screen_size.0, self.screen_size.1);
        view.apply_lazy_config_changes(&self.config, None);
        view
    }

    // fn create_waveform_view_no_window(
    //     &self,
    //     style: ui::Style,
    //     is_left_channel: bool,
    //     device: &wgpu::Device,
    //     queue: &Arc<wgpu::Queue>,
    //     swapchain_format: TextureFormat,
    // ) -> Box<dyn WaveformView> {
    //     // let device = window.wgpu.device();
    //     // let queue = window.wgpu.queue();
    //     // let swapchain_format = window.swapchain_format;
    //     let mut view: Box<dyn WaveformView> = match style {
    //         ui::Style::Ridgeline => Box::new(views::RidgelineWaveformView::new(
    //             device,
    //             queue,
    //             swapchain_format,
    //             RenderWindow::Primary,
    //             PanelAnchorPosition::Bottom,
    //             self.config.general.channels,
    //             is_left_channel,
    //         )),
    //         ui::Style::Compressed => Box::new(views::CompressedWaveformView::new(
    //             device,
    //             queue,
    //             swapchain_format,
    //             RenderWindow::Primary,
    //             PanelAnchorPosition::Bottom,
    //             self.config.general.channels,
    //             is_left_channel,
    //         )),
    //     };
    //     view.set_screen_size(self.screen_size.0, self.screen_size.1);
    //     view.apply_lazy_config_changes(&self.config, None);
    //     view
    // }

    fn configure_primary_wgpu_surface(
        &mut self,
        wgpu: &Rc<dyn WgpuSurface>,
        anchor_position: PanelAnchorPosition,
    ) {
        let config_window = self.config_window.clone();
        let fps_callback = Box::new(move |fps: u32| {
            config_window
                .upgrade_in_event_loop(move |window| {
                    window.set_primary_fps(fps as i32);
                })
                .unwrap();
        });

        let window = WaveformWindow::new(wgpu, RenderWindow::Primary, fps_callback);
        self.primary_waveform_window = Some((window, anchor_position));

        self.left_waveform_view =
            Some(self.create_waveform_view(wgpu, self.config.style, RenderWindow::Primary, true));

        if self.config.general.channels == ui::RenderChannels::Both {
            self.right_waveform_view = Some(self.create_waveform_view(
                wgpu,
                self.config.style,
                RenderWindow::Primary,
                false,
            ));
        }
    }

    fn configure_secondary_wgpu_surface(
        &mut self,
        wgpu_surface: &Rc<dyn WgpuSurface>,
        anchor_position: PanelAnchorPosition,
    ) {
        let config_window = self.config_window.clone();
        let fps_callback = Box::new(move |fps: u32| {
            config_window
                .upgrade_in_event_loop(move |window| {
                    window.set_secondary_fps(fps as i32);
                })
                .unwrap();
        });

        let window = WaveformWindow::new(wgpu_surface, RenderWindow::Secondary, fps_callback);
        self.secondary_waveform_window = Some((window, anchor_position));

        self.right_waveform_view = Some(self.create_waveform_view(
            wgpu_surface,
            self.config.style,
            RenderWindow::Secondary,
            false,
        ));
    }

    fn primary_resized(&mut self) {
        if let Some((waveform_window, _)) = self.primary_waveform_window.as_mut() {
            waveform_window.reconfigure();
        }
    }
    fn secondary_resized(&mut self) {
        if let Some((waveform_window, _)) = self.secondary_waveform_window.as_mut() {
            waveform_window.reconfigure();
        }
    }

    fn process_audio(&mut self, timestamp: u32) {
        let data: Vec<f32> = self
            .audio_input_ringbuf_cons
            .as_mut()
            .unwrap()
            .pop_iter()
            .collect();
        if data.iter().all(|&x| x == 0.0) {
            self.last_non_zero_sample_age += data.len();
            if self.last_non_zero_sample_age > VERTEX_BUFFER_SIZE * 2 {
                // Stop requesting new frames and let the audio thread know if they
                // should wake us up once non-zero samples are available.
                self.animation_stopped.store(true, Ordering::Relaxed);

                // Hide the FPS counter
                self.config_window
                    .upgrade_in_event_loop(move |window| {
                        window.set_primary_fps(0);
                        window.set_secondary_fps(0);
                    })
                    .unwrap();
            }
        } else {
            self.last_non_zero_sample_age = 0;
        }

        let fft = self.fft.as_ref().unwrap();
        let fft_inout_buffer = self.fft_inout_buffer.as_mut().unwrap();
        let fft_scratch = self.fft_scratch.as_mut().unwrap();

        if let Some(left_waveform_window) = &mut self.left_waveform_view {
            left_waveform_window.process_audio(
                timestamp,
                &data,
                fft.as_ref(),
                fft_inout_buffer,
                fft_scratch,
            );
        }
        if let Some(right_waveform_window) = &mut self.right_waveform_view {
            right_waveform_window.process_audio(
                timestamp,
                &data,
                fft.as_ref(),
                fft_inout_buffer,
                fft_scratch,
            );
        }
    }
    fn render(&mut self, wgpu: &Rc<dyn WgpuSurface>, surface_texture: &wgpu::SurfaceTexture) {
        if !self.lazy_config_changes.is_empty() {
            if let Some(waveform_view) = self.left_waveform_view.as_mut() {
                waveform_view
                    .apply_lazy_config_changes(&self.config, Some(&self.lazy_config_changes));
            }
            if let Some(waveform_view) = self.right_waveform_view.as_mut() {
                waveform_view
                    .apply_lazy_config_changes(&self.config, Some(&self.lazy_config_changes));
            }
        }

        if let Some((window, _)) = self.primary_waveform_window.as_mut() {
            if window.surface_id() == wgpu.surface_id() {
                window.render(
                    wgpu,
                    surface_texture,
                    &mut self.left_waveform_view,
                    &mut self.right_waveform_view,
                );
            }
        }

        if let Some((window, _)) = self.secondary_waveform_window.as_mut() {
            if window.surface_id() == wgpu.surface_id() {
                window.render(
                    wgpu,
                    surface_texture,
                    &mut self.left_waveform_view,
                    &mut self.right_waveform_view,
                );
            }
        }

        // Restart tracking changed from the UI to apply on the next frame.
        self.lazy_config_changes.clear();
    }

    fn set_screen_size(&mut self, width: u32, height: u32) {
        self.screen_size.0 = width;
        self.screen_size.1 = height;
        if let Some(view) = &mut self.left_waveform_view {
            view.set_screen_size(width, height);
        }
        if let Some(view) = &mut self.right_waveform_view {
            view.set_screen_size(width, height);
        }
    }

    pub fn recreate_views(&mut self) {
        if let Some((window, _)) = self.primary_waveform_window.as_ref() {
            self.left_waveform_view = Some(self.create_waveform_view(
                &window.wgpu,
                self.config.style,
                RenderWindow::Primary,
                true,
            ));

            if self.config.general.channels == ui::RenderChannels::Both {
                self.right_waveform_view = Some(self.create_waveform_view(
                    &window.wgpu,
                    self.config.style,
                    RenderWindow::Primary,
                    false,
                ));
            } else {
                self.right_waveform_view = None;
            }
        }
        if let Some((window, _)) = self.secondary_waveform_window.as_ref() {
            self.right_waveform_view = Some(self.create_waveform_view(
                &window.wgpu,
                self.config.style,
                RenderWindow::Secondary,
                false,
            ));
        }
    }
}

enum AppMessage {
    ApplicationStateCallback(Box<dyn FnOnce(&mut ApplicationState) + Send>),
    WlrWaylandEventHandlerCallback(
        Box<
            dyn FnOnce(
                    &mut WlrWaylandEventHandler,
                    &Connection,
                    &QueueHandle<WlrWaylandEventHandler>,
                ) + Send,
        >,
    ),
    WlrGlobalCanvasCallback(
        Box<dyn FnOnce(&mut dyn GlobalCanvas<Context = WlrCanvasContext>, WlrCanvasContext) + Send>,
    ),
}

trait GlobalCanvas {
    type Context;
    fn app_state(&mut self) -> &mut ApplicationState;
    fn apply_panel_width_change(&mut self);
    fn apply_panel_exclusive_ratio_change(&mut self);
    fn apply_panel_layout(&mut self, context: &Self::Context);
    fn set_panel_layer(&mut self, layer: ui::PanelLayer);
}

impl AppMessage {
    pub fn to_app<F>(f: F) -> Self
    where
        F: FnOnce(&mut ApplicationState) + Send + 'static,
    {
        AppMessage::ApplicationStateCallback(Box::new(f))
    }

    pub fn to_event_handler2<F>(f: F) -> Self
    where
        F: FnOnce(&mut WlrWaylandEventHandler, &Connection, &QueueHandle<WlrWaylandEventHandler>)
            + Send
            + 'static,
    {
        AppMessage::WlrWaylandEventHandlerCallback(Box::new(f))
    }

    // Crap this can't be compile time, it needs a box dyn
    pub fn to_event_handler<F>(f: F) -> Self
    where
        F: FnOnce(&mut dyn GlobalCanvas<Context = WlrCanvasContext>, WlrCanvasContext)
            + Send
            + 'static,
    {
        AppMessage::WlrGlobalCanvasCallback(Box::new(f))
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Render in a normal top-level window.
    #[arg(short = 'w', long = "window")]
    window: bool,

    /// Show the configuration window for any running instances and exit.
    /// If no instance is running, start the application with the configuration window open.
    #[arg(short = 'c', long = "config")]
    config: bool,

    /// Quit the running instance of beskope and exit.
    #[arg(short = 'q', long = "quit")]
    quit: bool,
}

const CONFIG_COMMAND: &[u8] = b"config";
const QUIT_COMMAND: &[u8] = b"quit";

pub struct SlintWgpuSurface {
    // adapter: wgpu::Adapter,
    device: wgpu::Device,
    // surface: wgpu::Surface<'static>,
    queue: Arc<wgpu::Queue>,
    surface_configuration: Option<wgpu::SurfaceConfiguration>,
    // pub layer: LayerSurface,
}

impl WgpuSurface for SlintWgpuSurface {
    // fn adapter(&self) -> &wgpu::Adapter {
    //     &self.adapter
    // }

    fn device(&self) -> &wgpu::Device {
        &self.device
    }

    // fn surface(&self) -> &wgpu::Surface<'static> {
    //     &self.surface
    // }

    fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    fn surface_id(&self) -> u32 {
        0 // Only one window will be used
    }

    fn swapchain_format(&self) -> Option<wgpu::TextureFormat> {
        self.surface_configuration
            .as_ref()
            .map(|config| config.format)
    }
}

pub fn main() {
    let args = Args::parse();
    let socket_path = "beskope.sock".to_ns_name::<GenericNamespaced>().unwrap();

    let mut show_config = false;
    if args.config || args.quit {
        // Try to connect to the socket and send a config message
        use interprocess::local_socket::traits::Stream;
        match interprocess::local_socket::Stream::connect(socket_path.clone()) {
            Ok(mut stream) => {
                let command = match args {
                    Args { config: true, .. } => CONFIG_COMMAND,
                    Args { quit: true, .. } => QUIT_COMMAND,
                    _ => unreachable!(),
                };
                let _ = stream.write_all(command);
                // Exit after sending the command
                return;
            }
            Err(_) => {
                // No running instance, show the config window in the new instance if requested
                if args.config {
                    show_config = true;
                } else if args.quit {
                    eprintln!("No running instance to quit.");
                    std::process::exit(1);
                }
            }
        }
    }

    let (app_msg_tx, app_msg_rx) = mpsc::channel::<AppMessage>();
    let request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>> =
        Arc::new(Mutex::new(Arc::new(|| {})));

    let send_app_msg = {
        let app_msg_tx = app_msg_tx.clone();
        let request_redraw_callback = request_redraw_callback.clone();
        move |msg| {
            app_msg_tx.send(msg).unwrap();
            request_redraw_callback.lock().unwrap()();
        }
    };

    let config = match ui::Configuration::load() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Failed to load configuration: {}", e);
            ui::Configuration::default()
        }
    };
    let config_window = ui::init(send_app_msg.clone());
    config_window.update_from_configuration(&config);
    if show_config {
        config_window.show().unwrap();
    }

    // Spawn the wlr panel rendering in a separate thread, this is supported with wayland
    if args.window {
        let config_window_weak = config_window.as_weak();
        let request_redraw_callback = request_redraw_callback.clone();
        let config = config.clone();
        std::thread::spawn(move || {
            let app_state =
                ApplicationState::new(config, request_redraw_callback.clone(), config_window_weak);

            let mut layers_even_queue = wlr_layers::WlrWaylandEventLoop::new(
                app_state,
                app_msg_rx,
                request_redraw_callback,
            );
            layers_even_queue.run_event_loop();
        });
    }

    // Listen for toggle commands on the local socket in a background thread
    match ListenerOptions::new().name(socket_path).create_sync() {
        Ok(listener) => {
            let config_window_weak = config_window.as_weak();
            thread::spawn(move || {
                for mut conn in listener.incoming().flatten() {
                    let mut buf = [0u8; 16];
                    if let Ok(n) = conn.read(&mut buf) {
                        match &buf[..n] {
                            CONFIG_COMMAND => {
                                let config_window_weak = config_window_weak.clone();
                                send_app_msg(AppMessage::to_app(move |app| {
                                    let config = app.config.clone();

                                    config_window_weak
                                        .upgrade_in_event_loop(move |window| {
                                            window.update_from_configuration(&config);
                                            window.show().unwrap();
                                        })
                                        .unwrap();
                                }));
                            }
                            QUIT_COMMAND => std::process::exit(0),
                            _ => eprintln!("Received unknown command: {:?}", &buf[..n]),
                        }
                    }
                }
            });
        }
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            eprintln!(
                "Warning: beskope.sock file is occupied, possibly due to another instance running, \
                --toggle will not work for this one."
            );
        }
        Err(e) => {
            eprintln!("Failed to bind to socket: {e}");
            std::process::exit(1);
        }
    };

    let test_window = ui::TestWindow::new().unwrap();

    if !args.window {
        let test_window_weak = test_window.as_weak();
        let config_window_weak = config_window.as_weak();
        let config = config.clone();
        let mut app_state_capture = None;
        let mut wgpu = None;
        test_window
            .window()
            .set_rendering_notifier(move |state, graphics_api| {
                let test_window = test_window_weak.upgrade().unwrap();
                match state {
                    slint::RenderingState::RenderingSetup => {
                        match app_state_capture {
                            None => {
                                // Initialize the application state
                                let request_redraw_callback = request_redraw_callback.clone();
                                let mut app_state = ApplicationState::new(
                                    config.clone(),
                                    request_redraw_callback,
                                    config_window_weak.clone(),
                                );
                                app_state.initialize_audio_and_fft();
                                if let slint::GraphicsAPI::WGPU24 {
                                    device,
                                    queue,
                                    config: surface_config,
                                    ..
                                } = graphics_api
                                {
                                    wgpu = Some(Rc::new(SlintWgpuSurface {
                                        device: device.clone(),
                                        queue: Arc::new(queue.clone()),
                                        surface_configuration: Some(surface_config.clone()),
                                    })
                                        as Rc<dyn WgpuSurface>);
                                    app_state.configure_primary_wgpu_surface(
                                        &wgpu.as_ref().unwrap(),
                                        PanelAnchorPosition::Bottom,
                                    );
                                    // app_state.left_waveform_view =
                                    //     Some(app_state.create_waveform_view_no_window(
                                    //         ui::Style::Compressed,
                                    //         true,
                                    //         device,
                                    //         &Arc::new(queue.clone()),
                                    //         surface_config.format,
                                    //     ));
                                }
                                app_state_capture = Some(app_state);
                            }
                            _ => {}
                        };
                    }
                    slint::RenderingState::BeforeRendering => {
                        if let slint::GraphicsAPI::WGPU24 {
                            frame: Some(frame), ..
                        } = graphics_api
                        {
                            let app_state = app_state_capture.as_mut().unwrap();
                            let tick = test_window.get_tick();
                            // println!("{}", tick);
                            app_state.process_audio(tick as u32);
                            // FIXME: Not every frame
                            app_state
                                .set_screen_size(frame.texture.width(), frame.texture.height());

                            // FIXME: Find a way to apply the dirty screen size in ridgeline
                            app_state.lazy_config_changes.insert(RIDGELINE_WIDTH);
                            // let waveform_view = app_state.left_waveform_view.as_mut().unwrap();

                            // let view = frame
                            //     .texture
                            //     .create_view(&wgpu::TextureViewDescriptor::default());

                            // let mut encoder =
                            //     device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            //         label: None,
                            //     });
                            app_state.render(&wgpu.as_ref().unwrap(), frame);

                            // waveform_view.render(&mut encoder, &view, None);

                            // queue.submit(Some(encoder.finish()));
                        }
                    }
                    _ => {}
                }
            })
            .unwrap();
        test_window.show().unwrap();
    }

    // Tie the main thread to the config window, since winit needs to be there on some platforms.
    slint::run_event_loop_until_quit().unwrap();
}
