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
use slint::wgpu_26::{WGPUConfiguration, WGPUSettings};
use std::collections::HashSet;
use std::io::prelude::*;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use view::WaveformView;

use crate::view::{RenderSurface, ViewSurface, ViewTransform, WindowMode};
use crate::wlr_layers::WlrCanvasContext;

mod audio;
mod ui;
mod view;
mod wlr_layers;

const VERTEX_BUFFER_SIZE: usize = 44100 * 3;
const FFT_SIZE: usize = 2048;
const NUM_CHANNELS: usize = 2;

pub trait WgpuSurface {
    fn device(&self) -> &wgpu::Device;
    fn queue(&self) -> &Arc<wgpu::Queue>;
    fn surface_id(&self) -> u32;
    fn swapchain_format(&self) -> Option<wgpu::TextureFormat>;
}

struct ApplicationState {
    pub config: ui::Configuration,
    pub lazy_config_changes: HashSet<usize>,
    window_mode: WindowMode,
    primary_view_surface: Option<ViewSurface>,
    secondary_view_surface: Option<ViewSurface>,
    last_non_zero_sample_age: usize,
    animation_stopped: Arc<AtomicBool>,
    pub left_waveform_view: Option<Box<dyn WaveformView>>,
    right_waveform_view: Option<Box<dyn WaveformView>>,
    screen_size: (u32, u32),
    view_transform_dirty: bool,
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
        window_mode: WindowMode,
        request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
        config_window: slint::Weak<ui::ConfigurationWindow>,
    ) -> ApplicationState {
        ApplicationState {
            config,
            window_mode,
            lazy_config_changes: HashSet::new(),
            primary_view_surface: None,
            secondary_view_surface: None,
            last_non_zero_sample_age: 0,
            animation_stopped: Arc::new(AtomicBool::new(false)),
            left_waveform_view: None,
            right_waveform_view: None,
            screen_size: (1, 1),
            view_transform_dirty: true,
            audio_input_ringbuf_cons: None,
            fft: None,
            fft_inout_buffer: None,
            fft_scratch: None,
            request_redraw_callback,
            config_window,
        }
    }

    /// Only updates the app state, the configuration window is updated before being shown.
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
        render_surface: RenderSurface,
        is_left_channel: bool,
    ) -> Box<dyn WaveformView> {
        let device = wgpu.device();
        let queue = wgpu.queue();
        let swapchain_format = wgpu.swapchain_format().unwrap();
        let mut view: Box<dyn WaveformView> = match style {
            ui::Style::Ridgeline => Box::new(view::RidgelineWaveformView::new(
                device,
                queue,
                swapchain_format,
                render_surface,
                is_left_channel,
            )),
            ui::Style::Compressed => Box::new(view::CompressedWaveformView::new(
                device,
                queue,
                swapchain_format,
                render_surface,
                is_left_channel,
            )),
        };

        let view_transform = ViewTransform::new(
            self.screen_size.0 as f32,
            self.screen_size.1 as f32,
            self.window_mode,
            self.config.general.layout,
            self.config.general.channels,
            is_left_channel,
        );
        view.apply_lazy_config_changes(&self.config, None, Some(&view_transform));

        view
    }

    /// Initializes the primary waveform window and the views it contains.
    fn initialize_primary_view_surface(&mut self, wgpu: &Rc<dyn WgpuSurface>) {
        let config_window = self.config_window.clone();
        let fps_callback = Box::new(move |fps: u32| {
            config_window
                .upgrade_in_event_loop(move |window| {
                    window.set_primary_fps(fps as i32);
                })
                .unwrap();
        });

        let window = ViewSurface::new(wgpu, RenderSurface::Primary, fps_callback);
        self.primary_view_surface = Some(window);

        self.left_waveform_view =
            Some(self.create_waveform_view(wgpu, self.config.style, RenderSurface::Primary, true));

        // FIXME: This creates it even if the layout is split, but it will be replaced later.
        if self.config.general.channels == ui::RenderChannels::Both {
            self.right_waveform_view = Some(self.create_waveform_view(
                wgpu,
                self.config.style,
                RenderSurface::Primary,
                false,
            ));
        }
    }

    /// Initializes the secondary waveform window and the views it contains.
    fn initialize_secondary_view_surface(&mut self, wgpu_surface: &Rc<dyn WgpuSurface>) {
        let config_window = self.config_window.clone();
        let fps_callback = Box::new(move |fps: u32| {
            config_window
                .upgrade_in_event_loop(move |window| {
                    window.set_secondary_fps(fps as i32);
                })
                .unwrap();
        });

        let window = ViewSurface::new(wgpu_surface, RenderSurface::Secondary, fps_callback);
        self.secondary_view_surface = Some(window);

        self.right_waveform_view = Some(self.create_waveform_view(
            wgpu_surface,
            self.config.style,
            RenderSurface::Secondary,
            false,
        ));
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

        if let Some(left_view_surface) = &mut self.left_waveform_view {
            left_view_surface.process_audio(
                timestamp,
                &data,
                fft.as_ref(),
                fft_inout_buffer,
                fft_scratch,
            );
        }
        if let Some(right_view_surface) = &mut self.right_waveform_view {
            right_view_surface.process_audio(
                timestamp,
                &data,
                fft.as_ref(),
                fft_inout_buffer,
                fft_scratch,
            );
        }
    }
    fn render(&mut self, wgpu: &Rc<dyn WgpuSurface>, surface_texture: &wgpu::Texture) {
        if !self.lazy_config_changes.is_empty() || self.view_transform_dirty {
            let channels = self.config.general.channels;
            let layout = self.config.general.layout;
            if let Some(waveform_view) = self.left_waveform_view.as_mut() {
                let maybe_view_transform = if self.view_transform_dirty {
                    Some(ViewTransform::new(
                        self.screen_size.0 as f32,
                        self.screen_size.1 as f32,
                        self.window_mode,
                        layout,
                        channels,
                        true,
                    ))
                } else {
                    None
                };

                waveform_view.apply_lazy_config_changes(
                    &self.config,
                    Some(&self.lazy_config_changes),
                    maybe_view_transform.as_ref(),
                );
            }
            if let Some(waveform_view) = self.right_waveform_view.as_mut() {
                let maybe_view_transform = if self.view_transform_dirty {
                    Some(ViewTransform::new(
                        self.screen_size.0 as f32,
                        self.screen_size.1 as f32,
                        self.window_mode,
                        layout,
                        channels,
                        false,
                    ))
                } else {
                    None
                };

                waveform_view.apply_lazy_config_changes(
                    &self.config,
                    Some(&self.lazy_config_changes),
                    maybe_view_transform.as_ref(),
                );
            }

            // Restart tracking changed from the UI to apply on the next frame.
            self.lazy_config_changes.clear();
            self.view_transform_dirty = false;
        }

        if let Some(window) = self.primary_view_surface.as_mut() {
            if window.surface_id() == wgpu.surface_id() {
                window.render(
                    wgpu,
                    surface_texture,
                    &mut self.left_waveform_view,
                    &mut self.right_waveform_view,
                );
            }
        }

        if let Some(window) = self.secondary_view_surface.as_mut() {
            if window.surface_id() == wgpu.surface_id() {
                window.render(
                    wgpu,
                    surface_texture,
                    &mut self.left_waveform_view,
                    &mut self.right_waveform_view,
                );
            }
        }
    }

    fn update_screen_size(&mut self, width: u32, height: u32) {
        if width != self.screen_size.0 || height != self.screen_size.1 {
            self.screen_size.0 = width;
            self.screen_size.1 = height;
            self.view_transform_dirty = true;
        }
    }

    pub fn recreate_views(&mut self) {
        if let Some(window) = self.primary_view_surface.as_ref() {
            self.left_waveform_view = Some(self.create_waveform_view(
                &window.wgpu,
                self.config.style,
                RenderSurface::Primary,
                true,
            ));

            if self.config.general.channels == ui::RenderChannels::Both {
                self.right_waveform_view = Some(self.create_waveform_view(
                    &window.wgpu,
                    self.config.style,
                    RenderSurface::Primary,
                    false,
                ));
            } else {
                self.right_waveform_view = None;
            }
        }
        if let Some(window) = self.secondary_view_surface.as_ref() {
            self.right_waveform_view = Some(self.create_waveform_view(
                &window.wgpu,
                self.config.style,
                RenderSurface::Secondary,
                false,
            ));
        }
    }
}

enum AppMessage {
    ApplicationStateCallback(Box<dyn FnOnce(&mut ApplicationState) + Send>),
    WlrGlobalCanvasCallback(Box<dyn FnOnce(&mut dyn GlobalCanvas, GlobalCanvasContext) + Send>),
    SlintGlobalCanvasCallback(Box<dyn FnOnce(&mut dyn GlobalCanvas, GlobalCanvasContext) + Send>),
}

enum GlobalCanvasContext {
    Wlr(WlrCanvasContext),
    Slint(()),
}

trait GlobalCanvas {
    fn app_state(&mut self) -> &mut ApplicationState;
    fn apply_panel_width_change(&mut self);
    fn apply_panel_exclusive_ratio_change(&mut self);
    fn apply_panel_layout(&mut self, context: &GlobalCanvasContext);
    fn set_panel_layer(&mut self, layer: ui::PanelLayer);
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
    device: wgpu::Device,
    queue: Arc<wgpu::Queue>,
    surface_configuration: Option<wgpu::SurfaceConfiguration>,
}

impl WgpuSurface for SlintWgpuSurface {
    fn device(&self) -> &wgpu::Device {
        &self.device
    }

    fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    fn surface_id(&self) -> u32 {
        // Only one window will be used
        0
    }

    fn swapchain_format(&self) -> Option<wgpu::TextureFormat> {
        self.surface_configuration
            .as_ref()
            .map(|config| config.format)
    }
}

struct SlintGlobalCanvas {
    ui_msg_rx: Receiver<AppMessage>,
    app_state: Option<ApplicationState>,
    wgpu_surface: Option<Rc<SlintWgpuSurface>>,
    window: slint::Weak<ui::CanvasWindow>,
}

impl SlintGlobalCanvas {
    fn process_messages(&mut self) {
        // Process UI callbacks here since some require the wayland connection to recreate windows.
        while let Ok(message) = self.ui_msg_rx.try_recv() {
            match message {
                AppMessage::ApplicationStateCallback(closure) => {
                    closure(self.app_state.as_mut().unwrap())
                }
                AppMessage::SlintGlobalCanvasCallback(closure) => {
                    closure(self, GlobalCanvasContext::Slint(()))
                }
                AppMessage::WlrGlobalCanvasCallback(_) => {
                    panic!("Incorrect GlobalCanvas callback type")
                }
            }
        }
    }
}

impl GlobalCanvas for SlintGlobalCanvas {
    fn app_state(&mut self) -> &mut ApplicationState {
        self.app_state.as_mut().unwrap()
    }

    fn apply_panel_width_change(&mut self) {
        // Applied through the view transform

        // Also update the exclusive ratio since it's based on the width
        self.apply_panel_exclusive_ratio_change();
    }

    fn apply_panel_exclusive_ratio_change(&mut self) {
        let app_state = self.app_state.as_ref().unwrap();
        let panel_width = match app_state.config.style {
            ui::Style::Compressed => {
                app_state.config.compressed.width as f32
                    * app_state.config.compressed.exclusive_ratio
            }
            ui::Style::Ridgeline => {
                app_state.config.ridgeline.width as f32 * app_state.config.ridgeline.exclusive_ratio
            }
        };

        self.window.upgrade().unwrap().set_view_width(panel_width);
    }

    fn apply_panel_layout(&mut self, _context: &GlobalCanvasContext) {
        let wgpu = self.wgpu_surface.as_ref().unwrap().clone() as Rc<dyn WgpuSurface>;
        self.app_state
            .as_mut()
            .unwrap()
            .initialize_primary_view_surface(&wgpu);
        self.apply_panel_exclusive_ratio_change();
    }

    fn set_panel_layer(&mut self, _layer: ui::PanelLayer) {}
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

    let send_msg = {
        let app_msg_tx = app_msg_tx.clone();
        let request_redraw_callback = request_redraw_callback.clone();
        move |msg| {
            app_msg_tx.send(msg).unwrap();
            request_redraw_callback.lock().unwrap()();
        }
    };
    let send_app_msg = {
        let app_msg_tx = app_msg_tx.clone();
        let request_redraw_callback = request_redraw_callback.clone();
        move |f| {
            let msg = AppMessage::ApplicationStateCallback(f);
            app_msg_tx.send(msg).unwrap();
            request_redraw_callback.lock().unwrap()();
        }
    };
    let send_canvas_msg = {
        let wlr = !args.window;
        let app_msg_tx = app_msg_tx.clone();
        let request_redraw_callback = request_redraw_callback.clone();
        move |f| {
            let msg = if wlr {
                AppMessage::WlrGlobalCanvasCallback(f)
            } else {
                AppMessage::SlintGlobalCanvasCallback(f)
            };
            app_msg_tx.send(msg).unwrap();
            // FIXME: This queues additional frame requests, this should check if it's stopped or not.
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

    // Initialize the Slint backend with WGPU
    // for both the canvas and the configuration windows.
    let mut wgpu_settings = WGPUSettings::default();
    // Slint defaults to WebGL2 limits, but use the real wgpu default.
    wgpu_settings.device_required_limits = wgpu::Limits::default();

    slint::BackendSelector::new()
        .require_wgpu_26(WGPUConfiguration::Automatic(wgpu_settings))
        .select()
        .expect("Unable to create Slint backend with WGPU based renderer");

    let config_window = ui::init(send_app_msg, send_canvas_msg);
    config_window.update_from_configuration(&config);
    if show_config {
        config_window.show().unwrap();
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
                                send_msg(AppMessage::ApplicationStateCallback(Box::new(
                                    move |app| {
                                        let config = app.config.clone();

                                        config_window_weak
                                            .upgrade_in_event_loop(move |window| {
                                                window.update_from_configuration(&config);
                                                window.show().unwrap();
                                            })
                                            .unwrap();
                                    },
                                )));
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

    // Keep the window ownership on the stack, it owns everything else when rendering inside a Slint window.
    let canvas_window = ui::CanvasWindow::new().unwrap();

    if !args.window {
        let config_window_weak = config_window.as_weak();
        let request_redraw_callback = request_redraw_callback.clone();
        let config = config.clone();
        // Spawn the wlr panel rendering in a separate thread, this is supported with wayland
        std::thread::spawn(move || {
            let app_state = ApplicationState::new(
                config,
                WindowMode::WindowPerPanel,
                request_redraw_callback.clone(),
                config_window_weak,
            );

            let mut layers_even_queue = wlr_layers::WlrWaylandEventLoop::new(
                app_state,
                app_msg_rx,
                request_redraw_callback,
            );
            layers_even_queue.run_event_loop();
        });
    } else {
        let canvas_window_weak = canvas_window.as_weak();
        let config_window_weak = config_window.as_weak();
        let mut config_holder = Some(config);

        let mut slint_global_canvas = SlintGlobalCanvas {
            ui_msg_rx: app_msg_rx,
            app_state: None,
            wgpu_surface: None,
            window: canvas_window_weak,
        };

        canvas_window
            .window()
            .set_rendering_notifier(move |state, graphics_api| {
                match state {
                    slint::RenderingState::RenderingSetup => {
                        if slint_global_canvas.app_state.is_none() {
                            // Initialize the application state
                            let request_redraw_callback = request_redraw_callback.clone();
                            let config = config_holder
                                .take()
                                .expect("Can't initialize the app state twice");
                            let mut app_state = ApplicationState::new(
                                config,
                                WindowMode::WindowPerScene,
                                request_redraw_callback,
                                config_window_weak.clone(),
                            );
                            app_state.initialize_audio_and_fft();
                            slint_global_canvas.app_state = Some(app_state);
                        }
                    }
                    slint::RenderingState::BeforeRendering => {
                        if let slint::GraphicsAPI::WGPU26 {
                            device,
                            queue,
                            surface_texture: Some(surface_texture),
                            surface_configuration: Some(surface_config),
                            ..
                        } = graphics_api
                        {
                            slint_global_canvas.process_messages();

                            if slint_global_canvas.wgpu_surface.is_none() {
                                slint_global_canvas.wgpu_surface =
                                    Some(Rc::new(SlintWgpuSurface {
                                        device: device.clone(),
                                        queue: Arc::new(queue.clone()),
                                        surface_configuration: Some(surface_config.clone()),
                                    }));
                                slint_global_canvas
                                    .apply_panel_layout(&GlobalCanvasContext::Slint(()));
                            }

                            let app_state = slint_global_canvas.app_state.as_mut().unwrap();

                            let tick = slint_global_canvas.window.upgrade().unwrap().get_tick();

                            app_state.process_audio(tick as u32);

                            app_state.update_screen_size(
                                surface_texture.width(),
                                surface_texture.height(),
                            );

                            let wgpu = slint_global_canvas.wgpu_surface.as_ref().unwrap().clone()
                                as Rc<dyn WgpuSurface>;
                            app_state.render(&wgpu, surface_texture);
                        }
                    }
                    _ => {}
                }
            })
            .unwrap();
        canvas_window.show().unwrap();
    }

    // Tie the main thread to the config window, since winit needs to be there on some platforms.
    slint::run_event_loop_until_quit().unwrap();
}
