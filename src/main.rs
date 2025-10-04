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
use std::sync::mpsc;
use std::thread;
use view::View;

use crate::surface::{GlobalCanvas, GlobalCanvasContext, WgpuSurface};
use crate::view::{
    FFT_SIZE, RenderSurface, VERTEX_BUFFER_SIZE, ViewSurface, ViewTransform, WindowMode,
};

mod audio;
mod surface;
mod ui;
mod view;

const NUM_CHANNELS: usize = 2;

struct ApplicationState {
    pub config: ui::Configuration,
    pub lazy_config_changes: HashSet<usize>,
    window_mode: WindowMode,
    primary_view_surface: Option<ViewSurface>,
    secondary_view_surface: Option<ViewSurface>,
    last_non_zero_sample_age: usize,
    animation_stopped: Arc<AtomicBool>,
    pub left_view: Option<Box<dyn View>>,
    right_view: Option<Box<dyn View>>,
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
            left_view: None,
            right_view: None,
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

    fn create_view(
        &self,
        wgpu: &Rc<dyn WgpuSurface>,
        style: ui::Style,
        render_surface: RenderSurface,
        is_left_channel: bool,
    ) -> Box<dyn View> {
        let device = wgpu.device();
        let queue = wgpu.queue();
        let swapchain_format = wgpu.swapchain_format().unwrap();
        let mut view: Box<dyn View> = match style {
            ui::Style::Ridgeline => Box::new(view::RidgelineView::new(
                device,
                queue,
                swapchain_format,
                render_surface,
                is_left_channel,
            )),
            ui::Style::Compressed => Box::new(view::CompressedView::new(
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

    /// Initializes the primary view surface and the views it contains.
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

        self.left_view =
            Some(self.create_view(wgpu, self.config.style, RenderSurface::Primary, true));

        // FIXME: This creates it even if the layout is split, but it will be replaced later.
        if self.config.general.channels == ui::RenderChannels::Both {
            self.right_view =
                Some(self.create_view(wgpu, self.config.style, RenderSurface::Primary, false));
        }
    }

    /// Initializes the secondary view surface and the views it contains.
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

        self.right_view = Some(self.create_view(
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

        if let Some(left_view_surface) = &mut self.left_view {
            left_view_surface.process_audio(
                timestamp,
                &data,
                fft.as_ref(),
                fft_inout_buffer,
                fft_scratch,
            );
        }
        if let Some(right_view_surface) = &mut self.right_view {
            right_view_surface.process_audio(
                timestamp,
                &data,
                fft.as_ref(),
                fft_inout_buffer,
                fft_scratch,
            );
        }
    }
    fn render_with_clear_color(
        &mut self,
        wgpu: &Rc<dyn WgpuSurface>,
        surface_texture: &wgpu::Texture,
        clear_color: wgpu::Color,
    ) {
        if !self.lazy_config_changes.is_empty() || self.view_transform_dirty {
            let channels = self.config.general.channels;
            let layout = self.config.general.layout;
            if let Some(view) = self.left_view.as_mut() {
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

                view.apply_lazy_config_changes(
                    &self.config,
                    Some(&self.lazy_config_changes),
                    maybe_view_transform.as_ref(),
                );
            }
            if let Some(view) = self.right_view.as_mut() {
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

                view.apply_lazy_config_changes(
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
                window.render_with_clear_color(
                    wgpu,
                    surface_texture,
                    &mut self.left_view,
                    &mut self.right_view,
                    clear_color,
                );
            }
        }

        if let Some(window) = self.secondary_view_surface.as_mut() {
            if window.surface_id() == wgpu.surface_id() {
                window.render_with_clear_color(
                    wgpu,
                    surface_texture,
                    &mut self.left_view,
                    &mut self.right_view,
                    clear_color,
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
            self.left_view = Some(self.create_view(
                &window.wgpu,
                self.config.style,
                RenderSurface::Primary,
                true,
            ));

            if self.config.general.channels == ui::RenderChannels::Both {
                self.right_view = Some(self.create_view(
                    &window.wgpu,
                    self.config.style,
                    RenderSurface::Primary,
                    false,
                ));
            } else {
                self.right_view = None;
            }
        }
        if let Some(window) = self.secondary_view_surface.as_ref() {
            self.right_view = Some(self.create_view(
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

            let mut layers_even_queue = surface::wayland::WlrWaylandEventLoop::new(
                app_state,
                app_msg_rx,
                request_redraw_callback,
            );
            layers_even_queue.run_event_loop();
        });
    } else {
        surface::slint::initialize_slint_surface(
            config,
            app_msg_rx,
            canvas_window,
            config_window.as_weak(),
            request_redraw_callback,
        );
    }

    if args.window {
        slint::run_event_loop().unwrap();
    } else {
        // Tie the main thread to the config window, since winit needs to be there on some platforms.
        slint::run_event_loop_until_quit().unwrap();
    }
}
