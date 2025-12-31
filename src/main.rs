// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use clap::Parser;
use interprocess::local_socket::{GenericNamespaced, ListenerOptions, prelude::*};
use num_complex::Complex;
use ringbuf::storage::Heap;
use ringbuf::traits::{Consumer, Split};
use ringbuf::wrap::caching::Caching;
use ringbuf::{HeapRb, SharedRb};
use slint::ComponentHandle;
use slint::wgpu_27::{WGPUConfiguration, WGPUSettings};
use std::collections::HashSet;
use std::io::prelude::*;
use std::rc::Rc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar};
use std::thread;
use tracing::instrument;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::filter;
use tracing_subscriber::prelude::*;
use view::{RidgelineView, View, WaveformModel};

use crate::surface::{GlobalCanvas, GlobalCanvasContext, WgpuSurface};
use crate::view::{ConstantQTransformModel, RenderSurface, ViewSurface, ViewTransform, WindowMode};

mod audio;
mod surface;
mod ui;
mod view;

const NUM_CHANNELS: usize = 2;

#[derive(Copy, Clone, PartialEq, Eq)]
enum ChannelTransformMode {
    Raw,
    Cqt,
}

struct ApplicationState {
    pub config: ui::Configuration,
    pub lazy_config_changes: HashSet<usize>,
    window_mode: WindowMode,
    primary_view_surface: Option<ViewSurface>,
    secondary_view_surface: Option<ViewSurface>,
    animation_stopped: Arc<AtomicBool>,
    left_view: Option<Box<dyn View>>,
    right_view: Option<Box<dyn View>>,
    screen_size: (u32, u32),
    view_transform_dirty: bool,
    audio_input_ringbuf_cons: Option<Caching<Arc<SharedRb<Heap<f32>>>, false, true>>,
    audio_transform_control: Arc<(Mutex<(ChannelTransformMode, ChannelTransformMode)>, Condvar)>,
    cqt_left: Arc<Mutex<Vec<Complex<f64>>>>,
    cqt_right: Arc<Mutex<Vec<Complex<f64>>>>,
    config_window: slint::Weak<ui::ConfigurationWindow>,
}

impl ApplicationState {
    #[instrument(skip(config_window))]
    fn new(
        config: ui::Configuration,
        window_mode: WindowMode,
        config_window: slint::Weak<ui::ConfigurationWindow>,
        animation_stopped: Arc<AtomicBool>,
    ) -> ApplicationState {
        ApplicationState {
            config,
            window_mode,
            lazy_config_changes: HashSet::new(),
            primary_view_surface: None,
            secondary_view_surface: None,
            animation_stopped,
            left_view: None,
            right_view: None,
            screen_size: (1, 1),
            view_transform_dirty: true,
            audio_input_ringbuf_cons: None,
            audio_transform_control: Arc::new((
                Mutex::new((ChannelTransformMode::Raw, ChannelTransformMode::Raw)),
                Condvar::new(),
            )),
            cqt_left: Arc::new(Mutex::new(Vec::new())),
            cqt_right: Arc::new(Mutex::new(Vec::new())),
            config_window,
        }
    }

    /// Only updates the app state, the configuration window is updated before being shown.
    #[instrument(skip(self))]
    pub fn reload_configuration(&mut self) {
        let config = match ui::Configuration::load() {
            Ok(Some(config)) => config,
            Ok(None) => ui::Configuration::default(),
            Err(e) => {
                eprintln!("Failed to load configuration, will use default: {e}");
                ui::Configuration::default()
            }
        };
        self.config = config;
    }

    #[instrument(skip(self, request_redraw_callback))]
    fn initialize_audio_and_transform_thread(
        &mut self,
        request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
    ) {
        let (audio_input_ringbuf_prod, audio_input_ringbuf_cons) =
            HeapRb::<f32>::new(48_000 * NUM_CHANNELS).split();
        let (transform_thread_ringbuf_prod, transform_thread_ringbuf_cons) =
            HeapRb::<f32>::new(48_000 * NUM_CHANNELS).split();

        let (sample_rate_tx, sample_rate_rx) = std::sync::mpsc::sync_channel(1);

        std::thread::Builder::new()
            .name("pw-capture".into())
            .spawn({
                let animation_stopped = self.animation_stopped.clone();
                let audio_transform_control = self.audio_transform_control.clone();
                let sample_rate_tx = sample_rate_tx.clone();
                move || {
                    audio::initialize_audio_capture(
                        audio_input_ringbuf_prod,
                        animation_stopped,
                        request_redraw_callback,
                        audio_transform_control,
                        sample_rate_tx,
                    );
                }
            })
            .unwrap();

        self.audio_input_ringbuf_cons = Some(transform_thread_ringbuf_cons);

        // Block until the audio thread reports the sample rate.
        // FIXME: Keep updating the transform thread if the sampling rate changes.
        //        For now I haven't seen PipeWire do that even when changing audio devices.
        let sample_rate = sample_rate_rx.recv().unwrap() as f64;
        let bandwidth = (30.0, sample_rate / 2.0);
        let resolution = 36.0; // third-semitone resolution
        let guessed_cqt_size =
            f64::ceil(resolution * f64::log2(bandwidth.1 / bandwidth.0)) as usize;
        self.cqt_left
            .lock()
            .unwrap()
            .resize(guessed_cqt_size, Complex::<f64>::new(0.0, 0.0));
        self.cqt_right
            .lock()
            .unwrap()
            .resize(guessed_cqt_size, Complex::<f64>::new(0.0, 0.0));

        std::thread::Builder::new()
            .name("transform".into())
            .spawn({
                let cqt_buffer_left = self.cqt_left.clone();
                let cqt_buffer_right = self.cqt_right.clone();
                let audio_transform_control = self.audio_transform_control.clone();
                move || {
                    audio::init_audio_transform(
                        sample_rate as u32,
                        bandwidth,
                        resolution,
                        guessed_cqt_size,
                        audio_input_ringbuf_cons,
                        transform_thread_ringbuf_prod,
                        cqt_buffer_left,
                        cqt_buffer_right,
                        audio_transform_control,
                    );
                }
            })
            .unwrap();
    }

    #[instrument(skip(self, wgpu))]
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

        // Enable or disable the expensive CQT from the transform thread based on the selected style.
        {
            let mut transform_control_guard = self.audio_transform_control.0.lock().unwrap();
            let mode = if is_left_channel {
                &mut transform_control_guard.0
            } else {
                &mut transform_control_guard.1
            };
            match style {
                ui::Style::RidgelineFrequency => {
                    *mode = ChannelTransformMode::Cqt;
                }
                _ => {
                    *mode = ChannelTransformMode::Raw;
                }
            }
        }

        let mut view: Box<dyn View> = match style {
            ui::Style::RidgelineFrequency => {
                let model = ConstantQTransformModel::new(self.cqt_left.lock().unwrap().len());
                Box::new(RidgelineView::new(
                    device,
                    queue,
                    swapchain_format,
                    render_surface,
                    is_left_channel,
                    ui::Style::RidgelineFrequency,
                    model,
                ))
            }
            ui::Style::Ridgeline => {
                let model = WaveformModel::new(48_000 / 30);
                Box::new(RidgelineView::new(
                    device,
                    queue,
                    swapchain_format,
                    render_surface,
                    is_left_channel,
                    ui::Style::Ridgeline,
                    model,
                ))
            }
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
    #[instrument(skip(self, wgpu))]
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
    #[instrument(skip(self, wgpu_surface))]
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

    #[instrument(skip(self))]
    fn process_audio(&mut self, timestamp: u32) {
        // Get samples from the audio input ring buffer.
        // Will only be fed if the transform thread mode is not Raw for both channels.
        let data: Vec<f32> = self
            .audio_input_ringbuf_cons
            .as_mut()
            .unwrap()
            .pop_iter()
            .collect();

        let audio_data = view::AudioInputData {
            samples: &data,
            cqt_left: self.cqt_left.clone(),
            cqt_right: self.cqt_right.clone(),
        };
        if let Some(left_view_surface) = &mut self.left_view {
            left_view_surface.process_audio(timestamp, &audio_data);
        }
        if let Some(right_view_surface) = &mut self.right_view {
            right_view_surface.process_audio(timestamp, &audio_data);
        }
    }

    fn hide_fps_counters(&self) {
        self.config_window
            .upgrade_in_event_loop(move |window| {
                window.set_primary_fps(0);
                window.set_secondary_fps(0);
            })
            .unwrap();
    }

    #[instrument(skip(self, wgpu))]
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

        if let Some(window) = self.primary_view_surface.as_mut()
            && window.surface_id() == wgpu.surface_id()
        {
            window.render_with_clear_color(
                wgpu,
                surface_texture,
                &mut self.left_view,
                &mut self.right_view,
                clear_color,
            );
        }

        if let Some(window) = self.secondary_view_surface.as_mut()
            && window.surface_id() == wgpu.surface_id()
        {
            window.render_with_clear_color(
                wgpu,
                surface_texture,
                &mut self.left_view,
                &mut self.right_view,
                clear_color,
            );
        }
    }

    #[instrument(skip(self))]
    fn update_screen_size(&mut self, width: u32, height: u32) {
        if width != self.screen_size.0 || height != self.screen_size.1 {
            self.screen_size.0 = width;
            self.screen_size.1 = height;
            self.view_transform_dirty = true;
        }
    }

    #[instrument(skip(self))]
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

#[allow(clippy::type_complexity)]
enum AppMessageCallback {
    ApplicationState(Box<dyn FnOnce(&mut ApplicationState) + Send>),
    LayerShellGlobalCanvas(Box<dyn FnOnce(&mut dyn GlobalCanvas, GlobalCanvasContext) + Send>),
    SlintGlobalCanvas(Box<dyn FnOnce(&mut dyn GlobalCanvas, GlobalCanvasContext) + Send>),
}

impl std::fmt::Debug for AppMessageCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppMessageCallback::ApplicationState(_) => {
                write!(f, "ApplicationState(<closure>)")
            }
            AppMessageCallback::LayerShellGlobalCanvas(_) => {
                write!(f, "LayerShellGlobalCanvas(<closure>)")
            }
            AppMessageCallback::SlintGlobalCanvas(_) => {
                write!(f, "SlintGlobalCanvas(<closure>)")
            }
        }
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

    /// Record a frame timing trace that can be viewed at https://ui.perfetto.dev
    #[arg(long = "trace")]
    trace: bool,
}

const CONFIG_COMMAND: &[u8] = b"config";
const QUIT_COMMAND: &[u8] = b"quit";

pub fn main() {
    let args = Args::parse();
    let socket_path = "beskope.sock".to_ns_name::<GenericNamespaced>().unwrap();

    let _tracing_guard = if args.trace {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().include_args(true).build();
        tracing_subscriber::registry()
            .with(chrome_layer.with_filter(filter::filter_fn(|metadata| {
                // Returns `true` if and only if the span or event's target is
                // "http_access".
                metadata
                    .module_path()
                    .iter()
                    .all(|path| !path.starts_with("naga"))
            })))
            .init();
        Some(guard)
    } else {
        None
    };

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

    let config = match ui::Configuration::load() {
        Ok(Some(config)) => config,
        Ok(None) => {
            println!("No configuration file found, showing the configuration dialog.");
            show_config = true;
            ui::Configuration::default()
        }
        Err(e) => {
            eprintln!("Failed to load configuration: {e}");
            ui::Configuration::default()
        }
    };

    let (app_msg_tx, app_msg_rx) = mpsc::channel::<AppMessageCallback>();
    let request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>> =
        Arc::new(Mutex::new(Arc::new(|| {})));
    let animation_stopped = Arc::new(AtomicBool::new(false));

    let send_msg = {
        let app_msg_tx = app_msg_tx.clone();
        let request_redraw_callback = request_redraw_callback.clone();
        let animation_stopped = animation_stopped.clone();
        move |msg| {
            let _ = app_msg_tx.send(msg);
            // Only request a redraw if the animation is currently stopped to avoid double frame requests that would be accumulated by the compositor.
            if animation_stopped.load(Ordering::Relaxed) {
                request_redraw_callback.lock().unwrap()();
            }
        }
    };
    let send_app_msg = {
        let app_msg_tx = app_msg_tx.clone();
        let request_redraw_callback = request_redraw_callback.clone();
        let animation_stopped = animation_stopped.clone();
        move |f| {
            let msg = AppMessageCallback::ApplicationState(f);
            let _ = app_msg_tx.send(msg);
            if animation_stopped.load(Ordering::Relaxed) {
                request_redraw_callback.lock().unwrap()();
            }
        }
    };

    // Check if Wayland is available before creating the UI
    let can_use_wayland_layer_shell = surface::wayland::can_use_wayland();
    if !args.window && !can_use_wayland_layer_shell {
        println!(
            "Wayland compositor with wlr-layer-shell protocol support not available, falling back to windowed mode."
        );
    }
    let use_wayland_layer_shell = !args.window && can_use_wayland_layer_shell;

    // Create the appropriate canvas message sender based on the mode we'll use
    let send_canvas_msg = {
        let app_msg_tx = app_msg_tx.clone();
        let request_redraw_callback = request_redraw_callback.clone();
        let animation_stopped = animation_stopped.clone();
        move |f| {
            let msg = if use_wayland_layer_shell {
                AppMessageCallback::LayerShellGlobalCanvas(f)
            } else {
                AppMessageCallback::SlintGlobalCanvas(f)
            };
            let _ = app_msg_tx.send(msg);
            if animation_stopped.load(Ordering::Relaxed) {
                request_redraw_callback.lock().unwrap()();
            }
        }
    };

    // Initialize the Slint backend with WGPU
    // for both the canvas and the configuration windows.
    let mut wgpu_settings = WGPUSettings::default();
    // Slint defaults to WebGL2 limits, but use the real wgpu default.
    wgpu_settings.device_required_limits = wgpu::Limits::default();

    slint::BackendSelector::new()
        .require_wgpu_27(WGPUConfiguration::Automatic(wgpu_settings))
        .renderer_name("skia".to_owned())
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
                                send_msg(AppMessageCallback::ApplicationState(Box::new(
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

    if use_wayland_layer_shell {
        let config_window_weak = config_window.as_weak();
        let request_redraw_callback = request_redraw_callback.clone();
        let config = config.clone();
        // Spawn the layer shell panel rendering in a separate thread, this is supported with wayland
        std::thread::Builder::new()
            .name("layer-render".into())
            .spawn(move || {
                let app_state = ApplicationState::new(
                    config,
                    WindowMode::WindowPerPanel,
                    config_window_weak,
                    animation_stopped,
                );

                let mut layers_even_queue = surface::wayland::WaylandEventLoop::new(
                    app_state,
                    app_msg_rx,
                    request_redraw_callback,
                );
                layers_even_queue.run_event_loop();
            })
            .unwrap();
    } else {
        surface::slint::initialize_slint_surface(
            config,
            app_msg_rx,
            canvas_window,
            config_window.as_weak(),
            request_redraw_callback,
            animation_stopped,
        );
    }

    if !use_wayland_layer_shell {
        slint::run_event_loop().unwrap();
    } else {
        // Tie the main thread to the config window, since winit needs to be there on some platforms.
        slint::run_event_loop_until_quit().unwrap();
    }
}
