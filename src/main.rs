use num_complex::Complex;
use ringbuf::storage::Heap;
use ringbuf::traits::{Consumer, Split};
use ringbuf::wrap::caching::Caching;
use ringbuf::{HeapRb, SharedRb};
use rustfft::{Fft, FftDirection, FftPlanner};
use signal_hook::iterator::Signals;
use slint::ComponentHandle;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use views::WaveformView;
use wayland_client::{Connection, QueueHandle};
use wgpu::TextureFormat;
use wlr_layers::{PanelAnchorPosition, WlrWaylandEventHandler};

mod audio;
mod ui;
mod views;
mod wlr_layers;

const VERTEX_BUFFER_SIZE: usize = 44100 * 3;
const FFT_SIZE: usize = 2048;
const NUM_CHANNELS: usize = 2;

pub trait WgpuSurface {
    fn adapter(&self) -> &wgpu::Adapter;
    fn device(&self) -> &wgpu::Device;
    fn surface(&self) -> &wgpu::Surface<'static>;
    fn queue(&self) -> &Arc<wgpu::Queue>;
    fn surface_id(&self) -> u32;
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
                waveform_view.render(&mut encoder, &view, &depth_texture_view);
            }
        }
        if let Some(waveform_view) = right_waveform_view {
            if waveform_view.render_window() == self.render_window {
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

struct ApplicationState {
    pub config: ui::Configuration,
    pub lazy_config_changes: HashSet<usize>,
    primary_waveform_window: Option<(WaveformWindow, PanelAnchorPosition)>,
    secondary_waveform_window: Option<(WaveformWindow, PanelAnchorPosition)>,
    last_non_zero_sample_age: usize,
    animation_stopped: Arc<AtomicBool>,
    left_waveform_view: Option<Box<dyn WaveformView>>,
    right_waveform_view: Option<Box<dyn WaveformView>>,
    screen_size: (u32, u32),
    audio_input_ringbuf_cons: Option<Caching<Arc<SharedRb<Heap<f32>>>, false, true>>,
    fft: Option<Arc<dyn Fft<f32>>>,
    fft_inout_buffer: Option<Vec<Complex<f32>>>,
    fft_scratch: Option<Vec<Complex<f32>>>,
    request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
}

impl ApplicationState {
    fn new(
        config: ui::Configuration,
        request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
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
        style: ui::Style,
        render_window: RenderWindow,
        is_left_channel: bool,
    ) -> Box<dyn WaveformView> {
        let (window, anchor_position) = match render_window {
            RenderWindow::Primary => self.primary_waveform_window.as_ref().unwrap(),
            RenderWindow::Secondary => self.secondary_waveform_window.as_ref().unwrap(),
        };
        let mut view: Box<dyn WaveformView> = match style {
            ui::Style::Ridgeline => Box::new(views::RidgelineWaveformView::new(
                &window,
                render_window,
                *anchor_position,
                self.config.general.channels,
                is_left_channel,
            )),
            ui::Style::Compressed => Box::new(views::CompressedWaveformView::new(
                &window,
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
            Some(self.create_waveform_view(self.config.style, RenderWindow::Primary, true));

        if self.config.general.channels == ui::RenderChannels::Both {
            self.right_waveform_view =
                Some(self.create_waveform_view(self.config.style, RenderWindow::Primary, false));
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
            Some(self.create_waveform_view(self.config.style, RenderWindow::Secondary, false));
    }

    fn primary_resized(&mut self, width: u32, height: u32) {
        if let Some((waveform_window, _)) = self.primary_waveform_window.as_mut() {
            waveform_window.reconfigure(width, height);
        }
    }
    fn secondary_resized(&mut self, width: u32, height: u32) {
        if let Some((waveform_window, _)) = self.secondary_waveform_window.as_mut() {
            waveform_window.reconfigure(width, height);
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
    fn render(&mut self, surface_id: u32) {
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
            if window.wgpu.surface_id() == surface_id {
                window.render(&mut self.left_waveform_view, &mut self.right_waveform_view);
            }
        }

        if let Some((window, _)) = self.secondary_waveform_window.as_mut() {
            if window.wgpu.surface_id() == surface_id {
                window.render(&mut self.left_waveform_view, &mut self.right_waveform_view);
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
        if self.primary_waveform_window.is_some() {
            self.left_waveform_view =
                Some(self.create_waveform_view(self.config.style, RenderWindow::Primary, true));

            if self.config.general.channels == ui::RenderChannels::Both {
                self.right_waveform_view = Some(self.create_waveform_view(
                    self.config.style,
                    RenderWindow::Primary,
                    false,
                ));
            } else {
                self.right_waveform_view = None;
            }
        }
        if self.secondary_waveform_window.is_some() {
            self.right_waveform_view =
                Some(self.create_waveform_view(self.config.style, RenderWindow::Secondary, false));
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
}

impl AppMessage {
    pub fn to_app<F>(f: F) -> Self
    where
        F: FnOnce(&mut ApplicationState) + Send + 'static,
    {
        AppMessage::ApplicationStateCallback(Box::new(f))
    }

    pub fn to_event_handler<F>(f: F) -> Self
    where
        F: FnOnce(&mut WlrWaylandEventHandler, &Connection, &QueueHandle<WlrWaylandEventHandler>)
            + Send
            + 'static,
    {
        AppMessage::WlrWaylandEventHandlerCallback(Box::new(f))
    }
}

pub fn main() {
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
    config_window.show().unwrap();

    // Spawn the wlr panel rendering in a separate thread, this is supported with wayland
    std::thread::spawn(move || {
        let app_state = ApplicationState::new(config, request_redraw_callback.clone());

        let mut layers_even_queue =
            wlr_layers::WlrWaylandEventLoop::new(app_state, app_msg_rx, request_redraw_callback);
        layers_even_queue.run_event_loop();
    });

    // The panels don't accept using input, so allow showing the config window again through SIGUSR1.
    std::thread::spawn({
        let mut signals = Signals::new(&[signal_hook::consts::SIGUSR1]).unwrap();
        let config_window_weak = config_window.as_weak();
        move || {
            for sig in signals.forever() {
                if sig == signal_hook::consts::SIGUSR1 {
                    let config_window_weak = config_window_weak.clone();
                    send_app_msg(AppMessage::ApplicationStateCallback(Box::new(move |app| {
                        let config = app.config.clone();

                        config_window_weak
                            .upgrade_in_event_loop(move |window| {
                                window.update_from_configuration(&config);
                                window.show().unwrap();
                            })
                            .unwrap();
                    })));
                }
            }
        }
    });

    // Tie the main thread to the config window, since winit needs to be there on some platforms.
    slint::run_event_loop_until_quit().unwrap();
}
