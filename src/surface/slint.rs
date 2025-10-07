// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use std::{
    rc::Rc,
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver},
    },
};

use mpris::{FindingError, Player, PlayerFinder, ProgressTick};
use slint::ComponentHandle;
use slint::SharedPixelBuffer;
use std::io::Read;

use crate::{
    AppMessageCallback, ApplicationState, GlobalCanvasContext, surface::GlobalCanvas, ui,
    view::WindowMode,
};

use super::WgpuSurface;

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
    ui_msg_rx: Receiver<AppMessageCallback>,
    app_state: Option<ApplicationState>,
    wgpu_surface: Option<Rc<SlintWgpuSurface>>,
    window: slint::Weak<ui::CanvasWindow>,
}

impl SlintGlobalCanvas {
    fn process_messages(&mut self) {
        // Process UI callbacks here since some require the wayland connection to recreate windows.
        while let Ok(message) = self.ui_msg_rx.try_recv() {
            match message {
                AppMessageCallback::ApplicationState(closure) => {
                    closure(self.app_state.as_mut().unwrap())
                }
                AppMessageCallback::SlintGlobalCanvas(closure) => {
                    closure(self, GlobalCanvasContext::Slint(()))
                }
                AppMessageCallback::WlrGlobalCanvas(_) => {
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
        let padding_ratio = match app_state.config.style {
            ui::Style::Compressed => {
                app_state.config.compressed.width_ratio as f32
                    * app_state.config.compressed.exclusive_ratio
            }
            ui::Style::Ridgeline => {
                app_state.config.ridgeline.width_ratio as f32
                    * app_state.config.ridgeline.exclusive_ratio
            }
        };

        let (horizontal, top, bottom) = match app_state.config.general.layout {
            ui::PanelLayout::SingleBottom => (0.0, 0.0, padding_ratio),
            ui::PanelLayout::SingleTop => (0.0, padding_ratio, 0.0),
            ui::PanelLayout::TwoPanels => (padding_ratio, 0.0, 0.0),
        };
        let window = self.window.upgrade().unwrap();
        window.set_horizontal_padding_ratio(horizontal);
        window.set_top_padding_ratio(top);
        window.set_bottom_padding_ratio(bottom);
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

type MprisMessage = Box<dyn FnOnce(&Player) + Send>;

pub fn initialize_slint_surface(
    config: ui::Configuration,
    ui_msg_rx: Receiver<AppMessageCallback>,
    canvas_window: ui::CanvasWindow,
    config_window_weak: slint::Weak<ui::ConfigurationWindow>,
    request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
) {
    let canvas_window_weak = canvas_window.as_weak();
    let mut config_holder = Some(config);

    let mut slint_global_canvas = SlintGlobalCanvas {
        ui_msg_rx,
        app_state: None,
        wgpu_surface: None,
        window: canvas_window_weak.clone(),
    };

    let (mpris_msg_tx, mpris_msg_rx) = mpsc::channel::<MprisMessage>();

    fn wait_for_player() -> Result<Player, FindingError> {
        let player_finder = PlayerFinder::new()?;
        loop {
            // FIXME: This is expensive, we should listen to DBus signals instead.
            let player = player_finder.find_active();
            if player.is_ok() {
                return player;
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }

    // Run mpris' progress_tracker.tick() in a separate thread
    std::thread::spawn(move || {
        let canvas_window_weak = canvas_window_weak;

        let mut maybe_player: Option<Player> = None;
        let mut maybe_progress_tracker: Option<mpris::ProgressTracker<'_>> = None;
        let mut last_status = None;
        let mut should_refresh_on_first_tick = true;

        loop {
            // Check if the current player is still active, else looks for a new one.
            if last_status != Some(mpris::PlaybackStatus::Playing)
                && last_status != Some(mpris::PlaybackStatus::Paused)
            {
                if let Ok(new_player) = wait_for_player() {
                    maybe_player = Some(new_player);
                    maybe_progress_tracker =
                        Some(maybe_player.as_ref().unwrap().track_progress(20).unwrap());
                }
            }

            // Send commands sent from the UI to the player
            while let Ok(closure) = mpris_msg_rx.try_recv() {
                // Ignore any command sent while there is no player, the UI should be disabled instead.
                if let Some(player) = maybe_player.as_ref() {
                    closure(player);
                }
            }

            if let Some(progress_tracker) = maybe_progress_tracker.as_mut() {
                let ProgressTick {
                    player_quit,
                    progress,
                    progress_changed,
                    ..
                } = progress_tracker.tick();
                if player_quit {
                    // This is the last tick for this player, reset everything already.
                    maybe_player = None;
                    maybe_progress_tracker = None;
                    last_status = None;
                    should_refresh_on_first_tick = true;

                    canvas_window_weak
                        .upgrade_in_event_loop(move |canvas_window| {
                            canvas_window.set_track_artist("".into());
                            canvas_window.set_track_title("".into());
                            canvas_window.set_track_album("".into());
                            canvas_window.set_playing(false);
                            canvas_window.set_art_image(slint::Image::default());
                        })
                        .unwrap();
                    continue;
                } else if progress_changed || should_refresh_on_first_tick {
                    should_refresh_on_first_tick = false;
                    last_status = Some(progress.playback_status());

                    let metadata = progress.metadata();
                    let art_url = metadata.art_url().map(|s| s.to_string());
                    let maybe_image =
                        art_url.as_ref().and_then(|url| {
                            fn load_bytes(
                                url: &str,
                            ) -> Result<Vec<u8>, Box<dyn std::error::Error>>
                            {
                                let parsed = url::Url::parse(url)?;
                                if parsed.scheme() == "file" {
                                    // Use url crate to parse and decode the path
                                    let path = parsed.to_file_path().map_err(|_| {
                                        format!("Failed to convert file URL to path: {url}")
                                    })?;
                                    std::fs::read(path).map_err(|e| e.into())
                                } else if parsed.scheme() == "http" || parsed.scheme() == "https" {
                                    let resp = ureq::request_url("GET", &parsed)
                                        .timeout(std::time::Duration::from_secs(5))
                                        .call()?;
                                    let mut reader = resp.into_reader();
                                    let mut buf = Vec::new();
                                    reader.read_to_end(&mut buf)?;
                                    Ok(buf)
                                } else {
                                    Err(format!("Unsupported art URL scheme: {url}").into())
                                }
                            }

                            fn decode_to_shared_pixel_buffer(
                                data: &[u8],
                            ) -> Result<
                                SharedPixelBuffer<slint::Rgba8Pixel>,
                                Box<dyn std::error::Error>,
                            > {
                                let dyn_img = image::load_from_memory(data)?;
                                let rgba8 = dyn_img.to_rgba8();
                                let (width, height) = rgba8.dimensions();
                                let mut buffer =
                                    SharedPixelBuffer::<slint::Rgba8Pixel>::new(width, height);
                                buffer.make_mut_bytes().copy_from_slice(&rgba8);
                                Ok(buffer)
                            }

                            match load_bytes(url)
                                .and_then(|bytes| decode_to_shared_pixel_buffer(&bytes))
                            {
                                Ok(buffer) => Some(buffer),
                                Err(e) => {
                                    println!(
                                        "Failed to load or decode art image for URL [{url}]: {e}"
                                    );
                                    None
                                }
                            }
                        });

                    let status = progress.playback_status();
                    let artist = progress
                        .metadata()
                        .artists()
                        .unwrap_or_default()
                        .join(", ")
                        .into();
                    let title = progress.metadata().title().unwrap_or_default().into();
                    let album = progress.metadata().album_name().unwrap_or_default().into();

                    canvas_window_weak
                        .upgrade_in_event_loop(move |canvas_window| {
                            canvas_window.set_track_artist(artist);
                            canvas_window.set_track_title(title);
                            canvas_window.set_track_album(album);
                            canvas_window.set_playing(status == mpris::PlaybackStatus::Playing);

                            if let Some(pixel_buffer) = maybe_image {
                                let image = slint::Image::from_rgba8(pixel_buffer);
                                canvas_window.set_art_image(image);
                            } else {
                                canvas_window.set_art_image(slint::Image::default());
                            }
                        })
                        .unwrap();
                }
            }
        }
    });

    canvas_window.on_previous({
        let mpris_msg_tx = mpris_msg_tx.clone();
        move || {
            mpris_msg_tx
                .send(Box::new(|player: &Player| {
                    player.previous().unwrap();
                }))
                .unwrap();
        }
    });
    canvas_window.on_play_pause({
        let mpris_msg_tx = mpris_msg_tx.clone();
        move || {
            mpris_msg_tx
                .send(Box::new(|player: &Player| {
                    player.play_pause().unwrap();
                }))
                .unwrap();
        }
    });
    canvas_window.on_next({
        let mpris_msg_tx = mpris_msg_tx.clone();
        move || {
            mpris_msg_tx
                .send(Box::new(|player: &Player| {
                    player.next().unwrap();
                }))
                .unwrap();
        }
    });

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
                            slint_global_canvas.wgpu_surface = Some(Rc::new(SlintWgpuSurface {
                                device: device.clone(),
                                queue: Arc::new(queue.clone()),
                                surface_configuration: Some(surface_config.clone()),
                            }));
                            slint_global_canvas.apply_panel_layout(&GlobalCanvasContext::Slint(()));
                        }

                        let app_state = slint_global_canvas.app_state.as_mut().unwrap();

                        let window = slint_global_canvas.window.upgrade().unwrap();
                        let tick = window.get_tick();

                        app_state.process_audio(tick as u32);

                        app_state
                            .update_screen_size(surface_texture.width(), surface_texture.height());

                        let wgpu = slint_global_canvas.wgpu_surface.as_ref().unwrap().clone()
                            as Rc<dyn WgpuSurface>;
                        // Grey clear color for Slint
                        let clear_color = wgpu::Color {
                            r: 0.66,
                            g: 0.66,
                            b: 0.66,
                            a: 1.0,
                        };
                        app_state.render_with_clear_color(&wgpu, surface_texture, clear_color);
                    }
                }
                _ => {}
            }
        })
        .unwrap();
    // FIXME: Figure out why it doesn't quit
    canvas_window.show().unwrap();
}
