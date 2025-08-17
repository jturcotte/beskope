// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use std::{
    rc::Rc,
    sync::{Arc, Mutex, mpsc::Receiver},
};

use slint::ComponentHandle;

use crate::{
    AppMessage, ApplicationState, GlobalCanvasContext, surface::GlobalCanvas, ui, view::WindowMode,
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

pub fn initialize_slint_surface(
    config: ui::Configuration,
    ui_msg_rx: Receiver<AppMessage>,
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
                            slint_global_canvas.wgpu_surface = Some(Rc::new(SlintWgpuSurface {
                                device: device.clone(),
                                queue: Arc::new(queue.clone()),
                                surface_configuration: Some(surface_config.clone()),
                            }));
                            slint_global_canvas.apply_panel_layout(&GlobalCanvasContext::Slint(()));
                        }

                        let app_state = slint_global_canvas.app_state.as_mut().unwrap();

                        let tick = slint_global_canvas.window.upgrade().unwrap().get_tick();

                        app_state.process_audio(tick as u32);

                        app_state
                            .update_screen_size(surface_texture.width(), surface_texture.height());

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
