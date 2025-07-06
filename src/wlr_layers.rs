// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
};
use smithay_client_toolkit::compositor::Region;
use std::cell::RefCell;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use wayland_client::EventQueue;
use wayland_client::backend::ObjectId;

use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_layer, delegate_output, delegate_registry,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    shell::wlr_layer::{
        Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
        LayerSurfaceConfigure,
    },
};
use wayland_client::{
    Connection, Proxy, QueueHandle,
    globals::registry_queue_init,
    protocol::{wl_output, wl_surface},
};

use crate::ui::{self, PanelLayer};
use crate::{AppMessage, ApplicationState, GlobalCanvas, GlobalCanvasContext, WgpuSurface};

impl CompositorHandler for WlrWaylandEventHandler {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_factor: i32,
    ) {
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        conn: &Connection,
        qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        time: u32,
    ) {
        // Process UI callbacks here since some require the wayland connection to recreate windows.
        while let Ok(message) = self.ui_msg_rx.try_recv() {
            match message {
                AppMessage::ApplicationStateCallback(closure) => closure(&mut self.app_state),
                AppMessage::WlrWaylandEventHandlerCallback(closure) => closure(self, conn, qh),
                AppMessage::WlrGlobalCanvasCallback(closure) => closure(
                    self,
                    GlobalCanvasContext::Wlr(WlrCanvasContext {
                        conn: conn.clone(),
                        qh: qh.clone(),
                    }),
                ),
                AppMessage::SlintGlobalCanvasCallback(_) => {
                    panic!("Incorrect GlobalCanvas callback type")
                }
            }
        }

        let wgpu = match surface.id() {
            id if self
                .primary_layer
                .as_ref()
                .map_or(false, |l| l.wl_surface().id() == id) =>
            {
                self.primary_wgpu.as_ref().unwrap()
            }
            id if self
                .secondary_layer
                .as_ref()
                .map_or(false, |l| l.wl_surface().id() == id) =>
            {
                self.secondary_wgpu.as_ref().unwrap()
            }
            _ => panic!(
                "Received frame callback for a surface that is not a primary or secondary layer: {}",
                surface.id()
            ),
        };
        if !self.surfaces_with_pending_render.contains(&wgpu) {
            self.surfaces_with_pending_render.push(wgpu.clone());

            if !self.app_state.animation_stopped.load(Ordering::Relaxed) {
                // Already tell the compositor that we want to draw again for the next output frame.
                surface.frame(&qh, surface.clone());
                surface.commit();
            }
        }

        if self.primary_layer.as_ref().map(|l| l.wl_surface().id()) == Some(surface.id()) {
            self.pending_render_timestamp = time;
        }
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        output: &wl_output::WlOutput,
    ) {
        // Assume that both layers are on the same output, so only process changes on the primary layer.
        if Some(surface.id()) == self.primary_layer.as_ref().map(|l| l.wl_surface().id()) {
            self.primary_surface_output = Some(output.id());
            if let Some(size) = self
                .output_state
                .info(output)
                .and_then(|info| info.logical_size)
            {
                self.app_state.set_screen_size(size.0 as u32, size.1 as u32);
                // Adjust the exclusive ratio in case the configuration loaded at startup was too large to fit the screen.
                self.apply_panel_exclusive_ratio_change();
            }
        }
    }

    fn surface_leave(
        &mut self,
        conn: &Connection,
        qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        output: &wl_output::WlOutput,
    ) {
        if Some(surface.id()) == self.primary_layer.as_ref().map(|l| l.wl_surface().id()) {
            if self.primary_surface_output == Some(output.id()) {
                self.primary_surface_output = None;

                // The surface might not be remapped for wlr_layer_shell,
                // reconfigure them to map them to the new default output.
                self.apply_panel_layout(conn, qh);
            } else {
                debug_assert!(false, "Surface leave called with a different output");
            }
        }
    }
}

impl OutputHandler for WlrWaylandEventHandler {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        if self.primary_surface_output == Some(output.id()) {
            if let Some(size) = self
                .output_state
                .info(&output)
                .and_then(|info| info.logical_size)
            {
                self.app_state.set_screen_size(size.0 as u32, size.1 as u32);
            }
        }
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }
}

impl LayerShellHandler for WlrWaylandEventHandler {
    fn closed(&mut self, conn: &Connection, qh: &QueueHandle<Self>, _layer: &LayerSurface) {
        // The client should destroy the resource after receiving this event,
        // and create a new surface if they so choose.
        self.apply_panel_layout(conn, qh);
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let (new_width, new_height) = configure.new_size;

        if !self.app_state_initialized {
            self.app_state_initialized = true;
            self.app_state.initialize_audio_and_fft();
        }

        if self.primary_layer.as_ref() == Some(layer) {
            // FIXME: Change the holder thing holding the anchor, this is weird. Maybe this should let the app state decide instead
            if let Some(anchor_position) = self.primary_wgpu_holder.take() {
                let wgpu = self.primary_wgpu.as_ref().unwrap().clone() as Rc<dyn WgpuSurface>;
                self.app_state
                    .configure_primary_wgpu_surface(&wgpu, anchor_position);
                // Render once to let wgpu finalize the surface initialization.
                // FIXME: Isn't the frame callback enough?
                // self.app_state.render(&wgpu);
            } else if Some(layer) == self.primary_layer.as_ref() {
                self.app_state.primary_resized();
                *self
                    .primary_wgpu
                    .as_mut()
                    .unwrap()
                    .must_reconfigure_with_size
                    .borrow_mut() = Some((new_width, new_height));
            }

            // Kick off the animation
            layer.wl_surface().frame(qh, layer.wl_surface().clone());
            layer.wl_surface().commit();
        }

        if self.secondary_layer.as_ref() == Some(layer) {
            if let Some(anchor_position) = self.secondary_wgpu_holder.take() {
                let wgpu = self.secondary_wgpu.as_ref().unwrap().clone() as Rc<dyn WgpuSurface>;
                self.app_state
                    .configure_secondary_wgpu_surface(&wgpu, anchor_position);
                // Render once to let wgpu finalize the surface initialization.
                // self.app_state.render(&wgpu);
            } else if Some(layer) == self.secondary_layer.as_ref() {
                self.app_state.secondary_resized();
                *self
                    .secondary_wgpu
                    .as_mut()
                    .unwrap()
                    .must_reconfigure_with_size
                    .borrow_mut() = Some((new_width, new_height));
            }

            // Kick off the animation
            layer.wl_surface().frame(qh, layer.wl_surface().clone());
            layer.wl_surface().commit();
        }
    }
}

pub struct WlrWgpuSurface {
    // adapter: wgpu::Adapter,
    device: wgpu::Device,
    surface: wgpu::Surface<'static>,
    queue: Arc<wgpu::Queue>,
    swapchain_format: wgpu::TextureFormat,
    pub layer: LayerSurface,
    pub must_reconfigure_with_size: RefCell<Option<(u32, u32)>>,
}

impl PartialEq for WlrWgpuSurface {
    fn eq(&self, other: &Self) -> bool {
        self.layer.wl_surface().id() == other.layer.wl_surface().id()
    }
}

impl WlrWgpuSurface {
    fn new(
        conn: Connection,
        _qh: QueueHandle<WlrWaylandEventHandler>,
        layer: LayerSurface,
    ) -> WlrWgpuSurface {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create the raw window handle for the surface.
        let raw_display_handle = RawDisplayHandle::Wayland(WaylandDisplayHandle::new(
            NonNull::new(conn.backend().display_ptr() as *mut _).unwrap(),
        ));
        let raw_window_handle = RawWindowHandle::Wayland(WaylandWindowHandle::new(
            NonNull::new(layer.wl_surface().id().as_ptr() as *mut _).unwrap(),
        ));

        let surface = unsafe {
            instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle,
                    raw_window_handle,
                })
                .unwrap()
        };

        // Pick a supported adapter
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("Failed to find suitable adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None))
            .expect("Failed to request device");

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        WlrWgpuSurface {
            // adapter,
            device,
            surface,
            queue: Arc::new(queue),
            swapchain_format,
            layer,
            must_reconfigure_with_size: RefCell::new(Some((1, 1))),
        }
    }
}

impl WgpuSurface for WlrWgpuSurface {
    // fn adapter(&self) -> &wgpu::Adapter {
    //     &self.adapter
    // }
    fn device(&self) -> &wgpu::Device {
        &self.device
    }
    // fn surface_texture(&self) -> &wgpu::SurfaceTexture {
    //     self.surface
    //         .get_current_texture()
    //         .expect("Failed to acquire next swap chain texture")
    // }
    // (&self) -> &wgpu::Surface<'static> {
    //     &self.surface
    // }
    fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }
    fn surface_id(&self) -> u32 {
        self.layer.wl_surface().id().protocol_id()
    }

    fn swapchain_format(&self) -> Option<wgpu::TextureFormat> {
        Some(self.swapchain_format)
    }
}

pub struct WlrWaylandEventHandler {
    ui_msg_rx: Receiver<AppMessage>,
    compositor: CompositorState,
    layer_shell: LayerShell,
    registry_state: RegistryState,
    output_state: OutputState,

    surfaces_with_pending_render: Vec<Rc<WlrWgpuSurface>>,
    pending_render_timestamp: u32,
    // panel_config: PanelConfig,
    request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
    primary_layer: Option<LayerSurface>,
    secondary_layer: Option<LayerSurface>,
    primary_wgpu: Option<Rc<WlrWgpuSurface>>,
    secondary_wgpu: Option<Rc<WlrWgpuSurface>>,
    primary_wgpu_holder: Option<PanelAnchorPosition>,
    secondary_wgpu_holder: Option<PanelAnchorPosition>,
    primary_surface_output: Option<ObjectId>,
    pub app_state: ApplicationState,
    app_state_initialized: bool,
}

pub struct WlrCanvasContext {
    pub conn: Connection,
    pub qh: QueueHandle<WlrWaylandEventHandler>,
}

impl GlobalCanvas for WlrWaylandEventHandler {
    fn app_state(&mut self) -> &mut ApplicationState {
        &mut self.app_state
    }

    fn apply_panel_width_change(&mut self) {
        self.apply_panel_width_change();
    }

    fn apply_panel_exclusive_ratio_change(&mut self) {
        self.apply_panel_exclusive_ratio_change();
    }

    fn apply_panel_layout(&mut self, context: &GlobalCanvasContext) {
        match context {
            GlobalCanvasContext::Wlr(WlrCanvasContext { conn, qh }) => {
                self.apply_panel_layout(&conn, &qh);
            }
            _ => panic!("Incorrect context type"),
        }
        // self.apply_panel_layout(&context.conn, &context.qh);
    }

    fn set_panel_layer(&mut self, layer: PanelLayer) {
        self.set_panel_layer(layer);
    }
}

impl WlrWaylandEventHandler {
    pub fn apply_panel_layout(&mut self, conn: &Connection, qh: &QueueHandle<Self>) {
        // Already destroy existing layers and wgpu surfaces
        self.primary_layer = None;
        self.secondary_layer = None;
        self.app_state.primary_waveform_window = None;
        self.app_state.secondary_waveform_window = None;
        self.app_state.left_waveform_view = None;
        self.app_state.right_waveform_view = None;

        // Unset the request redraw callback, holding references to old surfaces
        *self.request_redraw_callback.lock().unwrap() = Arc::new(|| {});

        let (primary_wgpu_and_anchor, secondary_wgpu_and_anchor) = match self
            .app_state
            .config
            .general
            .layout
        {
            crate::ui::PanelLayout::TwoPanels => (
                Some((
                    self.create_layer_surface(
                        Anchor::TOP | Anchor::LEFT | Anchor::BOTTOM,
                        conn,
                        qh,
                    ),
                    PanelAnchorPosition::Left,
                )),
                Some((
                    self.create_layer_surface(
                        Anchor::TOP | Anchor::RIGHT | Anchor::BOTTOM,
                        conn,
                        qh,
                    ),
                    PanelAnchorPosition::Right,
                )),
            ),
            crate::ui::PanelLayout::SingleTop => (
                Some((
                    self.create_layer_surface(Anchor::LEFT | Anchor::TOP | Anchor::RIGHT, conn, qh),
                    PanelAnchorPosition::Top,
                )),
                None,
            ),
            crate::ui::PanelLayout::SingleBottom => (
                Some((
                    self.create_layer_surface(
                        Anchor::LEFT | Anchor::BOTTOM | Anchor::RIGHT,
                        conn,
                        qh,
                    ),
                    PanelAnchorPosition::Bottom,
                )),
                None,
            ),
        };

        self.primary_layer = primary_wgpu_and_anchor.as_ref().map(|s| s.0.layer.clone());
        self.secondary_layer = secondary_wgpu_and_anchor
            .as_ref()
            .map(|s| s.0.layer.clone());
        (self.primary_wgpu, self.primary_wgpu_holder) = primary_wgpu_and_anchor.unzip();
        (self.secondary_wgpu, self.secondary_wgpu_holder) = secondary_wgpu_and_anchor.unzip();

        if let Some(wgpu) = self.primary_wgpu.as_ref() {
            self.surfaces_with_pending_render.push(wgpu.clone());
        }
        if let Some(wgpu) = self.secondary_wgpu.as_ref() {
            self.surfaces_with_pending_render.push(wgpu.clone());
        }
        self.update_request_redraw_callback(conn.clone(), qh.clone());
    }

    fn create_layer_surface(
        &self,
        anchor: Anchor,
        conn: &Connection,
        qh: &QueueHandle<Self>,
    ) -> Rc<WlrWgpuSurface> {
        // A layer surface is created from a surface.
        let surface = self.compositor.create_surface(&qh);
        // Let mouse events pass through the surface
        surface.set_input_region(Some(Region::new(&self.compositor).unwrap().wl_region()));

        // And then we create the layer shell.
        let layer =
            self.layer_shell
                .create_layer_surface(&qh, surface, Layer::Top, None::<String>, None);
        // Configure the layer surface, providing things like the anchor on screen, desired size and the keyboard
        // interactivity
        layer.set_anchor(anchor);
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);
        let (panel_layer, panel_width, exclusive_ratio) = match self.app_state.config.style {
            ui::Style::Compressed => (
                self.app_state.config.compressed.layer,
                self.app_state.config.compressed.width as u32,
                self.app_state.config.compressed.exclusive_ratio,
            ),
            ui::Style::Ridgeline => (
                self.app_state.config.ridgeline.layer,
                self.app_state.config.ridgeline.width as u32,
                self.app_state.config.ridgeline.exclusive_ratio,
            ),
        };
        if !anchor.intersects(Anchor::BOTTOM) || !anchor.intersects(Anchor::TOP) {
            layer.set_size(0, panel_width);
        } else {
            layer.set_size(panel_width, 0);
        }
        layer.set_exclusive_zone((panel_width as f32 * exclusive_ratio) as i32);
        layer.set_layer(Self::layer_to_wayland_layer(panel_layer));

        // In order for the layer surface to be mapped, we need to perform an initial commit with no attached\
        // buffer. For more info, see WaylandSurface::commit
        //
        // The compositor will respond with an initial configure that we can then use to present to the layer
        // surface with the correct options.
        layer.commit();

        let qh = qh.clone();
        Rc::new(WlrWgpuSurface::new(conn.clone(), qh, layer))
    }

    pub fn apply_panel_width_change(&mut self) {
        let panel_width = match self.app_state.config.style {
            ui::Style::Compressed => self.app_state.config.compressed.width as u32,
            ui::Style::Ridgeline => self.app_state.config.ridgeline.width as u32,
        };

        let (width, height) = match self.app_state.config.general.layout {
            crate::ui::PanelLayout::SingleTop | crate::ui::PanelLayout::SingleBottom => {
                (0, panel_width as u32)
            }
            crate::ui::PanelLayout::TwoPanels => (panel_width as u32, 0),
        };

        if let Some(layer) = self.primary_layer.as_ref() {
            layer.set_size(width, height);
            layer.commit();
        }
        if let Some(layer) = self.secondary_layer.as_ref() {
            layer.set_size(width, height);
            layer.commit();
        }
        // Apply the ratio relatively to the new width
        self.apply_panel_exclusive_ratio_change();
    }

    pub fn apply_panel_exclusive_ratio_change(&mut self) {
        let (panel_width, exclusive_ratio) = match self.app_state.config.style {
            ui::Style::Compressed => (
                self.app_state.config.compressed.width as f32,
                self.app_state.config.compressed.exclusive_ratio,
            ),
            ui::Style::Ridgeline => (
                self.app_state.config.ridgeline.width as f32,
                self.app_state.config.ridgeline.exclusive_ratio,
            ),
        };

        // The window manager won't protect against taking all the screen as exclusive, so leave 1/3
        // available for windows.
        let max_exclusive_zone = match self.app_state.config.general.layout {
            crate::ui::PanelLayout::SingleTop | crate::ui::PanelLayout::SingleBottom => {
                self.app_state.screen_size.1 as f32 * 0.66
            }
            crate::ui::PanelLayout::TwoPanels => self.app_state.screen_size.0 as f32 * 0.33,
        };
        let value = max_exclusive_zone.min(panel_width * exclusive_ratio);

        if let Some(layer) = self.primary_layer.as_ref() {
            layer.set_exclusive_zone(value as i32);
            layer.commit();
        }
        if let Some(layer) = self.secondary_layer.as_ref() {
            layer.set_exclusive_zone(value as i32);
            layer.commit();
        }
    }

    fn layer_to_wayland_layer(layer: PanelLayer) -> Layer {
        match layer {
            PanelLayer::Overlay => Layer::Overlay,
            PanelLayer::Top => Layer::Top,
            PanelLayer::Bottom => Layer::Bottom,
            PanelLayer::Background => Layer::Background,
        }
    }

    pub fn set_panel_layer(&mut self, layer: PanelLayer) {
        let wayland_layer = Self::layer_to_wayland_layer(layer);
        if let Some(layer) = self.primary_layer.as_ref() {
            layer.set_layer(wayland_layer);
            layer.commit();
        }
        if let Some(layer) = self.secondary_layer.as_ref() {
            layer.set_layer(wayland_layer);
            layer.commit();
        }
    }

    fn update_request_redraw_callback(&mut self, conn: Connection, qh: QueueHandle<Self>) {
        *self.request_redraw_callback.lock().unwrap() = {
            let primary_surface = self.primary_layer.as_ref().map(|l| l.wl_surface().clone());
            let secondary_surface = self
                .secondary_layer
                .as_ref()
                .map(|l| l.wl_surface().clone());
            Arc::new(move || {
                if let Some(surface) = &primary_surface {
                    surface.frame(&qh, surface.clone());
                    surface.commit();
                }
                if let Some(surface) = &secondary_surface {
                    surface.frame(&qh, surface.clone());
                    surface.commit();
                }
                // The event loop might be idle, make sure the requests make it to the compositor
                // so that we can get a response that gets us out of polling.
                conn.flush().unwrap();
            })
        };
    }
}

delegate_compositor!(WlrWaylandEventHandler);
delegate_output!(WlrWaylandEventHandler);

delegate_layer!(WlrWaylandEventHandler);

delegate_registry!(WlrWaylandEventHandler);

impl ProvidesRegistryState for WlrWaylandEventHandler {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState];
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum PanelAnchorPosition {
    Top,
    Bottom,
    Left,
    Right,
}

pub struct WlrWaylandEventLoop {
    event_queue: EventQueue<WlrWaylandEventHandler>,
    pub state: WlrWaylandEventHandler,
}

impl WlrWaylandEventLoop {
    pub fn new(
        app_state: ApplicationState,
        ui_msg_rx: Receiver<AppMessage>,
        request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
    ) -> WlrWaylandEventLoop {
        // All Wayland apps start by connecting the compositor (server).
        let conn = Connection::connect_to_env().expect(
            "Failed to connect to a Wayland compositor with wlr_layer_shell support \
            (X.Org and/or Gnome's Mutter are not supported).",
        );

        // Enumerate the list of globals to get the protocols the server implements.
        let (globals, event_queue) = registry_queue_init::<WlrWaylandEventHandler>(&conn).unwrap();
        let qh = event_queue.handle();

        // The compositor (not to be confused with the server which is commonly called the compositor) allows
        // configuring surfaces to be presented.
        let compositor =
            CompositorState::bind(&globals, &qh).expect("wl_compositor is not available");
        // This app uses the wlr layer shell, which may not be available with every compositor.
        let layer_shell = LayerShell::bind(&globals, &qh).expect(
            "Only Wayland compositors with wlr_layer_shell are supported. \
            https://wayland.app/protocols/wlr-layer-shell-unstable-v1#compositor-support",
        );

        let registry_state = RegistryState::new(&globals);
        let output_state = OutputState::new(&globals, &qh);

        let mut state = WlrWaylandEventHandler {
            ui_msg_rx,
            compositor,
            layer_shell,
            registry_state,
            output_state,
            surfaces_with_pending_render: Vec::new(),
            pending_render_timestamp: 0,
            request_redraw_callback,
            primary_layer: None,
            secondary_layer: None,
            primary_wgpu: None,
            secondary_wgpu: None,
            primary_wgpu_holder: None,
            secondary_wgpu_holder: None,
            primary_surface_output: None,
            app_state,
            app_state_initialized: false,
        };

        state.apply_panel_layout(&conn, &qh);
        state.update_request_redraw_callback(conn, qh);

        WlrWaylandEventLoop { event_queue, state }
    }

    pub fn run_event_loop(&mut self) {
        // We don't draw immediately, the configure will notify us when to first draw.
        loop {
            self.event_queue.blocking_dispatch(&mut self.state).unwrap();

            // With two windows, we have to render outside the frame callback, after we've
            // drained the pending wayland messages, so that we can request new frame callbacks
            // for both windows before we start interacting with the GPU for either of them
            // and potentially block the thread.
            if !self.state.surfaces_with_pending_render.is_empty() {
                self.state
                    .app_state
                    .process_audio(self.state.pending_render_timestamp);
            }
            for wgpu in self.state.surfaces_with_pending_render.drain(..) {
                if let Some((width, height)) = wgpu.must_reconfigure_with_size.take() {
                    let config = wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: wgpu.swapchain_format,
                        width: width.max(1),
                        height: height.max(1),
                        present_mode: wgpu::PresentMode::AutoVsync,
                        desired_maximum_frame_latency: 1,
                        alpha_mode: wgpu::CompositeAlphaMode::PreMultiplied,
                        view_formats: vec![],
                    };

                    // Reconfigure the wgpu surface if needed.
                    // wgpu.device.configure_surface(&wgpu.surface);
                    wgpu.surface.configure(wgpu.device(), &config);
                    // *wgpu.must_reconfigure_with_size.borrow_mut() = false;
                }
                let frame = wgpu
                    .surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                self.state
                    .app_state
                    .render(&(wgpu as Rc<dyn WgpuSurface>), &frame);
                frame.present();
            }
        }
    }
}
