// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
};
use smithay_client_toolkit::compositor::Region;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use tracing::{Level, instrument, span, warn};
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
    Connection, Dispatch, Proxy, QueueHandle,
    globals::registry_queue_init,
    protocol::{wl_output, wl_surface},
};
use wayland_protocols::wp::fractional_scale::v1::client::{
    wp_fractional_scale_manager_v1, wp_fractional_scale_v1,
};
use wayland_protocols::wp::viewporter::client::{wp_viewport, wp_viewporter};

use crate::ui::{self, PanelLayer};
use crate::{AppMessageCallback, ApplicationState, GlobalCanvas, GlobalCanvasContext, WgpuSurface};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SurfaceKind {
    Primary,
    Secondary,
}

impl CompositorHandler for WaylandEventHandler {
    #[instrument(skip(self))]
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_factor: i32,
    ) {
        // We only use fractional scale protocol, ignore wl_output integer scale
    }

    #[instrument(skip(self))]
    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
    }

    #[instrument(skip(self))]
    fn frame(
        &mut self,
        conn: &Connection,
        qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        time: u32,
    ) {
        // Process UI callbacks here since some require the wayland connection to recreate windows.
        while let Ok(message) = self.ui_msg_rx.try_recv() {
            let _span = span!(Level::INFO, "ui_msg_rx handler", message = ?message).entered();
            match message {
                AppMessageCallback::ApplicationState(closure) => closure(&mut self.app_state),
                AppMessageCallback::LayerShellGlobalCanvas(closure) => closure(
                    self,
                    GlobalCanvasContext::LayerShell(LayerShellCanvasContext {
                        conn: conn.clone(),
                        qh: qh.clone(),
                    }),
                ),
                AppMessageCallback::SlintGlobalCanvas(_) => {
                    panic!("Incorrect GlobalCanvas callback type")
                }
            }
        }

        if let Some(wgpu) = match surface.id() {
            id if self
                .primary_layer
                .as_ref()
                .is_some_and(|l| l.wl_surface().id() == id) =>
            {
                self.primary_wgpu.as_ref()
            }
            id if self
                .secondary_layer
                .as_ref()
                .is_some_and(|l| l.wl_surface().id() == id) =>
            {
                self.secondary_wgpu.as_ref()
            }
            _ => {
                // A frame might have been requested, the callback read from the socket but still be
                // pending in the event loop before we swap surfaces during layout changes,
                // so ignore any unknown surface.
                None
            }
        } && !self.surfaces_with_pending_render.contains(wgpu)
        {
            self.surfaces_with_pending_render.push(wgpu.clone());

            if !self.app_state.animation_stopped.load(Ordering::Relaxed) {
                // Already tell the compositor that we want to draw again for the next output frame.
                surface.frame(qh, surface.clone());
                surface.commit();
            } else {
                self.app_state.hide_fps_counters();
            }
        }

        if self.primary_layer.as_ref().map(|l| l.wl_surface().id()) == Some(surface.id()) {
            self.pending_render_timestamp = time;
        }
    }

    #[instrument(skip(self))]
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
            if let Some(info) = self.output_state.info(output) {
                if let Some(size) = info.logical_size {
                    self.app_state
                        .update_screen_size(size.0 as u32, size.1 as u32);
                    // Adjust the width and exclusive ratio.
                    self.apply_panel_width_change();
                }
            }
        }
    }

    #[instrument(skip(self))]
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

impl OutputHandler for WaylandEventHandler {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    #[instrument(skip(self))]
    fn new_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    #[instrument(skip(self))]
    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        if self.primary_surface_output == Some(output.id())
            && let Some(info) = self.output_state.info(&output)
        {
            if let Some(size) = info.logical_size {
                self.app_state
                    .update_screen_size(size.0 as u32, size.1 as u32);
            }
        }
    }

    #[instrument(skip(self))]
    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }
}

impl LayerShellHandler for WaylandEventHandler {
    #[instrument(skip(self))]
    fn closed(&mut self, conn: &Connection, qh: &QueueHandle<Self>, _layer: &LayerSurface) {
        // The client should destroy the resource after receiving this event,
        // and create a new surface if they so choose.
        self.apply_panel_layout(conn, qh);
    }

    #[instrument(skip(self))]
    fn configure(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        if !self.app_state_initialized {
            self.app_state_initialized = true;
            self.app_state.initialize_audio_and_transform_thread();
        }

        let (new_width, new_height) = configure.new_size;

        if self.primary_layer.as_ref() == Some(layer) {
            let layer_wgpu = self.primary_wgpu.as_ref().unwrap().clone();
            self.primary_configured_size = Some((new_width.max(1), new_height.max(1)));
            self.set_viewport_destination(SurfaceKind::Primary, new_width, new_height);
            self.configure_layer_surface(layer, &layer_wgpu, qh, new_width, new_height, true);

            if !self.primary_wgpu_configured {
                let wgpu = layer_wgpu.clone() as Rc<dyn WgpuSurface>;
                self.app_state.initialize_primary_view_surface(&wgpu);
                self.primary_wgpu_configured = true;
            }
        }

        if self.secondary_layer.as_ref() == Some(layer) {
            let layer_wgpu = self.secondary_wgpu.as_ref().unwrap().clone();
            self.secondary_configured_size = Some((new_width.max(1), new_height.max(1)));
            self.set_viewport_destination(SurfaceKind::Secondary, new_width, new_height);
            self.configure_layer_surface(layer, &layer_wgpu, qh, new_width, new_height, true);

            if !self.secondary_wgpu_configured {
                let wgpu = layer_wgpu.clone() as Rc<dyn WgpuSurface>;
                self.app_state.initialize_secondary_view_surface(&wgpu);
                self.secondary_wgpu_configured = true;
            }
        }
    }
}

pub struct LayerShellWgpuSurface {
    device: wgpu::Device,
    surface: wgpu::Surface<'static>,
    queue: Arc<wgpu::Queue>,
    swapchain_format: wgpu::TextureFormat,
    pub layer: LayerSurface,
}

impl PartialEq for LayerShellWgpuSurface {
    fn eq(&self, other: &Self) -> bool {
        self.layer.wl_surface().id() == other.layer.wl_surface().id()
    }
}

impl LayerShellWgpuSurface {
    #[instrument]
    fn new(
        conn: Connection,
        _qh: QueueHandle<WaylandEventHandler>,
        layer: LayerSurface,
    ) -> LayerShellWgpuSurface {
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

        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default()))
            .expect("Failed to request device");

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities
            .formats
            .iter()
            .find(|f| {
                matches!(
                    f,
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
                )
            })
            .unwrap_or(&swapchain_capabilities.formats[0])
            .clone();

        LayerShellWgpuSurface {
            device,
            surface,
            queue: Arc::new(queue),
            swapchain_format,
            layer,
        }
    }
}

impl WgpuSurface for LayerShellWgpuSurface {
    fn device(&self) -> &wgpu::Device {
        &self.device
    }

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

pub struct WaylandEventHandler {
    ui_msg_rx: Receiver<AppMessageCallback>,
    compositor: CompositorState,
    layer_shell: LayerShell,
    registry_state: RegistryState,
    output_state: OutputState,
    fractional_scale: Option<f64>,

    // Fractional scale staging protocol and viewporter support
    viewporter: Option<wp_viewporter::WpViewporter>,
    fractional_scale_manager: Option<wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1>,
    primary_viewport: Option<wp_viewport::WpViewport>,
    secondary_viewport: Option<wp_viewport::WpViewport>,
    primary_fractional_scale: Option<wp_fractional_scale_v1::WpFractionalScaleV1>,
    secondary_fractional_scale: Option<wp_fractional_scale_v1::WpFractionalScaleV1>,

    surfaces_with_pending_render: Vec<Rc<LayerShellWgpuSurface>>,
    pending_render_timestamp: u32,
    request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
    primary_layer: Option<LayerSurface>,
    secondary_layer: Option<LayerSurface>,
    primary_wgpu: Option<Rc<LayerShellWgpuSurface>>,
    secondary_wgpu: Option<Rc<LayerShellWgpuSurface>>,
    primary_wgpu_configured: bool,
    secondary_wgpu_configured: bool,
    primary_surface_output: Option<ObjectId>,
    primary_configured_size: Option<(u32, u32)>,
    secondary_configured_size: Option<(u32, u32)>,
    pub app_state: ApplicationState,
    app_state_initialized: bool,
}

pub struct LayerShellCanvasContext {
    pub conn: Connection,
    pub qh: QueueHandle<WaylandEventHandler>,
}

/// Test if Wayland with wlr_layer_shell support is available without full initialization.
pub fn can_use_wayland() -> bool {
    use wayland_client::Dispatch;
    use wayland_client::protocol::wl_registry;

    // Test if we can connect to Wayland
    let conn = match Connection::connect_to_env() {
        Ok(conn) => conn,
        Err(_) => return false,
    };

    // Get the display and registry
    let display = conn.display();
    let mut event_queue = conn.new_event_queue();
    let qh = event_queue.handle();

    // Simple state to track if we found the layer shell global
    struct GlobalCheck {
        has_layer_shell: bool,
    }

    impl Dispatch<wl_registry::WlRegistry, ()> for GlobalCheck {
        fn event(
            state: &mut Self,
            _registry: &wl_registry::WlRegistry,
            event: wl_registry::Event,
            _data: &(),
            _conn: &Connection,
            _qhandle: &QueueHandle<Self>,
        ) {
            if let wl_registry::Event::Global { interface, .. } = event {
                if interface == "zwlr_layer_shell_v1" {
                    state.has_layer_shell = true;
                }
            }
        }
    }

    impl Dispatch<wayland_client::protocol::wl_display::WlDisplay, ()> for GlobalCheck {
        fn event(
            _: &mut Self,
            _: &wayland_client::protocol::wl_display::WlDisplay,
            _: wayland_client::protocol::wl_display::Event,
            _: &(),
            _: &Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }

    let mut state = GlobalCheck {
        has_layer_shell: false,
    };
    let _registry = display.get_registry(&qh, ());

    // Roundtrip to get all globals
    if event_queue.roundtrip(&mut state).is_err() {
        return false;
    }

    state.has_layer_shell
}

impl GlobalCanvas for WaylandEventHandler {
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
            GlobalCanvasContext::LayerShell(LayerShellCanvasContext { conn, qh }) => {
                self.apply_panel_layout(conn, qh);
            }
            _ => panic!("Incorrect context type"),
        }
        // self.apply_panel_layout(&context.conn, &context.qh);
    }

    fn set_panel_layer(&mut self, layer: PanelLayer) {
        self.set_panel_layer(layer);
    }
}

impl WaylandEventHandler {
    #[instrument(skip(self))]
    pub fn apply_panel_layout(&mut self, conn: &Connection, qh: &QueueHandle<Self>) {
        // Already destroy existing layers and wgpu surfaces
        self.primary_layer = None;
        self.secondary_layer = None;
        self.primary_wgpu = None;
        self.secondary_wgpu = None;
        self.surfaces_with_pending_render.clear();
        self.primary_configured_size = None;
        self.secondary_configured_size = None;
        self.primary_viewport = None;
        self.secondary_viewport = None;
        self.primary_fractional_scale = None;
        self.secondary_fractional_scale = None;
        self.app_state.primary_view_surface = None;
        self.app_state.secondary_view_surface = None;
        self.app_state.left_view = None;
        self.app_state.right_view = None;

        // Unset the request redraw callback, holding references to old surfaces
        *self.request_redraw_callback.lock().unwrap() = Arc::new(|| {});

        let (primary_wgpu, secondary_wgpu) = match self.app_state.config.general.layout {
            crate::ui::PanelLayout::TwoPanels => (
                Some(self.create_layer_surface(
                    SurfaceKind::Primary,
                    Anchor::TOP | Anchor::LEFT | Anchor::BOTTOM,
                    conn,
                    qh,
                )),
                Some(self.create_layer_surface(
                    SurfaceKind::Secondary,
                    Anchor::TOP | Anchor::RIGHT | Anchor::BOTTOM,
                    conn,
                    qh,
                )),
            ),
            crate::ui::PanelLayout::SingleTop => (
                Some(self.create_layer_surface(
                    SurfaceKind::Primary,
                    Anchor::LEFT | Anchor::TOP | Anchor::RIGHT,
                    conn,
                    qh,
                )),
                None,
            ),
            crate::ui::PanelLayout::SingleBottom => (
                Some(self.create_layer_surface(
                    SurfaceKind::Primary,
                    Anchor::LEFT | Anchor::BOTTOM | Anchor::RIGHT,
                    conn,
                    qh,
                )),
                None,
            ),
        };

        self.primary_layer = primary_wgpu.as_ref().map(|s| s.layer.clone());
        self.secondary_layer = secondary_wgpu.as_ref().map(|s| s.layer.clone());
        self.primary_wgpu = primary_wgpu;
        self.secondary_wgpu = secondary_wgpu;
        self.primary_wgpu_configured = false;
        self.secondary_wgpu_configured = false;
        self.update_request_redraw_callback(conn.clone(), qh.clone());
    }

    #[instrument(skip(self))]
    fn create_layer_surface(
        &mut self,
        kind: SurfaceKind,
        anchor: Anchor,
        conn: &Connection,
        qh: &QueueHandle<Self>,
    ) -> Rc<LayerShellWgpuSurface> {
        // A layer surface is created from a surface.
        let surface = self.compositor.create_surface(qh);
        // Let mouse events pass through the surface
        surface.set_input_region(Some(Region::new(&self.compositor).unwrap().wl_region()));

        // And then we create the layer shell.
        let layer =
            self.layer_shell
                .create_layer_surface(qh, surface, Layer::Top, None::<String>, None);
        // Configure the layer surface, providing things like the anchor on screen, desired size and the keyboard
        // interactivity
        layer.set_anchor(anchor);
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);
        let panel_layer = match self.app_state.config.style {
            ui::Style::Compressed => self.app_state.config.compressed.layer,
            ui::Style::Ridgeline => self.app_state.config.ridgeline.layer,
            ui::Style::RidgelineFrequency => self.app_state.config.ridgeline_frequency.layer,
        };

        // Set width to 1 pixel until we get an output and size assigned by surface_enter
        if !anchor.intersects(Anchor::BOTTOM) || !anchor.intersects(Anchor::TOP) {
            layer.set_size(0, 1);
        } else {
            layer.set_size(1, 0);
        }
        layer.set_layer(Self::layer_to_wayland_layer(panel_layer));

        // Attach viewporter and fractional-scale objects if the compositor supports them.
        let viewport = self
            .viewporter
            .as_ref()
            .map(|vp| vp.get_viewport(&layer.wl_surface(), qh, kind));
        let fractional_scale = self
            .fractional_scale_manager
            .as_ref()
            .map(|mgr| mgr.get_fractional_scale(&layer.wl_surface(), qh, kind));

        match kind {
            SurfaceKind::Primary => {
                self.primary_viewport = viewport;
                self.primary_fractional_scale = fractional_scale;
            }
            SurfaceKind::Secondary => {
                self.secondary_viewport = viewport;
                self.secondary_fractional_scale = fractional_scale;
            }
        }

        // In order for the layer surface to be mapped, we need to perform an initial commit with no attached\
        // buffer. For more info, see WaylandSurface::commit
        //
        // The compositor will respond with an initial configure that we can then use to present to the layer
        // surface with the correct options.
        layer.commit();

        let qh = qh.clone();
        Rc::new(LayerShellWgpuSurface::new(conn.clone(), qh, layer))
    }

    #[instrument(skip(self))]
    pub fn apply_panel_width_change(&mut self) {
        // Compute panel width (ratio from configuration) into pixels depending on layout
        let panel_width_ratio = match self.app_state.config.style {
            ui::Style::Compressed => self.app_state.config.compressed.width_ratio,
            ui::Style::Ridgeline => self.app_state.config.ridgeline.width_ratio,
            ui::Style::RidgelineFrequency => self.app_state.config.ridgeline_frequency.width_ratio,
        };

        let (width, height) = match self.app_state.config.general.layout {
            crate::ui::PanelLayout::SingleTop | crate::ui::PanelLayout::SingleBottom => (
                0,
                (panel_width_ratio * self.app_state.screen_size.1 as f32) as u32,
            ),
            crate::ui::PanelLayout::TwoPanels => (
                (panel_width_ratio * self.app_state.screen_size.0 as f32) as u32,
                0,
            ),
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

    #[instrument(skip(self))]
    pub fn apply_panel_exclusive_ratio_change(&mut self) {
        let (panel_width_ratio, exclusive_ratio) = match self.app_state.config.style {
            ui::Style::Compressed => (
                self.app_state.config.compressed.width_ratio,
                self.app_state.config.compressed.exclusive_ratio,
            ),
            ui::Style::Ridgeline => (
                self.app_state.config.ridgeline.width_ratio,
                self.app_state.config.ridgeline.exclusive_ratio,
            ),
            ui::Style::RidgelineFrequency => (
                self.app_state.config.ridgeline_frequency.width_ratio,
                self.app_state.config.ridgeline_frequency.exclusive_ratio,
            ),
        };

        // The window manager won't protect against taking all the screen as exclusive, so leave 1/3
        // available for windows. Compute panel pixel width first.
        let panel_pixel_width = match self.app_state.config.general.layout {
            crate::ui::PanelLayout::SingleTop | crate::ui::PanelLayout::SingleBottom => {
                panel_width_ratio * self.app_state.screen_size.1 as f32
            }
            crate::ui::PanelLayout::TwoPanels => {
                panel_width_ratio * self.app_state.screen_size.0 as f32
            }
        };

        let max_exclusive_zone = match self.app_state.config.general.layout {
            crate::ui::PanelLayout::SingleTop | crate::ui::PanelLayout::SingleBottom => {
                self.app_state.screen_size.1 as f32 * 0.66
            }
            crate::ui::PanelLayout::TwoPanels => self.app_state.screen_size.0 as f32 * 0.33,
        };
        let value = max_exclusive_zone.min(panel_pixel_width * exclusive_ratio);

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

    #[instrument(skip(self))]
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
                let _span = span!(Level::INFO, "request_redraw_callback").entered();
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

    fn configure_layer_surface(
        &mut self,
        layer: &LayerSurface,
        wgpu: &Rc<LayerShellWgpuSurface>,
        qh: &QueueHandle<Self>,
        new_width: u32,
        new_height: u32,
        request_frame: bool,
    ) {
        let scale = self.effective_scale();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu.swapchain_format,
            width: ((new_width.max(1) as f64 * scale).ceil() as u32).max(1),
            height: ((new_height.max(1) as f64 * scale).ceil() as u32).max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 1,
            alpha_mode: wgpu::CompositeAlphaMode::PreMultiplied,
            view_formats: vec![],
        };
        wgpu.surface.configure(wgpu.device(), &config);

        self.surfaces_with_pending_render.push(wgpu.clone());

        if request_frame {
            // Kick off the animation
            layer.wl_surface().frame(qh, layer.wl_surface().clone());
            layer.wl_surface().commit();
        }
    }

    fn effective_scale(&self) -> f64 {
        self.fractional_scale.unwrap_or(1.0)
    }

    fn set_viewport_destination(&self, kind: SurfaceKind, width: u32, height: u32) {
        let viewport = match kind {
            SurfaceKind::Primary => self.primary_viewport.as_ref(),
            SurfaceKind::Secondary => self.secondary_viewport.as_ref(),
        };

        if let Some(viewport) = viewport {
            viewport.set_destination(width as i32, height as i32);
        }
    }

    fn update_fractional_scale(&mut self, new_scale_120: u32, qh: &QueueHandle<Self>) {
        const DENOMINATOR: f64 = 120.0;
        let new_scale = (new_scale_120 as f64 / DENOMINATOR).max(1.0);

        // Without viewporter support we cannot correctly map non-integer scales.
        if self.viewporter.is_none() {
            return;
        }

        if self.fractional_scale == Some(new_scale) {
            return;
        }

        self.fractional_scale = Some(new_scale);

        self.reconfigure_swapchains_for_scale(qh);
    }

    fn reconfigure_swapchains_for_scale(&mut self, qh: &QueueHandle<Self>) {
        if let (Some(layer), Some(wgpu), Some((width, height))) = (
            self.primary_layer.clone(),
            self.primary_wgpu.clone(),
            self.primary_configured_size,
        ) {
            self.configure_layer_surface(&layer, &wgpu, qh, width, height, false);
        }

        if let (Some(layer), Some(wgpu), Some((width, height))) = (
            self.secondary_layer.clone(),
            self.secondary_wgpu.clone(),
            self.secondary_configured_size,
        ) {
            self.configure_layer_surface(&layer, &wgpu, qh, width, height, false);
        }
    }

    #[instrument(skip(self))]
    fn render_pending(&mut self) {
        // With two windows, we have to render outside the frame callback, after we've
        // drained the pending wayland messages, so that we can request new frame callbacks
        // for both windows before we start interacting with the GPU for either of them
        // and potentially block the thread.
        if !self.surfaces_with_pending_render.is_empty() {
            self.app_state.process_audio(self.pending_render_timestamp);
        }
        let clear_color = wgpu::Color::TRANSPARENT;
        for wgpu in self.surfaces_with_pending_render.drain(..) {
            let frame = wgpu
                .surface
                .get_current_texture()
                .expect("Failed to acquire next swap chain texture");
            self.app_state.render_with_clear_color(
                &(wgpu as Rc<dyn WgpuSurface>),
                &frame.texture,
                clear_color,
            );

            let _span = span!(Level::INFO, "Presenting frame").entered();
            frame.present();
        }
    }
}

delegate_compositor!(WaylandEventHandler);
delegate_output!(WaylandEventHandler);

delegate_layer!(WaylandEventHandler);

delegate_registry!(WaylandEventHandler);

// Those Dispatch types are not implemented by smithay client toolkit, so we have to do it ourselves.
impl Dispatch<wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1, ()>
    for WaylandEventHandler
{
    fn event(
        _state: &mut Self,
        _proxy: &wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1,
        _event: wp_fractional_scale_manager_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wp_fractional_scale_v1::WpFractionalScaleV1, SurfaceKind> for WaylandEventHandler {
    fn event(
        state: &mut Self,
        _proxy: &wp_fractional_scale_v1::WpFractionalScaleV1,
        event: wp_fractional_scale_v1::Event,
        _data: &SurfaceKind,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wp_fractional_scale_v1::Event::PreferredScale { scale } = event {
            state.update_fractional_scale(scale, qh);
        }
    }
}

impl Dispatch<wp_viewporter::WpViewporter, ()> for WaylandEventHandler {
    fn event(
        _state: &mut Self,
        _proxy: &wp_viewporter::WpViewporter,
        _event: wp_viewporter::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wp_viewport::WpViewport, SurfaceKind> for WaylandEventHandler {
    fn event(
        _state: &mut Self,
        _proxy: &wp_viewport::WpViewport,
        _event: wp_viewport::Event,
        _data: &SurfaceKind,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl ProvidesRegistryState for WaylandEventHandler {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState];
}

pub struct WaylandEventLoop {
    event_queue: EventQueue<WaylandEventHandler>,
    pub state: WaylandEventHandler,
}

impl WaylandEventLoop {
    #[instrument(skip(app_state, ui_msg_rx, request_redraw_callback))]
    pub fn new(
        app_state: ApplicationState,
        ui_msg_rx: Receiver<AppMessageCallback>,
        request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
    ) -> WaylandEventLoop {
        // All Wayland apps start by connecting the compositor (server).
        let conn = Connection::connect_to_env().expect(
            "Wayland compositor connection failed despite pre-check. \
            This shouldn't happen unless the compositor became unavailable.",
        );

        // Enumerate the list of globals to get the protocols the server implements.
        let (globals, event_queue) = registry_queue_init::<WaylandEventHandler>(&conn).unwrap();
        let qh = event_queue.handle();

        // The compositor (not to be confused with the server which is commonly called the compositor) allows
        // configuring surfaces to be presented.
        let compositor =
            CompositorState::bind(&globals, &qh).expect("wl_compositor is not available");
        // This app uses the wlr layer shell, which may not be available with every compositor.
        let layer_shell = LayerShell::bind(&globals, &qh).expect(
            "wlr_layer_shell support lost despite pre-check. \
            This shouldn't happen unless the compositor became unavailable.",
        );

        let registry_state = RegistryState::new(&globals);
        let output_state = OutputState::new(&globals, &qh);

        let viewporter = globals.bind(&qh, 1..=1, ()).ok();
        let fractional_scale_manager = globals.bind(&qh, 1..=1, ()).ok();

        let mut state = WaylandEventHandler {
            ui_msg_rx,
            compositor,
            layer_shell,
            registry_state,
            output_state,
            fractional_scale: None,
            viewporter,
            fractional_scale_manager,
            primary_viewport: None,
            secondary_viewport: None,
            primary_fractional_scale: None,
            secondary_fractional_scale: None,
            surfaces_with_pending_render: Vec::new(),
            pending_render_timestamp: 0,
            request_redraw_callback,
            primary_layer: None,
            secondary_layer: None,
            primary_wgpu: None,
            secondary_wgpu: None,
            primary_wgpu_configured: false,
            secondary_wgpu_configured: false,
            primary_surface_output: None,
            primary_configured_size: None,
            secondary_configured_size: None,
            app_state,
            app_state_initialized: false,
        };

        state.apply_panel_layout(&conn, &qh);
        state.update_request_redraw_callback(conn, qh);

        WaylandEventLoop { event_queue, state }
    }

    #[instrument(skip(self))]
    pub fn run_event_loop(&mut self) {
        // We don't draw immediately, the configure will notify us when to first draw.
        loop {
            self.event_queue.blocking_dispatch(&mut self.state).unwrap();
            self.state.render_pending();
        }
    }
}
