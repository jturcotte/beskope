use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
};
use smithay_client_toolkit::compositor::Region;
use std::ptr::NonNull;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use wayland_client::EventQueue;

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
    globals::registry_queue_init,
    protocol::{wl_output, wl_surface},
    Connection, Proxy, QueueHandle,
};

use crate::{ApplicationState, PanelConfiguration, PanelLayer, UiMessage, WgpuSurface};

impl CompositorHandler for WlrWaylandEventHandler {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_factor: i32,
    ) {
        println!("Scale factor changed");
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
        println!("Transform changed");
    }

    fn frame(
        &mut self,
        conn: &Connection,
        qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _time: u32,
    ) {
        while let Ok(message) = self.ui_msg_rx.try_recv() {
            match message {
                UiMessage::ApplicationStateCallback(closure) => {
                    closure(self.app_state.windowed_state.as_mut().unwrap())
                }
                UiMessage::WlrWaylandEventHandlerCallback(closure) => closure(self, conn, qh),
            }
        }
        self.app_state.render();
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
        println!("Surface entered");
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
        println!("Surface left");
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
        println!("New output");
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
        println!("Update output");
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
        println!("Output destroyed");
    }
}

impl LayerShellHandler for WlrWaylandEventHandler {
    fn closed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _layer: &LayerSurface) {
        println!("Layer closed");
        // FIXME: handle instead
        self.exit = true;
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let (new_width, new_height) = configure.new_size;
        println!("Configure: {new_width}x{new_height}");

        if !self.app_state_initialized {
            self.app_state_initialized = true;
            self.app_state.initialize_app_state();
        }
        if let Some((wgpu_surface, anchor_position)) = self.left_wgpu_holder.take() {
            self.app_state
                .configure_left_wgpu_surface(wgpu_surface, anchor_position);
        }
        if let Some((wgpu_surface, anchor_position)) = self.right_wgpu_holder.take() {
            self.app_state
                .configure_right_wgpu_surface(wgpu_surface, anchor_position);
        }

        if Some(layer) == self.left_layer.as_ref() {
            self.app_state.left_resized(new_width, new_height);
        }
        if Some(layer) == self.right_layer.as_ref() {
            self.app_state.right_resized(new_width, new_height);
        }

        if Some(layer) == self.right_layer.as_ref() || self.right_layer.is_none() {
            // Render once to let wgpu finalize the surface initialization.
            // FIXME: I get a "layer_surface has never been configured" error from wlroots
            // if I do this after the first layer, but putting this here seems to work.
            self.app_state.render();
        }
    }
}

pub struct WlrWgpuSurface {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    surface: wgpu::Surface<'static>,
    queue: Arc<wgpu::Queue>,
    request_redraw_callback: Arc<dyn Fn() + Send + Sync>,

    pub layer: LayerSurface,
}

impl WlrWgpuSurface {
    fn new(
        conn: Connection,
        qh: QueueHandle<WlrWaylandEventHandler>,
        layer: LayerSurface,
    ) -> WlrWgpuSurface {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
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

        WlrWgpuSurface {
            adapter,
            device,
            surface,
            queue: Arc::new(queue),
            request_redraw_callback: {
                let surface = layer.wl_surface().clone();
                Arc::new(move || {
                    surface.frame(&qh, surface.clone());
                    surface.commit();
                })
            },
            layer,
        }
    }
}

impl WgpuSurface for WlrWgpuSurface {
    fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }
    fn device(&self) -> &wgpu::Device {
        &self.device
    }
    fn surface(&self) -> &wgpu::Surface<'static> {
        &self.surface
    }
    fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }
    fn request_redraw_callback(&self) -> &Arc<dyn Fn() + Send + Sync> {
        &self.request_redraw_callback
    }
}

pub struct WlrWaylandEventHandler {
    ui_msg_rx: Receiver<UiMessage>,
    compositor: CompositorState,
    layer_shell: LayerShell,
    registry_state: RegistryState,
    output_state: OutputState,

    exit: bool,
    panel_configuration: PanelConfiguration,
    request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
    left_layer: Option<LayerSurface>,
    right_layer: Option<LayerSurface>,
    left_wgpu_holder: Option<(Box<WlrWgpuSurface>, PanelAnchorPosition)>,
    right_wgpu_holder: Option<(Box<WlrWgpuSurface>, PanelAnchorPosition)>,
    app_state: ApplicationState,
    app_state_initialized: bool,
}

impl WlrWaylandEventHandler {
    pub fn set_panel_layout(
        &mut self,
        panel_layout: crate::PanelLayout,
        conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        let (left_wgpu_and_anchor, right_wgpu_and_anchor) = match panel_layout {
            crate::PanelLayout::TwoPanels => (
                Some((
                    self.create_surface(Anchor::TOP | Anchor::LEFT | Anchor::BOTTOM, conn, qh),
                    PanelAnchorPosition::Left,
                )),
                Some((
                    self.create_surface(Anchor::TOP | Anchor::RIGHT | Anchor::BOTTOM, conn, qh),
                    PanelAnchorPosition::Right,
                )),
            ),
            crate::PanelLayout::SingleTop => (
                Some((
                    self.create_surface(Anchor::LEFT | Anchor::TOP | Anchor::RIGHT, conn, qh),
                    PanelAnchorPosition::Top,
                )),
                None,
            ),
            crate::PanelLayout::SingleBottom => (
                Some((
                    self.create_surface(Anchor::LEFT | Anchor::BOTTOM | Anchor::RIGHT, conn, qh),
                    PanelAnchorPosition::Bottom,
                )),
                None,
            ),
        };

        // Already destroy existing WgpuSurfaces
        if let Some(app_state) = self.app_state.windowed_state.as_mut() {
            app_state.left_waveform_window = None;
            app_state.right_waveform_window = None;
        }
        self.left_layer = left_wgpu_and_anchor.as_ref().map(|s| s.0.layer.clone());
        self.right_layer = right_wgpu_and_anchor.as_ref().map(|s| s.0.layer.clone());
        self.left_wgpu_holder = left_wgpu_and_anchor;
        self.right_wgpu_holder = right_wgpu_and_anchor;

        self.update_request_redraw_callback(qh.clone());
    }

    fn create_surface(
        &self,
        anchor: Anchor,
        conn: &Connection,
        qh: &QueueHandle<Self>,
    ) -> Box<WlrWgpuSurface> {
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
        if !anchor.intersects(Anchor::BOTTOM) || !anchor.intersects(Anchor::TOP) {
            layer.set_size(0, self.panel_configuration.width as u32);
        } else {
            layer.set_size(self.panel_configuration.width as u32, 0);
        }
        layer.set_exclusive_zone(
            (self.panel_configuration.width as f32 * self.panel_configuration.exclusive_ratio)
                as i32,
        );

        // In order for the layer surface to be mapped, we need to perform an initial commit with no attached\
        // buffer. For more info, see WaylandSurface::commit
        //
        // The compositor will respond with an initial configure that we can then use to present to the layer
        // surface with the correct options.
        layer.commit();

        let qh = qh.clone();
        Box::new(WlrWgpuSurface::new(conn.clone(), qh, layer))
    }

    pub fn set_panel_width(&mut self, width: u32) {
        let (width, height) = if self.panel_configuration.layout == crate::PanelLayout::SingleTop
            || self.panel_configuration.layout == crate::PanelLayout::SingleBottom
        {
            (0, width)
        } else {
            (width, 0)
        };

        if let Some(layer) = self.left_layer.as_ref() {
            layer.set_size(width, height);
            layer.commit();
        }
        if let Some(layer) = self.right_layer.as_ref() {
            layer.set_size(width, height);
            layer.commit();
        }
        // Apply the ratio relatively to the new width
        self.set_panel_exclusive_ratio(self.panel_configuration.exclusive_ratio);
    }

    pub fn set_panel_exclusive_ratio(&mut self, ratio: f32) {
        if let Some(layer) = self.left_layer.as_ref() {
            layer.set_exclusive_zone((self.panel_configuration.width as f32 * ratio) as i32);
            layer.commit();
        }
        if let Some(layer) = self.right_layer.as_ref() {
            layer.set_exclusive_zone((self.panel_configuration.width as f32 * ratio) as i32);
            layer.commit();
        }
    }

    pub fn set_panel_layer(&mut self, layer: PanelLayer) {
        let wayland_layer = match layer {
            PanelLayer::Overlay => Layer::Overlay,
            PanelLayer::Top => Layer::Top,
            PanelLayer::Bottom => Layer::Bottom,
            PanelLayer::Background => Layer::Background,
        };
        println!("Setting layer to {:?}", wayland_layer);
        if let Some(layer) = self.left_layer.as_ref() {
            layer.set_layer(wayland_layer);
            layer.commit();
        }
        if let Some(layer) = self.right_layer.as_ref() {
            layer.set_layer(wayland_layer);
            layer.commit();
        }
    }

    fn update_request_redraw_callback(&mut self, qh: QueueHandle<Self>) {
        *self.request_redraw_callback.lock().unwrap() = {
            let left_surface = self.left_layer.as_ref().map(|l| l.wl_surface().clone());
            let right_surface = self.right_layer.as_ref().map(|l| l.wl_surface().clone());
            Arc::new(move || {
                if let Some(surface) = &left_surface {
                    surface.frame(&qh, surface.clone());
                    surface.commit();
                }
                if let Some(surface) = &right_surface {
                    surface.frame(&qh, surface.clone());
                    surface.commit();
                }
                // conn.flush().unwrap();
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

pub enum PanelAnchorPosition {
    Top,
    Bottom,
    Left,
    Right,
}

pub trait WlrLayerApplicationHandler {
    fn initialize_app_state(&mut self);
    fn configure_left_wgpu_surface(
        &mut self,
        wgpu_surface: Box<dyn WgpuSurface>,
        anchor_position: PanelAnchorPosition,
    );
    fn configure_right_wgpu_surface(
        &mut self,
        wgpu_surface: Box<dyn WgpuSurface>,
        anchor_position: PanelAnchorPosition,
    );
    fn left_resized(&mut self, width: u32, height: u32);
    fn right_resized(&mut self, width: u32, height: u32);
    fn render(&mut self);
}

pub struct WlrWaylandEventLoop {
    event_queue: EventQueue<WlrWaylandEventHandler>,
    pub state: WlrWaylandEventHandler,
}

impl WlrWaylandEventLoop {
    pub fn new(
        app_state: ApplicationState,
        ui_msg_rx: Receiver<UiMessage>,
        panel_configuration: PanelConfiguration,
        request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
    ) -> WlrWaylandEventLoop {
        // All Wayland apps start by connecting the compositor (server).
        let conn = Connection::connect_to_env().unwrap();

        // Enumerate the list of globals to get the protocols the server implements.
        let (globals, event_queue) = registry_queue_init::<WlrWaylandEventHandler>(&conn).unwrap();
        let qh = event_queue.handle();

        // The compositor (not to be confused with the server which is commonly called the compositor) allows
        // configuring surfaces to be presented.
        let compositor =
            CompositorState::bind(&globals, &qh).expect("wl_compositor is not available");
        // This app uses the wlr layer shell, which may not be available with every compositor.
        let layer_shell = LayerShell::bind(&globals, &qh).expect("layer shell is not available");

        let registry_state = RegistryState::new(&globals);
        let output_state = OutputState::new(&globals, &qh);

        let layout = panel_configuration.layout;
        let mut state = WlrWaylandEventHandler {
            ui_msg_rx,
            compositor,
            layer_shell,
            registry_state,
            output_state,
            exit: false,
            panel_configuration,
            request_redraw_callback,
            left_layer: None,
            right_layer: None,
            left_wgpu_holder: None,
            right_wgpu_holder: None,
            app_state,
            app_state_initialized: false,
        };

        state.set_panel_layout(layout, &conn, &qh);
        state.update_request_redraw_callback(qh);

        WlrWaylandEventLoop { event_queue, state }
    }

    pub fn run_event_loop(&mut self) {
        // We don't draw immediately, the configure will notify us when to first draw.
        loop {
            self.event_queue.blocking_dispatch(&mut self.state).unwrap();

            if self.state.exit {
                println!("exiting example");
                break;
            }
        }
    }
}
