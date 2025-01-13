use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
};
use smithay_client_toolkit::compositor::Region;
use std::ptr::NonNull;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
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

use crate::{UiMessage, WgpuSurface};

impl CompositorHandler for WlrLayersState {
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
                    closure(self.loop_state.app_state.as_mut().unwrap())
                }
                UiMessage::SetPanelLayout(layout) => {
                    self.set_panel_layout(layout, conn, qh);
                }
                UiMessage::SetPanelLayer(_) => {}
            }
        }
        self.loop_state.render();
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

impl OutputHandler for WlrLayersState {
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

impl LayerShellHandler for WlrLayersState {
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

        if !self.loop_state_initialized {
            self.loop_state_initialized = true;
            self.loop_state.initialize_app_state();
        }
        if let Some((wgpu_surface, anchor_position)) = self.left_wgpu_holder.take() {
            self.loop_state
                .configure_left_wgpu_surface(wgpu_surface, anchor_position);
        }
        if let Some((wgpu_surface, anchor_position)) = self.right_wgpu_holder.take() {
            self.loop_state
                .configure_right_wgpu_surface(wgpu_surface, anchor_position);
        }

        if Some(layer) == self.left_layer.as_ref() {
            self.loop_state.left_resized(new_width, new_height);
        }
        if Some(layer) == self.right_layer.as_ref() {
            self.loop_state.right_resized(new_width, new_height);
        }

        if Some(layer) == self.right_layer.as_ref() || self.right_layer.is_none() {
            // Render once to let wgpu finalize the surface initialization.
            // FIXME: I get a "layer_surface has never been configured" error from wlroots
            // if I do this after the first layer, but putting this here seems to work.
            self.loop_state.render();
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
        qh: QueueHandle<WlrLayersState>,
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
                let layer = layer.clone();
                Arc::new(move || {
                    let surface = layer.wl_surface();
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

pub struct WlrLayersState {
    ui_msg_rx: Receiver<UiMessage>,
    compositor: CompositorState,
    layer_shell: LayerShell,
    registry_state: RegistryState,
    output_state: OutputState,

    exit: bool,
    left_layer: Option<LayerSurface>,
    right_layer: Option<LayerSurface>,
    left_wgpu_holder: Option<(Box<WlrWgpuSurface>, PanelAnchorPosition)>,
    right_wgpu_holder: Option<(Box<WlrWgpuSurface>, PanelAnchorPosition)>,
    loop_state: crate::LoopState,
    loop_state_initialized: bool,
}

impl WlrLayersState {
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
        if let Some(app_state) = self.loop_state.app_state.as_mut() {
            app_state.left_waveform_window = None;
            app_state.right_waveform_window = None;
        }
        self.left_layer = left_wgpu_and_anchor.as_ref().map(|s| s.0.layer.clone());
        self.right_layer = right_wgpu_and_anchor.as_ref().map(|s| s.0.layer.clone());
        self.left_wgpu_holder = left_wgpu_and_anchor;
        self.right_wgpu_holder = right_wgpu_and_anchor;
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
            layer.set_size(0, 80);
        } else {
            layer.set_size(80, 0);
        }
        layer.set_exclusive_zone(80 / 2);

        // In order for the layer surface to be mapped, we need to perform an initial commit with no attached\
        // buffer. For more info, see WaylandSurface::commit
        //
        // The compositor will respond with an initial configure that we can then use to present to the layer
        // surface with the correct options.
        layer.commit();

        let qh = qh.clone();
        Box::new(WlrWgpuSurface::new(conn.clone(), qh, layer))
    }
}

delegate_compositor!(WlrLayersState);
delegate_output!(WlrLayersState);

delegate_layer!(WlrLayersState);

delegate_registry!(WlrLayersState);

impl ProvidesRegistryState for WlrLayersState {
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

pub struct WlrLayersEventQueue {
    conn: Connection,
    event_queue: EventQueue<WlrLayersState>,
    pub state: WlrLayersState,
}

impl WlrLayersEventQueue {
    pub fn new(app_state: crate::LoopState, ui_msg_rx: Receiver<UiMessage>) -> WlrLayersEventQueue {
        // All Wayland apps start by connecting the compositor (server).
        let conn = Connection::connect_to_env().unwrap();

        // Enumerate the list of globals to get the protocols the server implements.
        let (globals, event_queue) = registry_queue_init::<WlrLayersState>(&conn).unwrap();
        let qh = event_queue.handle();

        // The compositor (not to be confused with the server which is commonly called the compositor) allows
        // configuring surfaces to be presented.
        let compositor =
            CompositorState::bind(&globals, &qh).expect("wl_compositor is not available");
        // This app uses the wlr layer shell, which may not be available with every compositor.
        let layer_shell = LayerShell::bind(&globals, &qh).expect("layer shell is not available");

        let registry_state = RegistryState::new(&globals);
        let output_state = OutputState::new(&globals, &qh);

        let state = WlrLayersState {
            ui_msg_rx,
            compositor,
            layer_shell,
            registry_state,
            output_state,
            exit: false,
            left_layer: None,
            right_layer: None,
            left_wgpu_holder: None,
            right_wgpu_holder: None,
            loop_state: app_state,
            loop_state_initialized: false,
        };

        WlrLayersEventQueue {
            conn,
            event_queue,
            state,
        }
    }

    pub fn set_panel_layout(&mut self, panel_layout: crate::PanelLayout) {
        self.state
            .set_panel_layout(panel_layout, &self.conn, &self.event_queue.handle());
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
