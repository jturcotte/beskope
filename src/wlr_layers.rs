use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
};
use std::ptr::NonNull;
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

use crate::WgpuSurface;

impl CompositorHandler for WlrLayersState {
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
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _time: u32,
    ) {
        self.app_state.render();
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
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
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }
}

impl LayerShellHandler for WlrLayersState {
    fn closed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _layer: &LayerSurface) {
        // FIXME: handle instead
        self.exit = true;
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let (new_width, new_height) = configure.new_size;
        println!("Configure: {new_width}x{new_height}");

        if let Some(wgpu) = self.wgpu_holder.take() {
            self.app_state.initialize(wgpu);
        }

        self.app_state.resized(new_width, new_height);

        // Render once to let wgpu finalize the surface initialization.
        self.app_state.render();
    }
}

struct WlrLayersState {
    registry_state: RegistryState,
    output_state: OutputState,

    _layer: LayerSurface,
    exit: bool,
    wgpu_holder: Option<WgpuSurface>,
    app_state: Box<dyn WlrLayerApplicationHandler>,
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

pub trait WlrLayerApplicationHandler {
    fn initialize(&mut self, wgpu_surface: WgpuSurface);
    fn resized(&mut self, width: u32, height: u32);
    fn render(&mut self);
}

pub struct WlrLayersEventQueue {
    event_queue: EventQueue<WlrLayersState>,
    state: WlrLayersState,
}

// FIXME: I don't need this separate stuff
impl WlrLayersEventQueue {
    pub fn new(app_state: impl WlrLayerApplicationHandler + 'static) -> WlrLayersEventQueue {
        // All Wayland apps start by connecting the compositor (server).
        let conn = Connection::connect_to_env().unwrap();

        // Enumerate the list of globals to get the protocols the server implements.
        let (globals, event_queue) = registry_queue_init(&conn).unwrap();
        let qh = event_queue.handle();

        // The compositor (not to be confused with the server which is commonly called the compositor) allows
        // configuring surfaces to be presented.
        let compositor =
            CompositorState::bind(&globals, &qh).expect("wl_compositor is not available");
        // This app uses the wlr layer shell, which may not be available with every compositor.
        let layer_shell = LayerShell::bind(&globals, &qh).expect("layer shell is not available");

        // A layer surface is created from a surface.
        let surface = compositor.create_surface(&qh);

        // And then we create the layer shell.
        let layer =
            layer_shell.create_layer_surface(&qh, surface, Layer::Top, Some("simple_layer"), None);
        // Configure the layer surface, providing things like the anchor on screen, desired size and the keyboard
        // interactivity
        layer.set_anchor(Anchor::TOP | Anchor::LEFT | Anchor::BOTTOM);
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);
        layer.set_size(64, 0);
        // FIXME: Remove once transparency and mouse passthrough work
        layer.set_exclusive_zone(64);

        // In order for the layer surface to be mapped, we need to perform an initial commit with no attached\
        // buffer. For more info, see WaylandSurface::commit
        //
        // The compositor will respond with an initial configure that we can then use to present to the layer
        // surface with the correct options.
        layer.commit();

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

        let layer_clone = layer.clone();
        let qh_clone = qh.clone();
        let wgpu = WgpuSurface {
            adapter,
            device,
            surface,
            queue: Arc::new(queue),
            request_redraw: Arc::new(move || {
                let surface = layer_clone.wl_surface();
                surface.frame(&qh_clone, surface.clone());
                surface.commit();
            }),
        };

        let state = WlrLayersState {
            registry_state: RegistryState::new(&globals),
            output_state: OutputState::new(&globals, &qh),
            _layer: layer,
            exit: false,
            wgpu_holder: Some(wgpu),
            app_state: Box::new(app_state),
        };

        WlrLayersEventQueue { event_queue, state }
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
