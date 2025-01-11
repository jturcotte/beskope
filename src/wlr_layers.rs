use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
};
use smithay_client_toolkit::compositor::Region;
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

        if let (Some(left_wgpu), Some(right_wgpu)) =
            (self.left_wgpu_holder.take(), self.right_wgpu_holder.take())
        {
            self.app_state.initialize(left_wgpu, right_wgpu);
        }

        if layer == &self.left_layer {
            self.app_state.left_resized(new_width, new_height);
        }
        if layer == &self.right_layer {
            self.app_state.right_resized(new_width, new_height);

            // Render once to let wgpu finalize the surface initialization.
            // FIXME: I get a "layer_surface has never been configured" error from wlroot
            // if I do this after the first layer, but putting this here should work for now.
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

struct WlrLayersState {
    registry_state: RegistryState,
    output_state: OutputState,

    exit: bool,
    left_layer: LayerSurface,
    right_layer: LayerSurface,
    left_wgpu_holder: Option<Box<WlrWgpuSurface>>,
    right_wgpu_holder: Option<Box<WlrWgpuSurface>>,
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
    fn initialize(
        &mut self,
        left_wgpu_surface: Box<dyn WgpuSurface>,
        right_wgpu_surface: Box<dyn WgpuSurface>,
    );
    fn left_resized(&mut self, width: u32, height: u32);
    fn right_resized(&mut self, width: u32, height: u32);
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

        let create_surface = |conn, anchor| {
            // A layer surface is created from a surface.
            let surface = compositor.create_surface(&qh);
            // Let mouse events pass through the surface
            surface.set_input_region(Some(Region::new(&compositor).unwrap().wl_region()));

            // And then we create the layer shell.
            let layer =
                layer_shell.create_layer_surface(&qh, surface, Layer::Top, None::<String>, None);
            // Configure the layer surface, providing things like the anchor on screen, desired size and the keyboard
            // interactivity
            layer.set_anchor(anchor);
            layer.set_keyboard_interactivity(KeyboardInteractivity::None);
            layer.set_size(80, 0);
            layer.set_exclusive_zone(80 / 2);

            // In order for the layer surface to be mapped, we need to perform an initial commit with no attached\
            // buffer. For more info, see WaylandSurface::commit
            //
            // The compositor will respond with an initial configure that we can then use to present to the layer
            // surface with the correct options.
            layer.commit();

            let qh = qh.clone();
            Box::new(WlrWgpuSurface::new(conn, qh, layer))
        };

        let left_wgpu = create_surface(conn.clone(), Anchor::TOP | Anchor::LEFT | Anchor::BOTTOM);
        let right_wgpu = create_surface(conn, Anchor::TOP | Anchor::RIGHT | Anchor::BOTTOM);

        let state = WlrLayersState {
            registry_state,
            output_state,
            exit: false,
            left_layer: left_wgpu.layer.clone(),
            right_layer: right_wgpu.layer.clone(),
            left_wgpu_holder: Some(left_wgpu),
            right_wgpu_holder: Some(right_wgpu),
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
