// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use std::{
    collections::HashSet,
    rc::Rc,
    time::{Duration, Instant},
};

use crate::{surface::WgpuSurface, ui};

use cgmath::{Matrix4, Rad, SquareMatrix, Vector3};
use num_complex::Complex;
use rustfft::Fft;
use wgpu::{CommandEncoder, TextureView};

mod compressed;
mod ridgeline;

pub use compressed::CompressedView;
pub use ridgeline::RidgelineView;

pub const VERTEX_BUFFER_SIZE: usize = 44100 * 3;
pub const FFT_SIZE: usize = 2048;

/// Trait of a visualization view of an audio channel
pub trait View {
    /// Target render window of this view (e.g. right channel view is on the secondary window)
    fn render_surface(&self) -> RenderSurface;

    /// Get the current configuration with a list of changes triggered by the UI since the last frame.
    /// `config_changes=None` means that the view is new (everything changed).
    fn apply_lazy_config_changes(
        &mut self,
        config: &ui::Configuration,
        config_changes: Option<&HashSet<usize>>,
        view_transform_change: Option<&ViewTransform>,
    );

    fn render(
        &self,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        depth_texture_view: &TextureView,
        clear_color: Option<wgpu::Color>,
    );

    fn process_audio(
        &mut self,
        timestamp: u32,
        data: &[f32],
        fft: &dyn Fft<f32>,
        fft_inout_buffer: &mut [Complex<f32>],
        fft_scratch: &mut [Complex<f32>],
    );
}

/// State and logic to transform the view onto the surface depending on the configured layout.
#[derive(Debug, Clone, Copy)]
pub struct ViewTransform {
    window_mode: WindowMode,
    layout: ui::PanelLayout,
    is_left_channel: bool,
    scene_width: f32,
    scene_height: f32,
    transform_matrix: Matrix4<f32>,
    is_vertical: bool,
}

impl ViewTransform {
    pub fn new(
        screen_width: f32,
        screen_height: f32,
        window_mode: WindowMode,
        layout: ui::PanelLayout,
        channels: ui::RenderChannels,
        is_left_channel: bool,
    ) -> Self {
        let is_vertical = match layout {
            ui::PanelLayout::TwoPanels => true,
            ui::PanelLayout::SingleTop => false,
            ui::PanelLayout::SingleBottom => false,
        };

        // Identity transform is a horizontal waveform scrolling from right to left.
        let rotation = Matrix4::from_angle_z(Rad(-std::f32::consts::FRAC_PI_2));
        let mirror_h = Matrix4::from_nonuniform_scale(-1.0, 1.0, 1.0);
        let mirror_v = Matrix4::from_nonuniform_scale(1.0, -1.0, 1.0);
        let half = Matrix4::from_nonuniform_scale(0.5, 1.0, 1.0);
        let translate_half_left = Matrix4::from_translation(Vector3::new(-1.0, 0.0, 0.0));
        let translate_half_right = Matrix4::from_translation(Vector3::new(1.0, 0.0, 0.0));

        let transform_matrix = match (layout, channels, is_left_channel) {
            (ui::PanelLayout::SingleTop, ui::RenderChannels::Single, _) => mirror_v,
            (ui::PanelLayout::SingleTop, ui::RenderChannels::Both, true) => {
                mirror_v * half * translate_half_left
            }
            (ui::PanelLayout::SingleTop, ui::RenderChannels::Both, false) => {
                mirror_v * half * translate_half_right * mirror_h
            }
            (ui::PanelLayout::SingleBottom, ui::RenderChannels::Single, _) => Matrix4::identity(),
            (ui::PanelLayout::SingleBottom, ui::RenderChannels::Both, true) => {
                half * translate_half_left
            }
            (ui::PanelLayout::SingleBottom, ui::RenderChannels::Both, false) => {
                half * translate_half_right * mirror_h
            }
            (ui::PanelLayout::TwoPanels, _, true) => rotation * mirror_h,
            (ui::PanelLayout::TwoPanels, _, false) => rotation * mirror_v * mirror_h,
        };

        ViewTransform {
            window_mode,
            layout,
            is_left_channel,
            scene_width: screen_width,
            scene_height: screen_height,
            transform_matrix,
            is_vertical,
        }
    }

    pub fn get_window_coords(&self, panel_width_ratio: f32) -> (f32, f32, f32, f32) {
        // This and the window_x/y assume that the surface is on screen edges.
        // Other panels positioned in-between could make the perspective transform incorrect
        // if they are large and this would require using the actual layer surface position
        // on the screen instead of using the anchor.
        let (window_width, window_height) = match self.layout {
            ui::PanelLayout::SingleBottom => {
                (self.scene_width, self.scene_height * panel_width_ratio)
            }
            ui::PanelLayout::SingleTop => (self.scene_width, self.scene_height * panel_width_ratio),
            ui::PanelLayout::TwoPanels => (self.scene_width * panel_width_ratio, self.scene_height),
        };

        let (window_x, window_y) = match (self.layout, self.is_left_channel) {
            (ui::PanelLayout::SingleTop, _) => (0.0, 0.0),
            (ui::PanelLayout::SingleBottom, _) => (0.0, self.scene_height - window_height),
            (ui::PanelLayout::TwoPanels, true) => (0.0, 0.0),
            (ui::PanelLayout::TwoPanels, false) => (self.scene_width - window_width, 0.0),
        };
        (window_x, window_y, window_width, window_height)
    }

    pub fn get_window_to_scene_transform(&self, panel_width_ratio: f32) -> Matrix4<f32> {
        let (_, _, window_width, window_height) = self.get_window_coords(panel_width_ratio);

        if self.window_mode == WindowMode::WindowPerScene {
            // The views are first transformed to fit either on the side of the screen inside their own panel,
            // or on the side of the window.
            // This transform the per-panel views so that they appear inside the window the same as if they'd render
            // onto the desktop.
            match (self.layout, self.is_left_channel) {
                (ui::PanelLayout::TwoPanels, true) => {
                    // Scaling is done, move the origin back to the left edge of the window
                    Matrix4::from_translation(Vector3::new(-1.0, 0.0, 0.0))
                    // Scale with 0.0 as the origin
                        * Matrix4::from_nonuniform_scale(
                            window_width / self.scene_width,
                            window_height / self.scene_height,
                            1.0,
                        )
                        // The geometry are 2.0 wide, move right so that the left edge is at 0.0
                        * Matrix4::from_translation(Vector3::new(1.0, 0.0, 0.0))
                }
                (ui::PanelLayout::TwoPanels, false) => {
                    Matrix4::from_translation(Vector3::new(1.0, 0.0, 0.0))
                        * Matrix4::from_nonuniform_scale(
                            window_width / self.scene_width,
                            window_height / self.scene_height,
                            1.0,
                        )
                        * Matrix4::from_translation(Vector3::new(-1.0, 0.0, 0.0))
                }
                (ui::PanelLayout::SingleTop, _) => {
                    Matrix4::from_translation(Vector3::new(0.0, 1.0, 0.0))
                        * Matrix4::from_nonuniform_scale(
                            window_width / self.scene_width,
                            window_height / self.scene_height,
                            1.0,
                        )
                        * Matrix4::from_translation(Vector3::new(0.0, -1.0, 0.0))
                }
                (ui::PanelLayout::SingleBottom, _) => {
                    Matrix4::from_translation(Vector3::new(0.0, -1.0, 0.0))
                        * Matrix4::from_nonuniform_scale(
                            window_width / self.scene_width,
                            window_height / self.scene_height,
                            1.0,
                        )
                        * Matrix4::from_translation(Vector3::new(0.0, 1.0, 0.0))
                }
            }
        } else {
            Matrix4::identity()
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    waveform_index: u32,
    should_offset: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct YValue {
    y: f32,
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// Contains common resources to all views when views of multiple channels are rendered
/// onto the same surface.
pub struct ViewSurface {
    surface_id: u32,
    pub wgpu: Rc<dyn WgpuSurface>,
    depth_texture: Option<wgpu::Texture>,
    render_surface: RenderSurface,
    depth_texture_size: (u32, u32),
    last_fps_dump_time: Instant,
    frame_count: u32,
    fps_callback: Box<dyn Fn(u32)>,
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum WindowMode {
    WindowPerPanel,
    WindowPerScene,
}

/// When rendering on Wayland with one panel per channel, Secondary is the right channel.
/// In single surface cases, only Primary is used.
#[derive(PartialEq, Clone, Copy, Debug)]
pub enum RenderSurface {
    Primary,
    Secondary,
}

impl ViewSurface {
    pub fn new(
        wgpu: &Rc<dyn WgpuSurface>,
        render_surface: RenderSurface,
        fps_callback: Box<dyn Fn(u32)>,
    ) -> ViewSurface {
        ViewSurface {
            surface_id: wgpu.surface_id(),
            wgpu: wgpu.clone(),
            depth_texture: None,
            render_surface,
            // Force reconfiguration of the depth texture on the first render
            depth_texture_size: (u32::MAX, u32::MAX),
            last_fps_dump_time: Instant::now(),
            frame_count: 0,
            fps_callback,
        }
    }

    pub fn surface_id(&self) -> u32 {
        self.surface_id
    }

    pub fn render_with_clear_color(
        &mut self,
        wgpu: &Rc<dyn WgpuSurface>,
        surface_texture: &wgpu::Texture,
        left_view: &mut Option<Box<dyn View>>,
        right_view: &mut Option<Box<dyn View>>,
        clear_color: wgpu::Color,
    ) {
        if self.depth_texture_size != (surface_texture.width(), surface_texture.height()) {
            // Create the depth texture
            self.depth_texture = Some(wgpu.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width: surface_texture.width(),
                    height: surface_texture.height(),
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
            self.depth_texture_size = (surface_texture.width(), surface_texture.height());
        }

        let texture_view = surface_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture_view = self
            .depth_texture
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = wgpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Only clear the first view that renders to this surface
        let mut clear_flag = Some(clear_color);

        if let Some(view) = left_view {
            if view.render_surface() == self.render_surface {
                view.render(
                    &mut encoder,
                    &texture_view,
                    &depth_texture_view,
                    clear_flag.take(),
                );
            }
        }
        if let Some(view) = right_view {
            if view.render_surface() == self.render_surface {
                view.render(
                    &mut encoder,
                    &texture_view,
                    &depth_texture_view,
                    clear_flag.take(),
                );
            }
        }
        wgpu.queue().submit(Some(encoder.finish()));

        let now = Instant::now();
        self.frame_count += 1;
        if now.duration_since(self.last_fps_dump_time) >= Duration::from_secs(1) {
            (self.fps_callback)(self.frame_count);
            self.frame_count = 0;
            self.last_fps_dump_time = now;
        }
    }
}
