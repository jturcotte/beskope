// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use std::collections::HashSet;

use crate::{RenderWindow, WindowMode, ui};

use cgmath::{Matrix4, Rad, SquareMatrix, Vector3};
use num_complex::Complex;
use rustfft::Fft;
use wgpu::{CommandEncoder, TextureView};

mod compressed;
mod ridgeline;

pub use compressed::CompressedWaveformView;
pub use ridgeline::RidgelineWaveformView;

pub trait WaveformView {
    /// Target render window of this view (e.g. right channel view is on the secondary window)
    fn render_window(&self) -> RenderWindow;

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
    );

    fn process_audio(
        self: &mut Self,
        timestamp: u32,
        data: &[f32],
        fft: &dyn Fft<f32>,
        fft_inout_buffer: &mut [Complex<f32>],
        fft_scratch: &mut [Complex<f32>],
    );
}

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

    pub fn get_window_coords(&self, panel_width: f32) -> (f32, f32, f32, f32) {
        // This and the window_x/y assume that the surface is on screen edges.
        // Other panels positioned in-between could make the perspective transform incorrect
        // if they are large and this would require using the actual layer surface position
        // on the screen instead of using the anchor.
        let (window_width, window_height) = match self.layout {
            ui::PanelLayout::SingleBottom => (self.scene_width, panel_width),
            ui::PanelLayout::SingleTop => (self.scene_width, panel_width),
            ui::PanelLayout::TwoPanels => (panel_width, self.scene_height),
        };

        let (window_x, window_y) = match (self.layout, self.is_left_channel) {
            (ui::PanelLayout::SingleTop, _) => (0.0, 0.0),
            (ui::PanelLayout::SingleBottom, _) => (0.0, self.scene_height - window_height),
            (ui::PanelLayout::TwoPanels, true) => (0.0, 0.0),
            (ui::PanelLayout::TwoPanels, false) => (self.scene_width - window_width, 0.0),
        };
        (window_x, window_y, window_width, window_height)
    }

    pub fn get_window_to_scene_transform(&self, panel_width: f32) -> Matrix4<f32> {
        let (_, _, window_width, window_height) = self.get_window_coords(panel_width);

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
