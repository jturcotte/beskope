use std::collections::HashSet;

use crate::{RenderWindow, ui};

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

    fn set_screen_size(&mut self, screen_width: u32, screen_height: u32);

    /// Get the current configuration with a list of changes triggered by the UI since the last frame.
    /// `config_changes=None` means that the view is new (everything changed).
    fn apply_lazy_config_changes(
        &mut self,
        config: &ui::Configuration,
        config_changes: Option<&HashSet<usize>>,
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

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WaveformConfigUniform {
    fill_color: [f32; 4],
    stroke_color: [f32; 4],
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
