use crate::{RenderWindow, ui};

use num_complex::Complex;
use rustfft::Fft;
use wgpu::{CommandEncoder, TextureView};

mod compressed;
mod ridgeline;

pub use compressed::CompressedWaveformView;
pub use ridgeline::RidgelineWaveformView;

pub trait WaveformView {
    fn render_window(&self) -> RenderWindow;
    fn update_transform(&mut self, panel_width: u32, screen_width: u32, screen_height: u32);
    fn set_waveform_config(&mut self, config: ui::WaveformConfig);
    // FIXME: Remove? How will the view-specific config be abstracted?
    fn set_time_curve_control_points(&mut self, _points: Vec<ui::ControlPoint>);
    fn update_config(&mut self);
    fn process_audio(
        self: &mut Self,
        timestamp: u32,
        data: &[f32],
        fft: &dyn Fft<f32>,
        fft_inout_buffer: &mut [Complex<f32>],
        fft_scratch: &mut [Complex<f32>],
    );
    fn render(
        &self,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        depth_texture_view: &TextureView,
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
