// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

struct YValueOffset {
    offset: u32,
};

@group(0) @binding(0)
var<uniform> y_value_offset: YValueOffset;

@group(0) @binding(1)
var<storage, read> y_values: array<f32>;

@group(0) @binding(2)
var<uniform> transform: mat4x4<f32>;

struct WaveformConfig {
    fill_color: vec4<f32>,
    stroke_color: vec4<f32>,
};

@group(0) @binding(3)
var<uniform> waveform_config: WaveformConfig;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) waveform_index: u32,
    @location(2) should_offset: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let y_index = (input.waveform_index + y_value_offset.offset) % arrayLength(&y_values);
    let offset = input.should_offset * (y_values[y_index] * 0.95);
    var position = vec4(input.position, 0.0, 1.0);
    position.y += offset;

    // Apply the transformation matrix
    output.position = transform * position;

    return output;
}

@fragment
fn fs_fill_main() -> @location(0) vec4<f32> {
    let color = waveform_config.fill_color;
    return vec4<f32>(color.rgb * color.a, color.a);
}

@fragment
fn fs_stroke_main() -> @location(0) vec4<f32> {
    let color = waveform_config.stroke_color;
    return vec4<f32>(color.rgb * color.a, color.a);
}
