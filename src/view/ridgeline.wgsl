// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

struct AudioSync {
    y_value_offsets: array<u32, 32>,
    progress: f32,
    num_instances: f32,
}

@group(0) @binding(0)
var<storage, read> audio_sync: AudioSync;

@group(0) @binding(1)
var<storage, read> y_values: array<f32>;

@group(0) @binding(2)
var<uniform> transform: mat4x4<f32>;

struct WaveformConfig {
    fill_color: vec4<f32>,
    highlight_color: vec4<f32>,
    stroke_color: vec4<f32>,
    apply_highlight_to_front_instance: f32,
};

@group(0) @binding(3)
var<uniform> waveform_config: WaveformConfig;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) waveform_index: u32,
    @location(2) should_offset: f32,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// Helper for fill color logic
fn compute_fill_color(y: f32, should_offset: f32, instance_index: u32) -> vec4<f32> {
    if waveform_config.apply_highlight_to_front_instance == 0.0 && instance_index == 0u {
        // Mix 50% with the highlight color for all values of the front instance to reduce the contrast.
        // with other instances behind it.
        return mix(waveform_config.fill_color, waveform_config.highlight_color, 0.5);
    }

    // Mix fill_color.rgb with the highlight_color based on abs(y)
    let t = clamp(abs(y), 0.0, 1.0);
    return mix(waveform_config.fill_color, waveform_config.highlight_color, t);
}

// Helper for stroke color logic
fn compute_stroke_color() -> vec4<f32> {
    return waveform_config.stroke_color;
}

fn compute_vertex_common(
    input: VertexInput,
    color: vec4<f32>
) -> VertexOutput {
    const FAR_Z = -1.0;

    var output: VertexOutput;
    // Use the instance-specific offset from the offsets array
    let instance_offset = audio_sync.y_value_offsets[input.instance_index];
    // waveform_index points to the last audio sample buffer position at the last vertex.
    // Adding instance_offset from the buffer start points the last vertex to the last sample
    // for this frame.
    let y_index = (input.waveform_index + instance_offset) % arrayLength(&y_values);
    let offset = input.should_offset * y_values[y_index];

    // Calculate position with vertical scaling and offset for each instance
    var position = vec4(input.position, 0.0, 1.0);
    // input.position is the static vertex data, now apply vertical offset based on audio data
    position.y += offset;

    const EPSILON = 1e-5;
    if input.instance_index == 0 {
        // Instance 0 is special in that it's updated every frame and must not vary in z.
        position.z = -EPSILON;
    } else {
        let instance_depth_gap = FAR_Z / audio_sync.num_instances;
        position.z = (-EPSILON * 2.0) + (f32(input.instance_index - 1) + audio_sync.progress) * instance_depth_gap;
    }
    let alpha_factor = 1.0 - f32(input.instance_index) / audio_sync.num_instances;
    output.position = transform * position;
    output.color = vec4(color.rgb * color.a * alpha_factor, color.a * alpha_factor);
    return output;
}

@vertex
fn vs_fill_main(input: VertexInput) -> VertexOutput {
    let instance_offset = audio_sync.y_value_offsets[input.instance_index];
    let y_index = (input.waveform_index + instance_offset) % arrayLength(&y_values);
    let y = y_values[y_index];
    let color = compute_fill_color(y, input.should_offset, input.instance_index);
    return compute_vertex_common(input, color);
}

@vertex
fn vs_stroke_main(input: VertexInput) -> VertexOutput {
    let color = compute_stroke_color();
    return compute_vertex_common(input, color);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
