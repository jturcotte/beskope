struct YValueOffset {
    offset: u32,
};

@group(0) @binding(0)
var<uniform> y_value_offset: YValueOffset;

@group(0) @binding(1)
var<storage, read> y_values: array<f32>;

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
    let offset = input.should_offset * (y_values[y_index] * 0.5);
    output.position = vec4(input.position, 0.0, 1.0);
    output.position.y += offset;

    return output;
}

@fragment
fn fs_fill_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}

@fragment
fn fs_stroke_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
