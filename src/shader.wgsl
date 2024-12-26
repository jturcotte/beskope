struct YValueOffset {
    offset: u32,
};

@group(0) @binding(0)
var<uniform> y_value_offset: YValueOffset;

@group(0) @binding(1)
var<storage, read> y_values: array<f32>;

struct VertexInput {
    @location(0) position: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main( @builtin(vertex_index) my_index: u32, input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let y_index = (my_index + y_value_offset.offset) % arrayLength(&y_values);
    output.position = vec4<f32>(input.position.x, y_values[y_index] * 0.25 + 0.75, 0.0, 1.0);
    output.color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return output;
}

@vertex
fn vs_straight(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(position.x, position.y, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}