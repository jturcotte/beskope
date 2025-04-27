struct AudioSync {
    y_value_offsets: array<u32, 32>,
    progress: f32,
}

@group(0) @binding(0)
var<storage, read> audio_sync: AudioSync;

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
    @builtin(instance_index) instance_index: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) alpha_factor: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    const DEFAULT_AMPLITUDE = 0.15;
    const FAR_Z = -1.0;
    const NUM_INSTANCES = 32.0;

    // Use the instance-specific offset from the offsets array
    let instance_offset = audio_sync.y_value_offsets[31 - input.instance_index];
    // waveform_index points to the last audio sample buffer position at the last vertex.
    // Adding instance_offset from the buffer start points the last vertex to the last sample
    // for this frame.
    let y_index = (input.waveform_index + instance_offset) % arrayLength(&y_values);
    let offset = input.should_offset * (y_values[y_index] * DEFAULT_AMPLITUDE);

    // Calculate position with vertical scaling and offset for each instance
    var position = vec4(input.position, 0.0, 1.0);
    // transform will scale the waveform and put its center at the screen edge,
    // offset it first by the default amplitude so that the minimum amplitude ends up at the edge.
    position.y += DEFAULT_AMPLITUDE + offset;

    const INSTANCE_DEPTH_GAP = FAR_Z / NUM_INSTANCES;
    const EPSILON = 1e-5;
    if input.instance_index == 0 {
        // Instance 0 is special in that it's updated every frame and must not vary in z.
        position.z = -EPSILON;
    } else {
        position.z = (f32(input.instance_index - 1) + audio_sync.progress) * INSTANCE_DEPTH_GAP;
    }
    output.alpha_factor = 1.0 - f32(input.instance_index) / 32.0;

    // Apply the transformation matrix
    output.position = transform * position;

    return output;
}

@fragment
fn fs_fill_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = waveform_config.fill_color;
    return vec4<f32>(color.rgb * color.a * input.alpha_factor, color.a * input.alpha_factor);
}

@fragment
fn fs_stroke_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = waveform_config.stroke_color;
    return vec4<f32>(color.rgb * color.a * input.alpha_factor, color.a * input.alpha_factor);
}
