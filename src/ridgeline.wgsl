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
    @location(0) color: vec4<f32>,
};

// Helper for fill color logic
fn compute_fill_color(y: f32, should_offset: f32, instance_index: u32) -> vec4<f32> {
    if instance_index == 0u {
        // Mix 50% with the highlight color for the front instance to reduce the contrast.
        return vec4(mix(waveform_config.fill_color.rgb, vec3<f32>(1.0, 1.0, 1.0), 0.5),
                    waveform_config.fill_color.a);
    }

    // Mix fill_color.rgb with white based on abs(y)
    // FIXME: Make this configurable
    let t = clamp(abs(y), 0.0, 1.0);
    return vec4(mix(waveform_config.fill_color.rgb, vec3<f32>(1.0, 1.0, 1.0), t),
                waveform_config.fill_color.a);
}

// Helper for stroke color logic
fn compute_stroke_color() -> vec4<f32> {
    return waveform_config.stroke_color;
}

fn compute_vertex_common(
    input: VertexInput,
    color: vec4<f32>
) -> VertexOutput {
    const DEFAULT_AMPLITUDE = 0.15;
    const FAR_Z = -1.0;
    const NUM_INSTANCES = 32.0;

    var output: VertexOutput;
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
        position.z = EPSILON + (f32(input.instance_index - 1) + audio_sync.progress) * INSTANCE_DEPTH_GAP;
    }
    let alpha_factor = 1.0 - f32(input.instance_index) / 32.0;
    output.position = transform * position;
    output.color = vec4(color.rgb * color.a * alpha_factor, color.a * alpha_factor);
    return output;
}

@vertex
fn vs_fill_main(input: VertexInput) -> VertexOutput {
    let instance_offset = audio_sync.y_value_offsets[31 - input.instance_index];
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
