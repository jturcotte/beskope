use pipewire as pw;
use pw::{properties::properties, spa};
use ringbuf::SharedRb;
use ringbuf::storage::Heap;
use ringbuf::traits::Producer;
use ringbuf::wrap::caching::Caching;
use spa::param::format::{MediaSubtype, MediaType};
use spa::param::format_utils;
use spa::pod::Pod;
use std::sync::{Arc, Mutex};

pub fn initialize_audio_capture(
    arc_process_audio: Arc<Mutex<bool>>,
    mut audio_input_ringbuf_prod: Caching<Arc<SharedRb<Heap<f32>>>, true, false>,
    request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
) {
    pw::init();

    let main_loop = pw::main_loop::MainLoop::new(None).unwrap();
    let context = pw::context::Context::new(&main_loop).unwrap();
    let core = context.connect(None).unwrap();
    let format = spa::param::audio::AudioInfoRaw::default();

    let props = properties! {
        // Needed by StreamFlags::AUTOCONNECT
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Monitor",
        *pw::keys::MEDIA_ROLE => "Music",
        *pw::keys::NODE_LATENCY => "256/48000",
        // Capture from the sink monitor ports
        *pw::keys::STREAM_CAPTURE_SINK => "true",
    };

    let stream = pw::stream::Stream::new(&core, "audio-capture", props).unwrap();

    let _listener = stream
        .add_local_listener_with_user_data(format)
        .param_changed(|_, format, id, param| {
            // NULL means to clear the format
            let Some(param) = param else {
                return;
            };
            if id != pw::spa::param::ParamType::Format.as_raw() {
                return;
            }

            let (media_type, media_subtype) = match format_utils::parse_format(param) {
                Ok(v) => v,
                Err(_) => return,
            };

            // only accept raw audio
            if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
                return;
            }

            // call a helper function to parse the format for us.
            format
                .parse(param)
                .expect("Failed to parse param changed to AudioInfoRaw");

            println!(
                "Capturing rate:{} channels:{}",
                format.rate(),
                format.channels()
            );
        })
        .process(move |stream, format| match stream.dequeue_buffer() {
            None => println!("out of buffers"),
            Some(mut buffer) => {
                let datas = buffer.datas_mut();
                if datas.is_empty() {
                    return;
                }

                assert!(datas.len() == 1, "is n_datas always 1 for audio?");
                let data = &mut datas[0];
                let n_channels = format.channels();
                let n_samples = data.chunk().size() as usize / std::mem::size_of::<f32>();

                assert!(
                    n_channels == 2,
                    "Only support monitor port of stereo audio devices"
                );

                if let Some(samples) = data.data() {
                    let process_audio = *arc_process_audio.lock().unwrap();
                    if process_audio {
                        let f32_samples: &[f32] = unsafe {
                            std::slice::from_raw_parts(samples.as_ptr() as *const f32, n_samples)
                        };

                        audio_input_ringbuf_prod.push_slice(f32_samples);

                        request_redraw_callback.lock().unwrap()();
                    }
                }
            }
        })
        .register()
        .unwrap();

    /* Make one parameter with the supported formats. The SPA_PARAM_EnumFormat
     * id means that this is a format enumeration (of 1 value).
     * We leave the channels and rate empty to accept the native graph
     * rate and channels. */
    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
    let obj = pw::spa::pod::Object {
        type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: pw::spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };
    let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(obj),
    )
    .unwrap()
    .0
    .into_inner();

    let mut params = [Pod::from_bytes(&values).unwrap()];

    stream
        .connect(
            spa::utils::Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
            &mut params,
        )
        .unwrap();

    main_loop.run();
}
