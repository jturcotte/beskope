// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use num_complex::Complex;
use pipewire as pw;
use pw::{properties::properties, spa};
use qdft::QDFT32;
use ringbuf::SharedRb;
use ringbuf::storage::Heap;
use ringbuf::traits::{Consumer, Observer, Producer};
use ringbuf::wrap::caching::Caching;
use spa::param::format::{MediaSubtype, MediaType};
use spa::param::format_utils;
use spa::pod::Pod;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Condvar, Mutex};

use crate::ChannelTransformMode;

pub fn initialize_audio_capture(
    mut audio_input_ringbuf_prod: Caching<Arc<SharedRb<Heap<f32>>>, true, false>,
    animation_stopped: Arc<AtomicBool>,
    request_redraw_callback: Arc<Mutex<Arc<dyn Fn() + Send + Sync>>>,
    audio_transform_control: Arc<(Mutex<(ChannelTransformMode, ChannelTransformMode)>, Condvar)>,
    sample_rate_tx: SyncSender<u32>,
) {
    pw::init();

    let main_loop = pw::main_loop::MainLoop::new(None).unwrap();
    let context = pw::context::Context::new(&main_loop).unwrap();
    let core = context
        .connect(None)
        .expect("This application requires the PipeWire daemon to be running");
    let format = spa::param::audio::AudioInfoRaw::default();

    let props = properties! {
        // Needed by StreamFlags::AUTOCONNECT
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Monitor",
        *pw::keys::MEDIA_ROLE => "Music",
        *pw::keys::NODE_LATENCY => "128/48000",
        // Capture from the sink monitor ports
        *pw::keys::STREAM_CAPTURE_SINK => "true",
    };

    let stream = pw::stream::Stream::new(&core, "audio-capture", props).unwrap();
    let mut last_non_zero_sample_age: usize = 0;

    let _listener = stream
        .add_local_listener_with_user_data(format)
        .param_changed({
            let sample_rate_tx = sample_rate_tx.clone();
            let mut sample_rate_sent = false;
            move |_, format, id, param| {
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

                if !sample_rate_sent {
                    let _ = sample_rate_tx.send(format.rate());
                    sample_rate_sent = true;
                }
            }
        })
        .process(move |stream, format| match stream.dequeue_buffer() {
            None => eprintln!("out of buffers"),
            Some(mut buffer) => {
                let datas = buffer.datas_mut();
                if datas.is_empty() {
                    return;
                }

                assert!(datas.len() == 1, "is n_datas always 1 for audio?");
                let data = &mut datas[0];
                let n_channels = format.channels();
                let n_samples = data.chunk().size() as usize / std::mem::size_of::<f32>();

                if let Some(samples) = data.data() {
                    let f32_samples: &[f32] = unsafe {
                        std::slice::from_raw_parts(samples.as_ptr() as *const f32, n_samples)
                    };

                    let mut should_forward_data = true;

                    let has_only_zeroes = f32_samples.iter().all(|&sample| sample == 0.0);

                    // Check if we need to restart or stop the rendering
                    if animation_stopped.load(Ordering::Relaxed) {
                        if has_only_zeroes {
                            should_forward_data = false;
                        } else {
                            animation_stopped.store(false, Ordering::Relaxed);
                            request_redraw_callback.lock().unwrap()();
                        }
                    } else if has_only_zeroes {
                        last_non_zero_sample_age += f32_samples.len();
                        if last_non_zero_sample_age > format.rate() as usize * 2 * 5 {
                            // Stop requesting new frames and let the audio thread know if they
                            // should wake us up once non-zero samples are available.
                            animation_stopped.store(true, Ordering::Relaxed);
                        }
                    } else {
                        last_non_zero_sample_age = 0;
                    }

                    if should_forward_data {
                        if n_channels == 1 {
                            // Duplicate the channel sample to make it stereo
                            let interleaved_samples =
                                f32_samples.iter().flat_map(|&sample| [sample, sample]);
                            audio_input_ringbuf_prod.push_iter(interleaved_samples);
                        } else {
                            assert!(
                                n_channels == 2,
                                "Only support monitor port of mono or stereo audio devices"
                            );
                            // Just push the samples as is
                            audio_input_ringbuf_prod.push_slice(f32_samples);
                        }
                    }

                    // Wake transform thread
                    let (_lock, cvar) = &*audio_transform_control;
                    cvar.notify_one();
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

pub fn init_audio_transform(
    sample_rate: u32,
    bandwidth: (f64, f64),
    resolution: f64,
    guessed_cqt_size: usize,
    mut audio_input_ringbuf_cons: Caching<Arc<SharedRb<Heap<f32>>>, false, true>,
    mut transform_thread_ringbuf_prod: Caching<Arc<SharedRb<Heap<f32>>>, true, false>,
    cqt_buffer_left: Arc<Mutex<Vec<Complex<f64>>>>,
    cqt_buffer_right: Arc<Mutex<Vec<Complex<f64>>>>,
    audio_transform_control: Arc<(Mutex<(ChannelTransformMode, ChannelTransformMode)>, Condvar)>,
) {
    let quality = -1.0; // Something about making bass frequency response time better vs their resolution.
    let latency = 1.0;
    let window = Some((0.5, -0.5)); // Hann-like
    let mut cqt_left = QDFT32::new(
        sample_rate as f64,
        bandwidth,
        resolution,
        quality,
        latency,
        window,
    );
    let mut cqt_right = QDFT32::new(
        sample_rate as f64,
        bandwidth,
        resolution,
        quality,
        latency,
        window,
    );
    assert_eq!(guessed_cqt_size, cqt_left.size());

    loop {
        // Wait for enough samples to be available
        let (lock, cvar) = &*audio_transform_control;
        let ((left_mode, right_mode), data) = {
            let guard = cvar
                .wait_while(lock.lock().unwrap(), |_| {
                    audio_input_ringbuf_cons.occupied_len() == 0
                })
                .unwrap();

            (
                *guard,
                audio_input_ringbuf_cons.pop_iter().collect::<Vec<f32>>(),
            )
        };

        if left_mode == ChannelTransformMode::Raw || right_mode == ChannelTransformMode::Raw {
            // Forward the data for both channels even if only one is needed
            transform_thread_ringbuf_prod.push_slice(&data);
        }

        if left_mode == ChannelTransformMode::CQT {
            let mut lock = cqt_buffer_left.lock().unwrap();
            let buffer = lock.as_mut();
            for i in data.iter().skip(0).step_by(2) {
                cqt_left.qdft_scalar(&i, buffer);
            }
        }

        if right_mode == ChannelTransformMode::CQT {
            let mut lock = cqt_buffer_right.lock().unwrap();
            let buffer = lock.as_mut();
            for i in data.iter().skip(1).step_by(2) {
                cqt_right.qdft_scalar(&i, buffer);
            }
        }
    }
}
