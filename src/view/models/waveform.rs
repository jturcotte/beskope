// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use super::{AudioModel, ValuesRange};
use num_complex::Complex;
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer, Observer, RingBuffer};
use rustfft::{Fft, FftPlanner};
use std::sync::{Arc, Mutex};

pub const FFT_SIZE: usize = 2048;

pub struct WaveformModel {
    stride_len: usize,
    samples_ring: HeapRb<f32>,
    fft: Arc<dyn Fft<f32>>,
    fft_inout: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
}

impl WaveformModel {
    pub fn new(stride_len: usize) -> Self {
        // There should be enough samples to shift half the FFT window to align the lowest frequency bin.
        let samples_ring = HeapRb::<f32>::new(stride_len + FFT_SIZE / 2);
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let scratch_len = fft.get_inplace_scratch_len();
        Self {
            stride_len,
            samples_ring,
            fft,
            fft_inout: vec![Complex::default(); FFT_SIZE],
            fft_scratch: vec![Complex::default(); scratch_len],
        }
    }

    fn phase_alignment_samples(&mut self) -> usize {
        if !self.samples_ring.is_full() {
            return 0;
        }
        // Run an FFT on the accumulated latest FFT length samples as a way to find the peak frequency
        // and align the end of our waveform at the end of the vertex attribute buffer so that the eye
        // isn't totally lost frame over frame.
        let (first, second) = self.samples_ring.as_slices();
        self.fft_inout
            .iter_mut()
            .zip(
                first
                    .iter()
                    .chain(second.iter())
                    // Take the last FFT_SIZE samples
                    .skip(self.samples_ring.occupied_len().saturating_sub(FFT_SIZE)),
            )
            .for_each(|(dst, &y)| *dst = Complex::new(y, 0.));

        self.fft
            .process_with_scratch(&mut self.fft_inout, &mut self.fft_scratch);

        // Skipping k=0 makes sense as it doesn't really capture oscillations, also skip frequencies low enough that
        // aligning to them would prevent the waveform from scrolling enough to be noticeable at 60Hz refresh and 44100Hz sampling rates.
        let k_to_skip: usize = (FFT_SIZE as f64 / (44100.0 / 60.0)).ceil() as usize;

        // Find the peak frequency
        let peak_frequency_index = self
            .fft_inout
            .iter()
            .take(self.fft_inout.len() / 2)
            .enumerate()
            .skip(k_to_skip)
            .max_by(|(_, a), (_, b)| {
                a.norm()
                    .partial_cmp(&b.norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        fn phase_to_samples(phase: f32, k: usize, fft_size: usize) -> usize {
            // When e.g. k=2, the FFT identifies an oscillation that repeats 2 times in the FFT window.
            // To find the phase shift in samples, find where the phase in radians corresponds vs the FFT buffer size.
            ((phase + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * fft_size as f32
                / k as f32) as usize
        }

        // To be able to perform the inverse FFT, each frequency bin also has a phase.
        // Use this phase to align the waveform to the end of the buffer.
        // This here is the sine phase shift in radians.
        let phase_shift = self.fft_inout[peak_frequency_index].arg();

        phase_to_samples(phase_shift, peak_frequency_index, self.fft_inout.len())
    }
}

impl AudioModel for WaveformModel {
    fn stride_len(&self) -> usize {
        self.stride_len
    }

    fn values_range(&self) -> ValuesRange {
        return ValuesRange::NegativeOneToOne;
    }

    fn process_audio<I, W>(
        &mut self,
        samples: I,
        _cqt: Arc<Mutex<Vec<Complex<f64>>>>,
        write_values: W,
    ) where
        I: IntoIterator<Item = f32>,
        W: FnOnce(&[f32]),
    {
        self.samples_ring.push_iter_overwrite(samples.into_iter());

        let phase_samples = self.phase_alignment_samples();
        let (a, b) = self.samples_ring.as_slices();
        let temp: Vec<f32> = a
            .iter()
            .chain(b.iter())
            // Skip the beginning of the ring buffer to reach the position where we can take a full stride
            // while keeping the end of the stride aligned according to the phase calculation.
            .skip(
                self.samples_ring
                    .occupied_len()
                    .saturating_sub(self.stride_len + phase_samples),
            )
            .take(self.stride_len)
            .cloned()
            .collect();

        write_values(&temp);
    }
}
