// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ValuesRange {
    NegativeOneToOne,
    ZeroToOne,
}

pub trait AudioModel {
    fn stride_len(&self) -> usize;
    fn values_range(&self) -> ValuesRange;
    fn process_audio<I, W, T>(
        &mut self,
        samples: I,
        cqt: Arc<Mutex<Vec<Complex<f64>>>>,
        write_values: W,
        update_threshold: T,
    ) where
        I: IntoIterator<Item = f32>,
        W: FnOnce(&[f32]),
        T: FnOnce(f32);
}

pub mod cqt;
pub mod waveform;

use std::sync::{Arc, Mutex};

pub use cqt::ConstantQTransformModel;
use num_complex::Complex;
pub use waveform::WaveformModel;
