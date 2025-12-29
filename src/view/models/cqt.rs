// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use std::sync::{Arc, Mutex};

use crate::view::models::ValuesRange;

use super::AudioModel;
use num_complex::Complex;
use tracing::instrument;

pub struct ConstantQTransformModel {
    y_values: Vec<f32>,
}

impl ConstantQTransformModel {
    pub fn new(cqt_size: usize) -> Self {
        Self {
            y_values: vec![0.0; cqt_size],
        }
    }

    /// Converts complex outputs to magnitudes and then to dB
    #[instrument(skip(self, cqt))]
    fn update_magnitudes_db_scaled(&mut self, cqt: Arc<Mutex<Vec<Complex<f64>>>>) {
        // Range floor
        const MIN_DB: f64 = -70.0;
        // For a single sine wave, 0dBFS corresponds to a magnitude of 1.0.
        // But since the CQT will distribute energy across multiple bins, set a more reasonable
        // max to be able to reach the full highlight color in the shader.
        const MAX_DB: f64 = -25.0;
        // Avoid log of zero
        const FLOOR: f64 = 1e-9;

        // Compute dB values and normalize to 0..1 based on MIN_DB and MAX_DB
        let items = cqt.lock().unwrap();
        for (i, c) in items.iter().enumerate() {
            let magnitude = c.norm().max(FLOOR);
            let db = 20.0 * magnitude.log10();

            let clipped = db.max(MIN_DB);
            self.y_values[i] = ((clipped - MIN_DB) / (MAX_DB - MIN_DB)) as f32;
        }
    }
}

impl AudioModel for ConstantQTransformModel {
    fn stride_len(&self) -> usize {
        self.y_values.len()
    }

    fn values_range(&self) -> ValuesRange {
        ValuesRange::ZeroToOne
    }

    fn process_audio<I, W>(
        &mut self,
        _samples: I,
        cqt: Arc<Mutex<Vec<Complex<f64>>>>,
        write_values: W,
    ) where
        I: IntoIterator<Item = f32>,
        W: FnOnce(&[f32]),
    {
        self.update_magnitudes_db_scaled(cqt);
        write_values(&self.y_values);
    }
}
