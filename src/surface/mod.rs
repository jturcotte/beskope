// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use std::sync::Arc;

use wayland::LayerShellCanvasContext;

use crate::{ApplicationState, ui};

pub mod slint;
pub mod wayland;

pub trait WgpuSurface {
    fn device(&self) -> &wgpu::Device;
    fn queue(&self) -> &Arc<wgpu::Queue>;
    fn surface_id(&self) -> u32;
    fn swapchain_format(&self) -> Option<wgpu::TextureFormat>;
}

pub enum GlobalCanvasContext {
    LayerShell(LayerShellCanvasContext),
    Slint(()),
}

pub trait GlobalCanvas {
    fn app_state(&mut self) -> &mut ApplicationState;
    fn apply_panel_width_change(&mut self);
    fn apply_panel_exclusive_ratio_change(&mut self);
    fn apply_panel_layout(&mut self, context: &GlobalCanvasContext);
    fn set_panel_layer(&mut self, layer: ui::PanelLayer);
}
