// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use std::sync::Arc;

pub mod wayland;

pub trait WgpuSurface {
    fn device(&self) -> &wgpu::Device;
    fn queue(&self) -> &Arc<wgpu::Queue>;
    fn surface_id(&self) -> u32;
    fn swapchain_format(&self) -> Option<wgpu::TextureFormat>;
}
