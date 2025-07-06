// Copyright © 2025 Jocelyn Turcotte <turcotte.j@gmail.com>
// SPDX-License-Identifier: MIT

use crate::{AppMessage, GlobalCanvas, GlobalCanvasContext};
use slint::{
    ComponentHandle, Global, Model, VecModel,
    wgpu_24::{WGPUConfiguration, WGPUSettings},
};
use splines::Interpolation;

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, Write},
    mem::offset_of,
};

slint::include_modules!();

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneralConfig {
    pub channels: RenderChannels,
    pub layout: PanelLayout,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RidgelineConfig {
    pub layer: PanelLayer,
    pub width: u32,
    pub exclusive_ratio: f32,
    pub fill_color: slint::Color,
    pub stroke_color: slint::Color,
    pub highlight_color: slint::Color,
    pub horizon_offset: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeCurveConfig {
    pub control_points: Vec<ControlPoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedConfig {
    pub layer: PanelLayer,
    pub width: u32,
    pub exclusive_ratio: f32,
    pub fill_color: slint::Color,
    pub stroke_color: slint::Color,
    pub time_curve: TimeCurveConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Configuration {
    pub style: Style,
    pub general: GeneralConfig,
    pub ridgeline: RidgelineConfig,
    pub compressed: CompressedConfig,
}

// Change tracking IDs for changes applied lazily before rendering. Use their byte offset within the configuration struct.
pub const RIDGELINE_WIDTH: usize = offset_of!(Configuration, ridgeline.width);
pub const RIDGELINE_FILL_COLOR: usize = offset_of!(Configuration, ridgeline.fill_color);
pub const RIDGELINE_STROKE_COLOR: usize = offset_of!(Configuration, ridgeline.stroke_color);
pub const RIDGELINE_HIGHLIGHT_COLOR: usize = offset_of!(Configuration, ridgeline.highlight_color);
pub const RIDGELINE_HORIZON_OFFSET: usize = offset_of!(Configuration, ridgeline.horizon_offset);
pub const COMPRESSED_FILL_COLOR: usize = offset_of!(Configuration, compressed.fill_color);
pub const COMPRESSED_STROKE_COLOR: usize = offset_of!(Configuration, compressed.stroke_color);
pub const COMPRESSED_TIME_CURVE_CONTROL_POINTS: usize =
    offset_of!(Configuration, compressed.time_curve.control_points);

impl Default for Configuration {
    fn default() -> Self {
        Self {
            style: Style::Ridgeline,
            general: GeneralConfig {
                channels: RenderChannels::Both,
                layout: PanelLayout::TwoPanels,
            },
            ridgeline: RidgelineConfig {
                layer: PanelLayer::Bottom,
                width: 360,
                exclusive_ratio: 0.6,
                fill_color: slint::Color::from_argb_u8(205, 64, 112, 172),
                stroke_color: slint::Color::from_argb_u8(205, 0, 0, 0),
                highlight_color: slint::Color::from_argb_u8(205, 255, 255, 255),
                horizon_offset: 0.0,
            },
            compressed: CompressedConfig {
                layer: PanelLayer::Top,
                width: 80,
                exclusive_ratio: 0.55,
                fill_color: slint::Color::from_argb_u8(153, 128, 128, 128),
                stroke_color: slint::Color::from_argb_u8(205, 0, 0, 0),
                time_curve: TimeCurveConfig {
                    control_points: vec![
                        ControlPoint { t: -3.0, v: 0.0 },
                        ControlPoint { t: -1.75, v: 0.025 },
                        ControlPoint { t: -0.75, v: 0.10 },
                        ControlPoint { t: -0.25, v: 0.25 },
                        ControlPoint {
                            t: -1.0 / 15.0,
                            v: 0.4,
                        },
                        ControlPoint {
                            t: -1.0 / 60.0,
                            v: 0.6,
                        },
                    ],
                },
            },
        }
    }
}

fn get_config_path() -> Option<std::path::PathBuf> {
    let project_dirs = ProjectDirs::from("", "", "beskope")?;
    Some(project_dirs.config_dir().join("config.toml"))
}

impl Configuration {
    pub fn save(&self) -> io::Result<()> {
        let config_path = get_config_path().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "Config directory not found")
        })?;

        let toml_str = toml::to_string_pretty(self).map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("TOML serialization error: {}", e),
            )
        })?;
        let mut file = fs::File::create(config_path)?;
        file.write_all(toml_str.as_bytes())?;
        Ok(())
    }

    pub fn load() -> io::Result<Configuration> {
        if let Some(config_path) = get_config_path() {
            if !config_path.exists() {
                return Ok(Configuration::default());
            }
            let toml_str = fs::read_to_string(config_path)?;
            let config: Configuration = toml::from_str(&toml_str).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("TOML deserialization error: {}", e),
                )
            })?;
            Ok(config)
        } else {
            Ok(Configuration::default())
        }
    }
}

pub fn init(
    // send_app_msg: impl Fn(AppMessage) + Clone + 'static,
    send_to_app: impl Fn(Box<dyn FnOnce(&mut crate::ApplicationState) + Send + 'static>)
    + Clone
    + 'static,
    send_to_canvas: impl Fn(Box<dyn FnOnce(&mut dyn GlobalCanvas, GlobalCanvasContext) + Send>)
    + Clone
    + 'static,
) -> ConfigurationWindow {
    // FIXME: Weird to have this here if all the stuff is in main
    let mut wgpu_settings = WGPUSettings::default();
    // Slint defaults to WebGL2 limits, but use the real wgpu default.
    wgpu_settings.device_required_limits = wgpu::Limits::default();

    slint::BackendSelector::new()
        .require_wgpu_24(WGPUConfiguration::Automatic(wgpu_settings))
        .select()
        .expect("Unable to create Slint backend with WGPU based renderer");

    let window = ConfigurationWindow::new().unwrap();
    let time_curve_editor = TimeCurveEditor::get(&window);
    let backend = Backend::get(&window);

    time_curve_editor.on_change_point(move |control_points_model, i, cp| {
        control_points_model.set_row_data(i as usize, cp)
    });
    time_curve_editor.on_add_point(move |control_points_model, cp| {
        let vec_model = control_points_model
            .as_any()
            .downcast_ref::<VecModel<ControlPoint>>()
            .unwrap();
        vec_model.push(cp);
    });
    time_curve_editor.on_remove_point(move |control_points_model, i| {
        let vec_model = control_points_model
            .as_any()
            .downcast_ref::<VecModel<ControlPoint>>()
            .unwrap();
        vec_model.remove(i as usize);
    });
    time_curve_editor.on_draw_curve(move |control_points, width, height| {
        let control_points_iter: Vec<_> = (0..control_points.row_count())
            .map(|i| {
                let row_data = control_points.row_data(i).unwrap();
                (row_data.t, row_data.v, Interpolation::CatmullRom)
            })
            .collect();

        let control_points_with_prefix_suffix_iter =
            std::iter::once((-3.0, 0.0, Interpolation::Linear))
                .chain(control_points_iter)
                .chain(std::iter::once((0.0, 1.0, Interpolation::Linear)))
                .chain(std::iter::once((0.001, 1.0, Interpolation::Linear)));

        // Create the spline
        let spline = splines::Spline::from_vec(
            control_points_with_prefix_suffix_iter
                .map(|(x, y, interpolation)| splines::Key::new(x, y, interpolation))
                .collect(),
        );

        let mut svg_path = String::new();
        let mut command = 'M';
        for i in 0..(width as usize) {
            let t = (i as f32 / width as f32) * 3.0 - 3.0;
            if let Some(y) = spline.sample(t) {
                svg_path.push_str(&format!("{}{},{}", command, i, (1.0 - y) * height as f32));
            }
            command = 'L';
        }

        svg_path.into()
    });

    backend.on_style_changed({
        let send_to_canvas = send_to_canvas.clone();
        move |config| {
            send_to_canvas(Box::new(move |handler, context| {
                handler.app_state().config.style = config;
                handler.apply_panel_layout(&context);
            }));
        }
    });
    backend.on_panel_channels_changed({
        let send_to_app = send_to_app.clone();
        move |config| {
            send_to_app(Box::new(move |app_state| {
                app_state.config.general.channels = config;
                app_state.recreate_views();
            }));
        }
    });
    backend.on_panel_layout_changed({
        let send_to_canvas = send_to_canvas.clone();
        move |config| {
            send_to_canvas(Box::new(move |handler, context| {
                handler.app_state().config.general.layout = config;
                handler.apply_panel_layout(&context);
            }));
        }
    });
    backend.on_ridgeline_panel_layer_changed({
        let send_to_canvas = send_to_canvas.clone();
        move |config| {
            send_to_canvas(Box::new(move |handler, _| {
                handler.app_state().config.ridgeline.layer = config;
                if handler.app_state().config.style == Style::Ridgeline {
                    handler.set_panel_layer(config);
                }
            }));
        }
    });
    backend.on_ridgeline_panel_width_changed({
        let send_to_canvas = send_to_canvas.clone();
        move |config| {
            send_to_canvas(Box::new(move |handler, _| {
                handler.app_state().config.ridgeline.width = config as u32;
                handler
                    .app_state()
                    .lazy_config_changes
                    .insert(RIDGELINE_WIDTH);
                if handler.app_state().config.style == Style::Ridgeline {
                    handler.apply_panel_width_change();
                }
            }));
        }
    });
    backend.on_ridgeline_panel_exclusive_ratio_changed({
        let send_to_canvas = send_to_canvas.clone();
        move |config| {
            send_to_canvas(Box::new(move |handler, _| {
                handler.app_state().config.ridgeline.exclusive_ratio = config;
                if handler.app_state().config.style == Style::Ridgeline {
                    handler.apply_panel_exclusive_ratio_change();
                }
            }));
        }
    });
    backend.on_ridgeline_fill_color_changed({
        let send_to_app = send_to_app.clone();
        move |config| {
            send_to_app(Box::new(move |app_state| {
                app_state.config.ridgeline.fill_color = config;
                app_state.lazy_config_changes.insert(RIDGELINE_FILL_COLOR);
            }));
        }
    });
    backend.on_ridgeline_stroke_color_changed({
        let send_to_app = send_to_app.clone();
        move |config| {
            send_to_app(Box::new(move |app_state| {
                app_state.config.ridgeline.stroke_color = config;
                app_state.lazy_config_changes.insert(RIDGELINE_STROKE_COLOR);
            }));
        }
    });
    backend.on_ridgeline_highlight_color_changed({
        let send_to_app = send_to_app.clone();
        move |config| {
            send_to_app(Box::new(move |app_state| {
                app_state.config.ridgeline.highlight_color = config;
                app_state
                    .lazy_config_changes
                    .insert(RIDGELINE_HIGHLIGHT_COLOR);
            }));
        }
    });
    backend.on_ridgeline_horizon_offset_changed({
        let send_to_app = send_to_app.clone();
        move |offset| {
            send_to_app(Box::new(move |app_state| {
                app_state.config.ridgeline.horizon_offset = offset;
                app_state
                    .lazy_config_changes
                    .insert(RIDGELINE_HORIZON_OFFSET);
            }));
        }
    });
    backend.on_compressed_panel_layer_changed({
        let send_to_canvas = send_to_canvas.clone();
        move |config| {
            send_to_canvas(Box::new(move |handler, _| {
                handler.app_state().config.compressed.layer = config;
                if handler.app_state().config.style == Style::Compressed {
                    handler.set_panel_layer(config);
                }
            }));
        }
    });
    backend.on_compressed_panel_width_changed({
        let send_to_canvas = send_to_canvas.clone();
        move |config| {
            send_to_canvas(Box::new(move |handler, _| {
                handler.app_state().config.compressed.width = config as u32;
                if handler.app_state().config.style == Style::Compressed {
                    handler.apply_panel_width_change();
                }
            }));
        }
    });
    backend.on_compressed_panel_exclusive_ratio_changed({
        let send_to_canvas = send_to_canvas.clone();
        move |config| {
            send_to_canvas(Box::new(move |handler, _| {
                handler.app_state().config.compressed.exclusive_ratio = config;
                if handler.app_state().config.style == Style::Compressed {
                    handler.apply_panel_exclusive_ratio_change();
                }
            }));
        }
    });
    backend.on_compressed_fill_color_changed({
        let send_to_app = send_to_app.clone();
        move |config| {
            send_to_app(Box::new(move |app_state| {
                app_state.config.compressed.fill_color = config;
                app_state.lazy_config_changes.insert(COMPRESSED_FILL_COLOR);
            }));
        }
    });
    backend.on_compressed_stroke_color_changed({
        let send_to_app = send_to_app.clone();
        move |config| {
            send_to_app(Box::new(move |app_state| {
                app_state.config.compressed.stroke_color = config;
                app_state
                    .lazy_config_changes
                    .insert(COMPRESSED_STROKE_COLOR);
            }));
        }
    });
    backend.on_compressed_time_curve_control_points_changed({
        let send_to_app = send_to_app.clone();
        move |control_points_model| {
            let control_points: Vec<ControlPoint> = control_points_model.iter().collect();
            send_to_app(Box::new(move |app_state| {
                app_state.config.compressed.time_curve.control_points = control_points.clone();
                app_state
                    .lazy_config_changes
                    .insert(COMPRESSED_TIME_CURVE_CONTROL_POINTS);
            }));
        }
    });

    window.on_ok_clicked({
        let window_weak = window.as_weak();
        let send_to_app = send_to_app.clone();
        move || {
            send_to_app(Box::new(move |app_state| {
                if let Err(e) = app_state.config.save() {
                    eprintln!("Failed to save configuration: {}", e);
                }
            }));

            let window = window_weak.upgrade().unwrap();
            window.hide().unwrap();
        }
    });

    window.on_cancel_clicked({
        let window_weak = window.as_weak();
        let send_to_app = send_to_app.clone();
        move || {
            send_to_canvas(Box::new(move |handler, context| {
                handler.app_state().reload_configuration();
                // TODO: Track whether the layout configuration was reverted instead of
                //       unconditionally recreating surfaces.
                handler.apply_panel_layout(&context);
            }));

            let window = window_weak.upgrade().unwrap();
            window.hide().unwrap();
        }
    });

    window
}

impl ConfigurationWindow {
    pub fn update_from_configuration(&self, config: &Configuration) {
        self.invoke_set_style(config.style.clone());
        self.invoke_set_panel_channels(config.general.channels.clone());
        self.invoke_set_panel_layout(config.general.layout.clone());
        self.invoke_set_ridgeline_panel_layer(config.ridgeline.layer.clone());
        self.invoke_set_ridgeline_panel_width(config.ridgeline.width as i32);
        self.invoke_set_ridgeline_panel_exclusive_ratio(config.ridgeline.exclusive_ratio as f32);
        self.invoke_set_ridgeline_fill_color(config.ridgeline.fill_color.clone());
        self.invoke_set_ridgeline_stroke_color(config.ridgeline.stroke_color.clone());
        self.invoke_set_ridgeline_highlight_color(config.ridgeline.highlight_color.clone());
        self.invoke_set_ridgeline_horizon_offset(config.ridgeline.horizon_offset as f32);
        self.invoke_set_compressed_panel_layer(config.compressed.layer.clone());
        self.invoke_set_compressed_panel_width(config.compressed.width as i32);
        self.invoke_set_compressed_panel_exclusive_ratio(config.compressed.exclusive_ratio as f32);
        self.invoke_set_compressed_fill_color(config.compressed.fill_color.clone());
        self.invoke_set_compressed_stroke_color(config.compressed.stroke_color.clone());
        self.invoke_set_compressed_time_curve_control_points(slint::ModelRc::new(VecModel::from(
            config.compressed.time_curve.control_points.clone(),
        )));
    }
}
