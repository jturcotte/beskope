use crate::UiMessage;
use slint::{ComponentHandle, Global, Model, VecModel};
use splines::Interpolation;

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, Write},
};

slint::include_modules!();

#[derive(Clone, Serialize, Deserialize)]
pub struct PanelConfig {
    pub channels: RenderChannels,
    pub layout: PanelLayout,
    pub layer: PanelLayer,
    pub width: u32,
    pub exclusive_ratio: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TimeCurveConfig {
    pub control_points: Vec<ControlPoint>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Configuration {
    pub waveform: WaveformConfig,
    pub style: Style,
    pub panel: PanelConfig,
    pub time_curve: TimeCurveConfig,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            waveform: WaveformConfig {
                fill_color: slint::Color::from_argb_u8(205, 64, 112, 172),
                stroke_color: slint::Color::from_argb_u8(205, 0, 0, 0),
            },
            style: Style::Ridgeline,
            panel: PanelConfig {
                channels: RenderChannels::Both,
                layout: PanelLayout::TwoPanels,
                layer: PanelLayer::Bottom,
                width: 280,
                exclusive_ratio: 0.6,
            },
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

pub fn init(send_ui_msg: impl Fn(UiMessage) + Clone + 'static) -> ConfigurationWindow {
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

    backend.on_waveform_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::ApplicationStateCallback(Box::new(
                move |state| {
                    state.config.waveform = config.clone();
                    state.for_each_view_mut(|view| {
                        view.set_waveform_config(config.clone());
                    });
                },
            )));
        }
    });
    backend.on_style_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::ApplicationStateCallback(Box::new(
                move |state| {
                    state.config.style = config;
                    state.recreate_views();
                },
            )));
        }
    });
    backend.on_panel_channels_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::ApplicationStateCallback(Box::new(
                move |state| {
                    state.config.panel.channels = config;
                    state.recreate_views();
                },
            )));
        }
    });
    backend.on_panel_layout_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::WlrWaylandEventHandlerCallback(Box::new(
                move |handler, conn, qh| {
                    handler.app_state.config.panel.layout = config;
                    handler.apply_panel_layout(conn, qh);
                },
            )));
        }
    });
    backend.on_panel_layer_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::WlrWaylandEventHandlerCallback(Box::new(
                move |handler, _, _| {
                    handler.app_state.config.panel.layer = config;
                    handler.set_panel_layer(config);
                },
            )));
        }
    });
    backend.on_panel_width_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::WlrWaylandEventHandlerCallback(Box::new(
                move |handler, _, _| {
                    handler.app_state.config.panel.width = config as u32;
                    handler.set_panel_width(config as u32);
                    handler.app_state.set_panel_width(config as u32);
                },
            )));
        }
    });
    backend.on_panel_exclusive_ratio_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::WlrWaylandEventHandlerCallback(Box::new(
                move |handler, _, _| {
                    handler.app_state.config.panel.exclusive_ratio = config;
                    handler.set_panel_exclusive_ratio(config);
                },
            )));
        }
    });
    backend.on_time_curve_control_points_changed({
        let send = send_ui_msg.clone();
        move |control_points_model| {
            let control_points: Vec<ControlPoint> = control_points_model.iter().collect();
            send(UiMessage::ApplicationStateCallback(Box::new(move |ws| {
                ws.config.time_curve.control_points = control_points.clone();
                ws.for_each_view_mut(|view| {
                    view.set_time_curve_control_points(control_points.clone());
                });
            })));
        }
    });

    window.on_ok_clicked({
        let window_weak = window.as_weak();
        let send = send_ui_msg.clone();
        move || {
            send(UiMessage::ApplicationStateCallback(Box::new(
                move |state| {
                    if let Err(e) = state.config.save() {
                        eprintln!("Failed to save configuration: {}", e);
                    }
                },
            )));

            let window = window_weak.upgrade().unwrap();
            window.hide().unwrap();
        }
    });

    window.on_cancel_clicked({
        let window_weak = window.as_weak();
        let send = send_ui_msg.clone();
        move || {
            send(UiMessage::WlrWaylandEventHandlerCallback(Box::new(
                move |handler, conn, qh| {
                    handler.app_state.reload_configuration();
                    // TODO: Track whether the layout configuration was reverted instead of
                    //       unconditionally recreating surfaces.
                    handler.apply_panel_layout(conn, qh);
                },
            )));

            let window = window_weak.upgrade().unwrap();
            window.hide().unwrap();
        }
    });

    window
}

impl ConfigurationWindow {
    pub fn update_from_configuration(&self, config: &Configuration) {
        self.invoke_set_waveform_config(config.waveform.clone());
        self.invoke_set_style(config.style.clone());
        self.invoke_set_panel_channels(config.panel.channels.clone());
        self.invoke_set_panel_layout(config.panel.layout.clone());
        self.invoke_set_panel_layer(config.panel.layer.clone());
        self.invoke_set_panel_width(config.panel.width as i32);
        self.invoke_set_panel_exclusive_ratio(config.panel.exclusive_ratio as f32);
        self.invoke_set_time_curve_control_points(slint::ModelRc::new(VecModel::from(
            config.time_curve.control_points.clone(),
        )));
    }
}
