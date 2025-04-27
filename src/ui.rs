use crate::UiMessage;
use slint::{ComponentHandle, Global, Model, VecModel};
use splines::Interpolation;

slint::include_modules!();

pub fn init(send_ui_msg: impl Fn(UiMessage) + Clone + 'static) -> ConfigurationWindow {
    let window = ConfigurationWindow::new().unwrap();
    let configuration = Configuration::get(&window);

    configuration.on_waveform_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::ApplicationStateCallback(Box::new(
                move |state| {
                    state.set_waveform_config(config);
                },
            )));
        }
    });

    let time_curve_drawer = TimeCurveDrawer::get(&window);

    configuration.on_time_curve_control_point_changed({
        let window_weak = window.as_weak();
        let send = send_ui_msg.clone();
        move |i, cp| {
            let window = window_weak.upgrade().unwrap();
            let time_curve_drawer = TimeCurveDrawer::get(&window);
            time_curve_drawer.set_dummy_dep(!time_curve_drawer.get_dummy_dep());
            let configuration = Configuration::get(&window);
            let control_points = configuration.get_time_curve_control_points();
            control_points.set_row_data(i as usize, cp);

            let updated_points: Vec<_> = (0..control_points.row_count())
                .map(|i| control_points.row_data(i).unwrap())
                .collect();

            send(UiMessage::ApplicationStateCallback(Box::new(move |ws| {
                ws.set_time_curve_control_points(updated_points);
            })));
        }
    });
    configuration.on_time_curve_control_point_added({
        let window_weak = window.as_weak();
        let send = send_ui_msg.clone();
        move |cp| {
            let window = window_weak.upgrade().unwrap();
            let time_curve_drawer = TimeCurveDrawer::get(&window);
            time_curve_drawer.set_dummy_dep(!time_curve_drawer.get_dummy_dep());
            let configuration = Configuration::get(&window);
            let control_points = configuration.get_time_curve_control_points();
            let vec_model = control_points
                .as_any()
                .downcast_ref::<VecModel<ControlPoint>>()
                .unwrap();
            vec_model.push(cp);

            let updated_points: Vec<_> = (0..control_points.row_count())
                .map(|i| control_points.row_data(i).unwrap())
                .collect();

            send(UiMessage::ApplicationStateCallback(Box::new(move |ws| {
                ws.set_time_curve_control_points(updated_points);
            })));
        }
    });
    configuration.on_time_curve_control_point_removed({
        let window_weak = window.as_weak();
        let send = send_ui_msg.clone();
        move |i| {
            let window = window_weak.upgrade().unwrap();
            let time_curve_drawer = TimeCurveDrawer::get(&window);
            time_curve_drawer.set_dummy_dep(!time_curve_drawer.get_dummy_dep());
            let configuration = Configuration::get(&window);
            let control_points = configuration.get_time_curve_control_points();
            let vec_model = control_points
                .as_any()
                .downcast_ref::<VecModel<ControlPoint>>()
                .unwrap();
            vec_model.remove(i as usize);

            let updated_points: Vec<_> = (0..control_points.row_count())
                .map(|i| control_points.row_data(i).unwrap())
                .collect();

            send(UiMessage::ApplicationStateCallback(Box::new(move |ws| {
                ws.set_time_curve_control_points(updated_points);
            })));
        }
    });
    configuration.on_all_time_curve_control_points_changed({
        let send = send_ui_msg.clone();
        move |control_points_model| {
            let control_points = control_points_model.iter().collect();
            send(UiMessage::ApplicationStateCallback(Box::new(move |ws| {
                ws.set_time_curve_control_points(control_points);
            })));
        }
    });
    time_curve_drawer.on_draw_curve({
        let window_weak = window.as_weak();
        move |width, height, _| {
            let window = window_weak.upgrade().unwrap();
            let configuration = Configuration::get(&window);
            let control_points = configuration.get_time_curve_control_points();

            let control_points_iter: Vec<_> = (0..control_points.row_count())
                .map(|i| {
                    let row_data = control_points.row_data(i).unwrap();
                    (row_data.t, row_data.v, Interpolation::CatmullRom)
                })
                .collect();
            // println!("Control points: {:?}", control_points_iter);
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
        }
    });

    configuration.on_style_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::ApplicationStateCallback(Box::new(
                move |state| {
                    state.set_style(config);
                },
            )));
        }
    });
    configuration.on_panel_channels_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::ApplicationStateCallback(Box::new(
                move |state| {
                    state.set_channels(config);
                },
            )));
        }
    });
    configuration.on_panel_layout_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::WlrWaylandEventHandlerCallback(Box::new(
                move |handler, conn, qh| {
                    handler.set_panel_layout(config, conn, qh);
                },
            )));
        }
    });
    configuration.on_panel_layer_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::WlrWaylandEventHandlerCallback(Box::new(
                move |handler, _, _| {
                    handler.set_panel_layer(config);
                },
            )));
        }
    });
    configuration.on_panel_width_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::WlrWaylandEventHandlerCallback(Box::new(
                move |handler, _, _| {
                    handler.set_panel_width(config as u32);
                },
            )));
            send(UiMessage::ApplicationStateCallback(Box::new(
                move |state| {
                    state.set_panel_width(config as u32);
                },
            )));
        }
    });
    configuration.on_panel_exclusive_ratio_changed({
        let send = send_ui_msg.clone();
        move |config| {
            send(UiMessage::WlrWaylandEventHandlerCallback(Box::new(
                move |handler, _, _| {
                    handler.set_panel_exclusive_ratio(config);
                },
            )));
        }
    });

    window.on_ok_clicked({
        let window_weak = window.as_weak();
        move || {
            let window = window_weak.upgrade().unwrap();
            let configuration = Configuration::get(&window);
            if let Err(e) = crate::config::save_configuration(&configuration) {
                eprintln!("Failed to save configuration: {}", e);
            }
            window.hide().unwrap();
        }
    });

    window.on_cancel_clicked({
        let window_weak = window.as_weak();
        move || {
            let window = window_weak.upgrade().unwrap();
            let configuration = Configuration::get(&window);
            if let Err(e) = crate::config::load_configuration(&configuration) {
                eprintln!("Failed to reload the configuration from file: {}", e);
            }
            window.hide().unwrap();
        }
    });

    window.show().unwrap();
    window
}
