use directories::ProjectDirs;
use slint::{Model, ModelRc, VecModel};
use std::io::ErrorKind;
use toml_edit::{DocumentMut, table, value};

fn get_config_path() -> Option<std::path::PathBuf> {
    let project_dirs = ProjectDirs::from("", "", "beskope")?;
    Some(project_dirs.config_dir().join("config.toml"))
}

pub fn save_configuration(
    configuration: &crate::ui::Configuration,
) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = get_config_path().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "Config directory not found")
    })?;

    let mut doc = if config_path.exists() {
        let existing_data = std::fs::read_to_string(&config_path)?;
        existing_data
            .parse::<DocumentMut>()
            .unwrap_or_else(|_| DocumentMut::new())
    } else {
        DocumentMut::new()
    };

    let waveform = configuration.get_waveform();
    doc["waveform"] = table();
    // Use "#rrggbbaa" format for colors
    doc["waveform"]["fill_color"] = value(format!(
        "#{:02x}{:02x}{:02x}{:02x}",
        waveform.fill_color.red(),
        waveform.fill_color.green(),
        waveform.fill_color.blue(),
        waveform.fill_color.alpha()
    ));
    doc["waveform"]["stroke_color"] = value(format!(
        "#{:02x}{:02x}{:02x}{:02x}",
        waveform.stroke_color.red(),
        waveform.stroke_color.green(),
        waveform.stroke_color.blue(),
        waveform.stroke_color.alpha()
    ));

    doc["style"] = value(format!("{:?}", configuration.get_style()));

    doc["panel"] = table();
    doc["panel"]["channels"] = value(format!("{:?}", configuration.get_panel_channels()));
    doc["panel"]["layout"] = value(format!("{:?}", configuration.get_panel_layout()));
    doc["panel"]["layer"] = value(format!("{:?}", configuration.get_panel_layer()));
    doc["panel"]["width"] = value(configuration.get_panel_width() as i64);
    doc["panel"]["exclusive_ratio"] = value(configuration.get_panel_exclusive_ratio() as f64);

    let control_points = configuration.get_time_curve_control_points();
    let mut sorted_points: Vec<crate::ui::ControlPoint> = control_points.iter().collect();
    sorted_points.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
    let points_array: toml_edit::ArrayOfTables = sorted_points
        .iter()
        .map(|cp| {
            let mut table = toml_edit::Table::new();
            table["t"] = value(cp.t as f64);
            table["v"] = value(cp.v as f64);
            table
        })
        .collect();

    doc["time_curve"] = table();
    doc["time_curve"]["control_points"] = toml_edit::Item::ArrayOfTables(points_array);

    std::fs::create_dir_all(config_path.parent().unwrap())?;
    std::fs::write(config_path, doc.to_string())?;
    Ok(())
}

pub fn load_configuration(
    configuration: &crate::ui::Configuration,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(config_path) = get_config_path() {
        if !config_path.exists() {
            return Ok(());
        }
        let existing_data = std::fs::read_to_string(&config_path)?;
        let doc = existing_data.parse::<DocumentMut>()?;

        if let Some(style) = doc
            .get("style")
            .and_then(|i| i.as_str())
            .and_then(parse_style)
        {
            configuration.set_style(style.into());
        }

        if let Some(waveform_item) = doc.get("waveform") {
            let mut waveform = configuration.get_waveform();
            if let Some(color) = waveform_item.get("fill_color").and_then(|i| i.as_str()) {
                waveform.fill_color = parse_color(color)?;
            }
            if let Some(color) = waveform_item.get("stroke_color").and_then(|i| i.as_str()) {
                waveform.stroke_color = parse_color(color)?;
            }
            configuration.set_waveform(waveform);
        }

        if let Some(panel_item) = doc.get("panel") {
            if let Some(panel_channels) = panel_item
                .get("channels")
                .and_then(|i| i.as_str())
                .and_then(parse_panel_channels)
            {
                configuration.set_panel_channels(panel_channels.into());
            }
            if let Some(panel_layout) = panel_item
                .get("layout")
                .and_then(|i| i.as_str())
                .and_then(parse_panel_layout)
            {
                configuration.set_panel_layout(panel_layout.into());
            }
            if let Some(panel_layer) = panel_item
                .get("layer")
                .and_then(|i| i.as_str())
                .and_then(parse_panel_layer)
            {
                configuration.set_panel_layer(panel_layer.into());
            }
            if let Some(p_width) = panel_item.get("width").and_then(|i| i.as_integer()) {
                configuration.set_panel_width(p_width as i32);
            }
            if let Some(exclusive) = panel_item.get("exclusive_ratio").and_then(|i| i.as_float()) {
                configuration.set_panel_exclusive_ratio(exclusive as f32);
            }
        }
        if let Some(time_curve_item) = doc.get("time_curve") {
            if let Some(control_points) = time_curve_item
                .get("control_points")
                .and_then(|i| i.as_array_of_tables())
            {
                let mut parsed_control_points = Vec::new();
                for table in control_points.iter() {
                    if let (Some(t), Some(v)) = (
                        table.get("t").and_then(|t| t.as_float()),
                        table.get("v").and_then(|v| v.as_float()),
                    ) {
                        parsed_control_points.push(crate::ui::ControlPoint {
                            t: t as f32,
                            v: v as f32,
                        });
                    }
                }
                configuration.set_time_curve_control_points(ModelRc::new(VecModel::from(
                    parsed_control_points,
                )));
            }
        }
    }
    Ok(())
}

fn parse_color(s: &str) -> Result<slint::Color, Box<dyn std::error::Error>> {
    if s.starts_with('#') && s.len() == 9 {
        let r = u8::from_str_radix(&s[1..3], 16)?;
        let g = u8::from_str_radix(&s[3..5], 16)?;
        let b = u8::from_str_radix(&s[5..7], 16)?;
        let a = u8::from_str_radix(&s[7..9], 16)?;
        Ok(slint::Color::from_argb_u8(a, r, g, b))
    } else {
        Err(std::io::Error::new(ErrorKind::InvalidData, "Invalid color format").into())
    }
}

fn parse_style(s: &str) -> Option<crate::ui::Style> {
    match s {
        "Ridgeline" => Some(crate::ui::Style::Ridgeline),
        "Compressed" => Some(crate::ui::Style::Compressed),
        _ => None,
    }
}

fn parse_panel_channels(s: &str) -> Option<crate::ui::RenderChannels> {
    match s {
        "Single" => Some(crate::ui::RenderChannels::Single),
        "Both" => Some(crate::ui::RenderChannels::Both),
        _ => None,
    }
}

fn parse_panel_layout(s: &str) -> Option<crate::ui::PanelLayout> {
    match s {
        "TwoPanels" => Some(crate::ui::PanelLayout::TwoPanels),
        "SingleTop" => Some(crate::ui::PanelLayout::SingleTop),
        "SingleBottom" => Some(crate::ui::PanelLayout::SingleBottom),
        _ => None,
    }
}

fn parse_panel_layer(s: &str) -> Option<crate::ui::PanelLayer> {
    match s {
        "Overlay" => Some(crate::ui::PanelLayer::Overlay),
        "Top" => Some(crate::ui::PanelLayer::Top),
        "Bottom" => Some(crate::ui::PanelLayer::Bottom),
        "Background" => Some(crate::ui::PanelLayer::Background),
        _ => None,
    }
}
