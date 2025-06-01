# Beskope

A desktop waveform visualizer for Wayland and PipeWire.

![Riggeline Plot Style](https://github.com/user-attachments/assets/abca2b2f-6df2-4f89-815f-22c31ccde5ae)
![Compressed Line Style](https://github.com/user-attachments/assets/aaa3459b-f2b0-4c93-a407-92fc86535eed)

## Features
- Displays a waveform visualization of the currently played audio through the PipeWire monitor port
- Supports high frame rate rendering with reasonable CPU usage
- Integrates with KDE, Sway or Hyprland Wayland sessions as a transparent panel attached to your screen edge

## Install

[See the latest release page](https://github.com/jturcotte/beskope/releases/latest)

This requires a Wayland compositor supporting the 
[wlr layer shell protocol](https://wayland.app/protocols/wlr-layer-shell-unstable-v1#compositor-support)
(i.e. not Gnome).

## License

The source code is available under the terms of the MIT license
(See [LICENSE-MIT](LICENSE-MIT) for details).

However, because of the use of GPL dependencies, compiled binaries
are licensed under the terms of the GPLv3 (See [LICENSE-GPL](LICENSE-GPL)).
