# Beskope

A music visualizer for Linux and PipeWire.

- Displays a visualization of the currently played audio through the PipeWire monitor port
- Renders it as raw waveform or Constant-Q Transform spectrogram
- Supports high frame rate rendering with reasonable CPU usage

### Window mode
Rendered in a normal window together with MPRIS controls.

![Frequency Riggeline Plot Style](https://github.com/user-attachments/assets/daaf4b7c-82c9-4b59-b174-28c72dbfe222)

### Desktop mode
Integrates with KDE, Sway or Hyprland Wayland sessions as a transparent panel attached to your screen edge in desktop mode. This requires a Wayland compositor supporting the 
[wlr layer shell protocol](https://wayland.app/protocols/wlr-layer-shell-unstable-v1#compositor-support)
(i.e. not Gnome).

![Waveform Riggeline Plot Style](https://github.com/user-attachments/assets/abca2b2f-6df2-4f89-815f-22c31ccde5ae)
![Compressed Line Style](https://github.com/user-attachments/assets/aaa3459b-f2b0-4c93-a407-92fc86535eed)

## Install

[See the latest release page](https://github.com/jturcotte/beskope/releases/latest)

## License

The source code is available under the terms of the MIT license
(See [LICENSE-MIT](LICENSE-MIT) for details).

However, because of the use of GPL dependencies, compiled binaries
are licensed under the terms of the GPLv3 (See [LICENSE-GPL](LICENSE-GPL)).
