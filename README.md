# freeflow-linux

Push-to-talk voice dictation for Linux using Groq Whisper + LLM post-processing.

A Linux equivalent of [FreeFlow](https://github.com/zachlatta/freeflow) (macOS). Hold a
configurable hotkey (default: Right Ctrl) to record your voice — the audio is transcribed
by Groq Whisper and cleaned up by a Groq LLM, then pasted into whatever app you have focused.

## How it works

1. Hold the hotkey for 1 second (a beep confirms recording has started)
2. Speak
3. Release the hotkey — the transcript is cleaned up and pasted into the focused window

The 1-second hold threshold prevents accidental triggers.

## Requirements

### System packages

```bash
# Audio (PortAudio runtime)
sudo apt install libportaudio2 pipewire-alsa

# Paste — X11
sudo apt install xdotool xclip

# Paste — Wayland
sudo apt install wl-clipboard wtype      # wlroots compositors (Sway, Hyprland, KDE)
sudo apt install ydotool                 # GNOME Wayland (also needs ydotoold daemon)
```

### Python packages

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Setup

### 1. Groq API key

Get a free API key at [console.groq.com](https://console.groq.com/).

Set it in the config file (created automatically on first run):

```bash
~/.config/freeflow-linux/config.toml
```

Or export it as an environment variable:

```bash
export GROQ_API_KEY=gsk_...
```

### 2. Input group (required for hotkey capture)

freeflow-linux reads keyboard events directly from `/dev/input` via evdev. This requires
membership in the `input` group:

```bash
sudo usermod -aG input $USER
# Log out and back in for the group to take effect
```

### 3. GNOME Wayland: ydotool setup

If you use GNOME on Wayland, you need `ydotool` for paste to work:

```bash
sudo apt install ydotool
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/80-uinput.rules
sudo udevadm control --reload-rules
sudo systemctl enable --now ydotool
# Log out and back in
```

## Running

```bash
cd ~/code/freeflow-linux
.venv/bin/python freeflow_linux.py
```

### Dry-run (check config and devices without starting)

```bash
.venv/bin/python freeflow_linux.py --dry-run
```

## Configuration

Config file: `~/.config/freeflow-linux/config.toml` (created automatically on first run)

```toml
api_key = "gsk_..."          # Groq API key (or use GROQ_API_KEY env var)
hotkey = "KEY_RIGHTCTRL"     # Right Ctrl — change to KEY_F9 etc. if preferred
# audio_device = ""          # Leave empty to use system default mic
```

To find available hotkey names, run `evtest` and press the key you want.

## Autostart (systemd user service)

```bash
mkdir -p ~/.config/systemd/user
cp freeflow-linux.service.example ~/.config/systemd/user/freeflow-linux.service
# Edit the ExecStart path if needed
systemctl --user daemon-reload
systemctl --user enable --now freeflow-linux
```

Example service file:

```ini
[Unit]
Description=FreeFlow Linux voice dictation daemon
After=graphical-session.target

[Service]
ExecStart=/path/to/freeflow-linux/.venv/bin/python /path/to/freeflow-linux/freeflow_linux.py
Restart=on-failure
RestartSec=3
Environment=XDG_SESSION_TYPE=x11
Environment=DISPLAY=:0

[Install]
WantedBy=default.target
```

## Hotkey compatibility notes

- **Fn key**: Does not work — handled by keyboard firmware, never reaches the kernel
- **Right Ctrl** (default): Reliable, rarely used for other shortcuts
- **F9, ScrollLock, media keys**: All work well as alternatives

## X11 vs Wayland

| Feature | X11 | Wayland |
|---|---|---|
| Hotkey capture (evdev) | Yes | Yes |
| Paste | xclip + xdotool | wl-copy + wtype/ydotool |
| Window context | Yes (xdotool) | No |
| Terminal detection (Ctrl+Shift+V) | Yes | No |

## Credits

Inspired by [FreeFlow](https://github.com/zachlatta/freeflow) by Zach Latta.
Uses [Groq](https://groq.com/) for fast Whisper transcription and LLM post-processing.
