#!/usr/bin/env python3
"""
freeflow-linux: Push-to-talk voice dictation daemon for Linux.

Hold the configured hotkey (default: Right Ctrl) to record, release to transcribe
and paste into the focused application.

Usage:
    python3 freeflow_linux.py           # run daemon
    python3 freeflow_linux.py --dry-run  # check config/devices/session, then exit
"""

import argparse
import asyncio
import io
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Imports (with friendly error messages)
# ---------------------------------------------------------------------------

try:
    import toml
except ImportError:
    sys.exit("Missing dependency: pip install toml")

try:
    import numpy as np
except ImportError:
    sys.exit("Missing dependency: pip install numpy")

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sys.exit("Missing dependency: pip install sounddevice soundfile")

try:
    from evdev import InputDevice, categorize, ecodes, list_devices
except ImportError:
    sys.exit("Missing dependency: pip install evdev  (also needs 'input' group membership)")

try:
    from groq import Groq
except ImportError:
    sys.exit("Missing dependency: pip install groq")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path.home() / ".config" / "freeflow-linux" / "config.toml"

DEFAULT_CONFIG = """\
# freeflow-linux configuration
api_key = ""            # Groq API key (or set GROQ_API_KEY env var)
hotkey = "KEY_RIGHTCTRL"  # Right Ctrl — change to KEY_F9 etc. if preferred
# audio_device = ""    # Leave empty to use system default mic
"""

POST_PROCESSING_SYSTEM_PROMPT = """\
You are a dictation post-processor. You receive raw speech-to-text output and return clean text ready to be typed into an application.

Your job:
- Remove filler words (um, uh, you know, like) unless they carry meaning.
- Fix spelling, grammar, and punctuation errors.
- When the transcript already contains a word that is a close misspelling of a name or term from the context or custom vocabulary, correct the spelling. Never insert names or terms from context that the speaker did not say.
- Preserve the speaker's intent, tone, and meaning exactly.

Output rules:
- Return ONLY the cleaned transcript text, nothing else.
- If the transcription is empty, return exactly: EMPTY
- Do not add words, names, or content that are not in the transcription. The context is only for correcting spelling of words already spoken.
- Do not change the meaning of what was said.

/no_think"""


def load_config() -> dict:
    """Load config from file, creating default if missing. Env var overrides api_key."""
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(DEFAULT_CONFIG)
        print(f"[freeflow] Created default config at {CONFIG_PATH}")
        print(f"[freeflow] Set api_key in config or export GROQ_API_KEY")

    cfg = toml.loads(CONFIG_PATH.read_text())

    # Env var takes priority
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key:
        cfg["api_key"] = env_key

    cfg.setdefault("hotkey", "KEY_RIGHTCTRL")
    cfg.setdefault("audio_device", None)
    cfg.setdefault("api_base_url", "")

    return cfg


# ---------------------------------------------------------------------------
# Session / paste detection
# ---------------------------------------------------------------------------

def get_session_type() -> str:
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session in ("wayland", "x11"):
        return session
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "unknown"


def get_compositor() -> str:
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
    if "gnome" in desktop:
        return "gnome"
    if "kde" in desktop or "plasma" in desktop:
        return "kde"
    return "other"  # sway, hyprland, wlroots-based


def is_terminal_focused() -> bool:
    try:
        win_id = subprocess.run(
            ['xdotool', 'getactivewindow'],
            capture_output=True, text=True
        ).stdout.strip()
        xprop = subprocess.run(
            ['xprop', '-id', win_id, 'WM_CLASS'],
            capture_output=True, text=True
        ).stdout.lower()
        win_name = subprocess.run(
            ['xdotool', 'getactivewindow', 'getwindowname'],
            capture_output=True, text=True
        ).stdout.lower()
        terminals = [
            'xterm', 'alacritty', 'kitty', 'gnome-terminal', 'tilix',
            'wezterm', 'st', 'konsole', 'terminator', 'urxvt', 'rxvt',
            'foot', 'sakura', 'terminology', 'hyper', 'terminal',
            'xfce4-terminal', 'lxterminal', 'mate-terminal',
        ]
        combined = xprop + ' ' + win_name
        return any(t in combined for t in terminals)
    except Exception:
        return False


def paste_text(text: str, session: str):
    """Copy text to clipboard and simulate Ctrl+V in the focused application."""
    encoded = text.encode("utf-8")
    delay = 0.1

    if session == "x11":
        subprocess.run(["xclip", "-selection", "clipboard"], input=encoded, check=True)
        time.sleep(delay)
        if is_terminal_focused():
            subprocess.run(["xdotool", "key", "ctrl+shift+v"])
        else:
            subprocess.run(["xdotool", "key", "ctrl+v"])

    elif session == "wayland":
        # Type directly via ydotool — bypasses clipboard entirely (system services
        # can't reach the user's Wayland clipboard socket reliably).
        subprocess.run(
            ["ydotool", "type", "--key-delay", "1", "--file", "-"],
            input=encoded,
            check=True,
        )

    else:
        # Unknown session: try xclip (may work via XWayland)
        try:
            subprocess.run(["xclip", "-selection", "clipboard"], input=encoded)
        except FileNotFoundError:
            try:
                subprocess.run(["wl-copy", "--", text])
            except FileNotFoundError:
                pass
        print(f"[freeflow] Text copied to clipboard (unknown session — paste manually with Ctrl+V)")


# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------

def play_beep(frequency=880, duration=0.1, volume=0.3):
    """Play a short beep to signal readiness or state change."""
    try:
        t = np.linspace(0, duration, int(16000 * duration), False)
        tone = (np.sin(2 * np.pi * frequency * t) * volume * 32767).astype(np.int16)
        sd.play(tone, samplerate=16000, blocking=True)
    except Exception:
        pass  # never crash on beep failure


def play_error_beep():
    """Loud triple-beep at 1000 Hz — unmissable on the GPD speaker."""
    try:
        sr = 16000
        beep = (np.sin(2 * np.pi * 1000 * np.linspace(0, 0.12, int(sr * 0.12), False)) * 32767).astype(np.int16)
        gap = np.zeros(int(sr * 0.08), dtype=np.int16)
        sd.play(np.concatenate([beep, gap, beep, gap, beep]), samplerate=sr, blocking=True)
    except Exception:
        pass


class AudioRecorder:
    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = "int16"

    # After this many consecutive xrun errors, restart the stream automatically.
    XRUN_RESTART_THRESHOLD = 50
    # Proactively restart the stream every N seconds even if no errors, to prevent
    # the long-running ALSA stream from drifting into an unrecoverable state.
    STREAM_MAX_AGE_SECS = 1800  # 30 minutes

    def __init__(self, device=None):
        self._device = device
        self._frames: list = []
        self._recording = False
        self._stream = None
        self._lock = threading.Lock()
        self._xrun_count = 0
        self._stream_started_at = 0.0
        self._restarting = False

    def _open_stream(self):
        """Create and start a fresh InputStream."""
        def callback(indata, frame_count, time_info, status):
            if status:
                # Any status flag (xrun, etc.) — count it
                self._xrun_count += 1
                if self._xrun_count == self.XRUN_RESTART_THRESHOLD:
                    print(f"[freeflow] WARNING: {self.XRUN_RESTART_THRESHOLD} consecutive "
                          f"audio errors — will restart stream", flush=True)
            else:
                # Good callback — reset counter
                self._xrun_count = 0

            if self._recording:
                self._frames.append(indata.copy())

        stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=self.DTYPE,
            device=self._device,
            callback=callback,
        )
        stream.start()
        self._stream = stream
        self._stream_started_at = time.monotonic()
        self._xrun_count = 0
        print("[freeflow] Audio stream opened", flush=True)

    def _restart_stream(self):
        """Close the current stream and open a fresh one. Safe to call from any thread."""
        with self._lock:
            if self._restarting:
                return
            self._restarting = True

        try:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            self._open_stream()
            print("[freeflow] Audio stream restarted successfully", flush=True)
        finally:
            with self._lock:
                self._restarting = False

    def _check_health(self):
        """Check stream health and restart if needed. Called before each recording."""
        needs_restart = False
        age = time.monotonic() - self._stream_started_at

        if self._xrun_count >= self.XRUN_RESTART_THRESHOLD:
            print(f"[freeflow] Stream unhealthy ({self._xrun_count} xruns) — restarting",
                  flush=True)
            needs_restart = True
        elif age > self.STREAM_MAX_AGE_SECS and not self._recording:
            print(f"[freeflow] Stream age {age:.0f}s exceeds max — proactive restart",
                  flush=True)
            needs_restart = True
        elif self._stream is None or not self._stream.active:
            print("[freeflow] Stream not active — restarting", flush=True)
            needs_restart = True

        if needs_restart:
            self._restart_stream()

    def start_stream(self):
        """Call once at daemon startup — keeps stream warm to eliminate startup latency."""
        self._open_stream()

    def start_recording(self):
        """Called on key_down — zero latency since stream is already open."""
        self._check_health()
        self._frames = []
        self._recording = True

    def stop_recording(self) -> io.BytesIO:
        """Called on key_up — stop collecting and return WAV buffer."""
        self._recording = False

        if not self._frames:
            print("[freeflow] No audio frames captured", flush=True)
            return io.BytesIO()

        audio = np.concatenate(self._frames, axis=0)

        # Validate: reject silence / near-silence (corrupt xrun data reads as zeros)
        rms = float(np.sqrt(np.mean((audio.astype(np.float32) / 32768.0) ** 2)))
        peak = float(np.max(np.abs(audio)) / 32768.0)
        duration_secs = len(audio) / self.SAMPLE_RATE
        print(f"[freeflow] Captured {len(audio)} samples ({duration_secs:.1f}s) — "
              f"rms={rms:.4f} peak={peak:.4f}", flush=True)

        if rms < 0.0005 and peak < 0.005:
            print("[freeflow] Audio is silence/corrupt (rms/peak near zero) — "
                  "restarting stream and skipping", flush=True)
            self._restart_stream()
            return io.BytesIO()

        try:
            sf.write("/tmp/freeflow-last.wav", audio, self.SAMPLE_RATE, subtype="PCM_16")
        except Exception as e:
            print(f"[freeflow] Debug WAV dump failed: {e}", flush=True)

        buf = io.BytesIO()
        sf.write(buf, audio, self.SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buf.seek(0)
        buf.name = "audio.wav"  # Groq SDK may inspect filename for MIME type detection
        return buf


# ---------------------------------------------------------------------------
# Groq integration
# ---------------------------------------------------------------------------

# Fastest Groq model on this machine (~200ms vs 300-400ms for alternatives) AND
# faithful to user intent (doesn't drop content). The /no_think suffix in the
# system prompt disables Qwen3's thinking mode for low-latency replies, but the
# model still emits an empty <think></think> block we strip below.
# Any error -> raw Whisper. See feedback_snappiness-over-resilience memory.
POST_PROCESS_MODEL = "qwen/qwen3-32b"

# Match <think>...</think> blocks (Qwen3 emits these even with /no_think)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# Safety-net log: every transcript is appended here so nothing is ever lost.
HISTORY_LOG = Path("/tmp/freeflow-history.log")


def transcribe(client: Groq, audio_buf: io.BytesIO) -> str:
    """Single Whisper call. Raises on any error — caller handles."""
    audio_buf.seek(0)
    result = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=audio_buf,
    )
    return result.text.strip()


def post_process(client: Groq, transcript: str, context: str = "") -> str:
    """Single fast call to the post-processing model. Raises on any error —
    caller is expected to fall back to the raw transcript immediately."""
    user_message = (
        f"Instructions: Clean up RAW_TRANSCRIPTION and return only the cleaned "
        f"transcript text without surrounding quotes. Return EMPTY if there should be no result.\n\n"
        f'CONTEXT: "{context}"\n\n'
        f'RAW_TRANSCRIPTION: "{transcript}"'
    )

    response = client.chat.completions.create(
        model=POST_PROCESS_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": POST_PROCESSING_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    result = response.choices[0].message.content

    # Qwen3 emits an empty <think></think> block at the start even with /no_think
    result = _THINK_BLOCK_RE.sub("", result, count=1).strip()

    # Strip outer quotes if the LLM wrapped the entire response
    if len(result) >= 2 and result[0] == result[-1] and result[0] in ('"', "'"):
        result = result[1:-1].strip()

    if result == "EMPTY":
        return ""
    return result


def log_history(raw: str, cleaned: str, status: str):
    """Append a transcript to the safety-net log so nothing is ever lost."""
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        HISTORY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_LOG.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] status={status}\n")
            f.write(f"  raw:     {raw!r}\n")
            if cleaned and cleaned != raw:
                f.write(f"  cleaned: {cleaned!r}\n")
            f.write("\n")
    except Exception as e:
        print(f"[freeflow] History log write failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Context gathering (best-effort, X11 only)
# ---------------------------------------------------------------------------

def get_context(session: str) -> str:
    if session != "x11":
        return ""
    try:
        window_id = subprocess.check_output(
            ["xdotool", "getactivewindow"], stderr=subprocess.DEVNULL
        ).decode().strip()
        title = subprocess.check_output(
            ["xdotool", "getwindowname", window_id], stderr=subprocess.DEVNULL
        ).decode().strip()
        return f"Active window: {title}"
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def find_keyboard_devices() -> list:
    """Return all evdev devices that have EV_KEY capability (keyboards)."""
    keyboards = []
    for path in list_devices():
        try:
            dev = InputDevice(path)
            if ecodes.EV_KEY in dev.capabilities():
                keyboards.append(dev)
        except Exception:
            pass
    return keyboards


def resolve_hotkey(hotkey_name: str) -> int:
    """Convert a key name like 'KEY_RIGHTCTRL' to its evdev keycode."""
    try:
        return getattr(ecodes, hotkey_name)
    except AttributeError:
        print(f"[freeflow] Unknown hotkey '{hotkey_name}', falling back to KEY_RIGHTCTRL")
        return ecodes.KEY_RIGHTCTRL


# ---------------------------------------------------------------------------
# Main daemon logic
# ---------------------------------------------------------------------------

class FreeflowDaemon:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        groq_kwargs = {"api_key": cfg["api_key"]}
        if cfg.get("api_base_url"):
            groq_kwargs["base_url"] = cfg["api_base_url"]
        self._client = Groq(**groq_kwargs)
        self._recorder = AudioRecorder(device=cfg.get("audio_device") or None)
        self._hotkey_code = resolve_hotkey(cfg["hotkey"])
        self._session = get_session_type()
        self._recording = False
        self._lock = threading.Lock()
        self._pending_timer: threading.Timer | None = None

    def _activate_recording(self):
        """Called 1s after key_down if the key is still held."""
        with self._lock:
            if self._pending_timer is None:
                return  # cancelled by key_up
            self._pending_timer = None
            self._recording = True
        play_beep(frequency=440, duration=0.08, volume=0.2)  # beep = recording started
        print("[freeflow] Recording... (release key to transcribe)")
        self._recorder.start_recording()

    def on_hotkey_down(self):
        with self._lock:
            if self._recording or self._pending_timer is not None:
                return
            timer = threading.Timer(1.0, self._activate_recording)
            self._pending_timer = timer
        timer.start()

    def on_hotkey_up(self):
        with self._lock:
            if self._pending_timer is not None:
                # Released before 1s — cancel, no beep, no recording
                self._pending_timer.cancel()
                self._pending_timer = None
                return
            if not self._recording:
                return
            self._recording = False

        print("[freeflow] Processing...")
        audio_buf = self._recorder.stop_recording()

        # Empty buffer means no frames or corrupt audio — don't hit the API
        if audio_buf.getbuffer().nbytes == 0:
            print("[freeflow] No usable audio — skipping")
            play_error_beep()
            return

        # Transcribe. If Whisper fails, there's nothing to paste — beep + bail.
        try:
            raw = transcribe(self._client, audio_buf)
        except Exception as e:
            print(f"[freeflow] Transcription failed: {e}", flush=True)
            log_history("", "", f"transcribe_failed: {e}")
            play_error_beep()
            return

        if not raw:
            print("[freeflow] Empty transcription")
            log_history("", "", "empty_transcription")
            return
        print(f"[freeflow] Raw transcript: {raw!r}")

        # Post-process — single shot, no retries. ANY error -> paste raw.
        context = get_context(self._session)
        text_to_paste = raw
        status = "ok"
        try:
            cleaned = post_process(self._client, raw, context)
            if cleaned:
                text_to_paste = cleaned
                print(f"[freeflow] Cleaned: {cleaned!r}")
            else:
                print("[freeflow] Post-processor returned EMPTY — nothing to paste")
                log_history(raw, "", "post_empty")
                return
        except Exception as e:
            print(f"[freeflow] Post-process failed, pasting raw: {e}", flush=True)
            status = f"post_failed: {e}"

        # Paste. History log captures result either way.
        try:
            paste_text(text_to_paste, self._session)
            print("[freeflow] Pasted.")
            log_history(raw, text_to_paste, status)
        except Exception as e:
            print(f"[freeflow] Paste failed: {e}", flush=True)
            log_history(raw, text_to_paste, f"paste_failed: {e}")
            play_error_beep()

    async def _monitor_device(self, dev: InputDevice):
        try:
            async for event in dev.async_read_loop():
                if event.type == ecodes.EV_KEY:
                    e = categorize(event)
                    keycodes = e.keycode if isinstance(e.keycode, list) else [e.keycode]
                    # evdev key names are strings like 'KEY_RIGHTCTRL'
                    hotkey_name = self._cfg["hotkey"]
                    if hotkey_name in keycodes:
                        if e.keystate == e.key_down:
                            # Run blocking handler in thread pool to not block event loop
                            asyncio.get_event_loop().run_in_executor(None, self.on_hotkey_down)
                        elif e.keystate == e.key_up:
                            asyncio.get_event_loop().run_in_executor(None, self.on_hotkey_up)
        except OSError:
            pass  # Device disconnected

    async def run(self, devices: list):
        print(f"[freeflow] Monitoring {len(devices)} keyboard device(s)")
        print(f"[freeflow] Hotkey: {self._cfg['hotkey']}")
        print(f"[freeflow] Session: {self._session}")

        self._recorder.start_stream()
        play_beep(frequency=880, duration=0.1, volume=0.3)  # startup ready beep
        print(f"[freeflow] Ready — hold {self._cfg['hotkey']} to dictate")

        await asyncio.gather(*[self._monitor_device(dev) for dev in devices])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="freeflow-linux voice dictation daemon")
    parser.add_argument("--dry-run", action="store_true", help="Check config/devices/session and exit")
    args = parser.parse_args()

    cfg = load_config()

    api_key = cfg.get("api_key", "").strip()
    print(f"[freeflow] Groq API key: {'set (' + api_key[:8] + '...)' if api_key else 'NOT SET'}")
    print(f"[freeflow] Hotkey: {cfg['hotkey']}")
    print(f"[freeflow] Config: {CONFIG_PATH}")

    # Detect session
    session = get_session_type()
    compositor = get_compositor() if session == "wayland" else "n/a"
    print(f"[freeflow] Session: {session}" + (f" / compositor: {compositor}" if session == "wayland" else ""))

    # Find keyboard devices
    try:
        devices = find_keyboard_devices()
    except PermissionError:
        print("[freeflow] ERROR: Cannot read /dev/input — add yourself to the 'input' group:")
        print("           sudo usermod -aG input $USER  (then log out and back in)")
        sys.exit(1)

    if not devices:
        print("[freeflow] WARNING: No keyboard devices found in /dev/input")
    else:
        print(f"[freeflow] Found {len(devices)} keyboard device(s):")
        for dev in devices:
            print(f"           {dev.path}: {dev.name}")

    if args.dry_run:
        print("[freeflow] Dry-run complete.")
        return

    if not api_key:
        print("[freeflow] ERROR: No Groq API key. Set api_key in config or export GROQ_API_KEY")
        sys.exit(1)

    if not devices:
        print("[freeflow] ERROR: No keyboard devices to monitor. Cannot start.")
        sys.exit(1)

    daemon = FreeflowDaemon(cfg)
    try:
        asyncio.run(daemon.run(devices))
    except KeyboardInterrupt:
        print("\n[freeflow] Stopped.")


if __name__ == "__main__":
    main()
