# Fisch Macro

An automated fishing macro for Fisch (Roblox). Watches the reel minigame with
OpenCV and drives the bar with a closed-loop controller.

## Platform support

| Platform | Window tracking | Notes |
|---|---|---|
| macOS | Quartz | Needs Accessibility + Screen Recording |
| Windows | Win32 / DWM | Antivirus will flag an unsigned build |
| Linux (X11) | python-xlib, `xdotool` fallback | Works with [Sober](https://sober.vinegarhq.org/) |
| Linux (Wayland) | ✗ | Not possible — see below |

**Wayland does not work and cannot be made to.** Neither `mss` (screen capture)
nor `pyautogui` (input injection) can reach another application's surface under
a Wayland compositor, which is the default on Raspberry Pi OS. Switch to X11:

```bash
sudo raspi-config
```

Advanced Options → Wayland → X11, then log back in. The macro detects a Wayland
session and says so rather than failing with empty captures.

## Install

```bash
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11+. Requires `tkinter` for the GUI — bundled with python.org and
Homebrew builds, `sudo apt install python3-tk` on Debian/Ubuntu/Pi OS.

## Run

```bash
python main.py
```

Press **F6** to start and stop (configurable in the Settings tab). Roblox must
be in **windowed** mode, not fullscreen — every ROI is measured relative to the
window, so the macro needs to find its rectangle.

### Headless

For a display too small for the control panel — the panel is 372×572 with a
340×500 minimum, so it does not fit an 800×480 screen at all:

```bash
python main.py --headless
```

Same engine, same F6 hotkey, status to the console instead of widgets:

```
[window] 800x480 at (0, 33), scale=1.0x
[macro] started
[state] Reeling
[stats] caught=12 failed=3 casts=15 rate=80% uptime=42m10s (17/hr)
```

| Flag | Effect |
|---|---|
| `--headless` | Run without the GUI |
| `--no-autostart` | Wait for the hotkey instead of starting at once |
| `--profile NAME` | Switch color profile before starting |
| `--list-profiles` | Print available profiles and exit |

`tkinter` is imported lazily, so headless runs on a machine without it.

## macOS permissions

System Settings → Privacy & Security, grant both to your terminal (or to
`FischMacro.app`):

- **Accessibility** — mouse clicks and the global hotkey
- **Screen Recording** — capturing the reel bar

Without them the macro starts, sees nothing, and clicks nothing.

## Calibration

Profiles live in `profiles/` and store HSV color ranges plus three ROIs — the
reel bar, the progress bar, and the shake area. **ROIs are stored as fractions
of the game window**, not screen pixels, so a profile calibrated on one display
carries over to another.

Two things to know:

- **Calibrate with the game window at the size you will run it.** Roblox mixes
  proportional and fixed-pixel offsets, so the bar's *fractional* position
  shifts between 1080p and 800×480. Size the window to the target resolution
  before calibrating, rather than calibrating fullscreen on a big monitor.
- **The progress ROI gets thin on small displays.** It spans roughly 1.4% of
  window height — about 6 pixels at 480p versus 15 at 1080p. That is near the
  floor for reliable detection and is usually what needs the most tuning.

### Raspberry Pi workflow

The Pi has no room for the calibration UI on a 4" screen, so:

1. Plug in an external monitor, size the game window to 800×480, run the GUI
   and calibrate there.
2. Unplug, and run `python main.py --headless` on the small display.

Sober runs Roblox's Android client, which lays the fishing UI out differently
from the desktop client — expect to recalibrate rather than reuse a profile
from a Mac or PC.

## Building a standalone app

```bash
pip install pyinstaller
pyinstaller fisch-macro.spec --noconfirm
```

Output lands in `dist/` — `FischMacro.app` on macOS, a `FischMacro/` folder
elsewhere. On Windows that folder holds two executables: `FischMacro.exe` for
the GUI and `FischMacro-cli.exe` for `--headless`, because a windowed Windows
binary has no stdout.

Roughly 160 MB per platform, most of it OpenCV. (`opencv-python-headless` does
not help — measured, it produces a comparable bundle.)

Builds cannot be cross-compiled; each OS builds its own.
[`.github/workflows/build.yml`](.github/workflows/build.yml) does all of them
on tag push and attaches the artifacts to a release.

A packaged build keeps settings and profiles outside the bundle — in
`~/Library/Application Support/FischMacro`, `%APPDATA%\FischMacro`, or
`~/.config/fisch-macro` — seeded from the bundled defaults on first run.

### Signing

Neither build is signed by a certificate authority, so:

- **macOS** ad-hoc signs the bundle, which gives it a stable identity so
  permission grants survive relaunches, but Gatekeeper still warns. Right-click
  → Open the first time. Proper notarization needs an Apple Developer account.
- **Windows** SmartScreen shows "unrecognized app" for any unsigned binary, and
  Defender may flag a program that injects input and captures the screen. This
  is inherent to what the macro does, not a build defect.

## Tests

```bash
pytest tests/ -q
```
