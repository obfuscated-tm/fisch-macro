# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the Fisch macro.

Build with:  pyinstaller fisch-macro.spec --noconfirm

Produces a one-directory build (not one-file): a one-file build re-extracts
~200 MB to a temp directory on every launch, which is slow everywhere and
worse on a slow disk.

On Windows two executables are produced from one shared collection — a
windowed one for the GUI, and a console one, because a windowed Windows binary
has no stdout and `--headless` would print into the void. macOS and Linux need
only one; the .app's inner binary still writes to the terminal when run from
one directly.
"""

import os
import sys
import tempfile

sys.path.insert(0, SPECPATH)

from src.paths import APP_NAME  # noqa: E402
from src.version import VERSION_FILE, resolve_version  # noqa: E402

BUNDLE_ID = "com.obfuscated-tm.fischmacro"

# From the VERSION file, or from FISCH_MACRO_VERSION when CI is building a
# release of a tag the working tree does not know about. Never a literal here:
# a second copy of the number is a copy that goes stale.
VERSION = resolve_version()

IS_WINDOWS = sys.platform.startswith("win")
IS_MACOS = sys.platform == "darwin"

# Read-only defaults shipped with the app. src/paths.py copies these into the
# per-user data directory on first run rather than writing inside the bundle.
datas = [
    ("profiles", "profiles"),
    ("settings.json", "."),
]

# The VERSION file on disk holds the development version; a release is named by
# its tag, which the checkout has no way to know. Write the resolved version to
# a scratch file and ship that, so `--version` from a downloaded build reports
# the release it came from rather than whatever the tree happened to say.
_version_dir = tempfile.mkdtemp(prefix="fisch-macro-version-")
with open(os.path.join(_version_dir, VERSION_FILE), "w", encoding="utf-8") as f:
    f.write(VERSION + "\n")
datas.append((os.path.join(_version_dir, VERSION_FILE), "."))

# These backends are selected at runtime by string, so static analysis does not
# see them and they would be left out of the bundle.
hiddenimports = [
    "mss.darwin",
    "mss.linux",
    "mss.windows",
]

if IS_MACOS:
    hiddenimports += [
        "pynput.keyboard._darwin",
        "pynput.mouse._darwin",
        "Quartz",
        "ApplicationServices",
    ]
elif IS_WINDOWS:
    hiddenimports += [
        "pynput.keyboard._win32",
        "pynput.mouse._win32",
    ]
else:
    hiddenimports += [
        "pynput.keyboard._xorg",
        "pynput.mouse._xorg",
        "Xlib",
        "Xlib.display",
    ]

# The scientific stack NumPy is often installed alongside, none of which this
# project imports. `unittest` is deliberately NOT excluded: third-party
# packages import unittest.mock at runtime, and it saves ~1 MB out of 160.
#
# Swapping opencv-python for opencv-python-headless is not worth it either —
# measured on macOS arm64 it produced a byte-for-byte comparable bundle.
excludes = [
    "matplotlib",
    "scipy",
    "pandas",
    "IPython",
    "notebook",
    "pytest",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

executables = []

gui_exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-packed binaries trip antivirus heuristics far harder
    console=not (IS_WINDOWS or IS_MACOS),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
executables.append(gui_exe)

if IS_WINDOWS:
    executables.append(
        EXE(
            pyz,
            a.scripts,
            [],
            exclude_binaries=True,
            name=f"{APP_NAME}-cli",
            debug=False,
            bootloader_ignore_signals=False,
            strip=False,
            upx=False,
            console=True,
            disable_windowed_traceback=False,
        )
    )

coll = COLLECT(
    *executables,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)

if IS_MACOS:
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=None,
        # macOS ties Accessibility and Screen Recording grants to the bundle
        # identifier. Keep this stable across releases or every build asks for
        # both permissions again.
        bundle_identifier=BUNDLE_ID,
        version=VERSION,
        info_plist={
            "CFBundleName": "Fisch Macro",
            "CFBundleDisplayName": "Fisch Macro",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "LSMinimumSystemVersion": "11.0",
            "LSApplicationCategoryType": "public.app-category.utilities",
            # Without this the app runs through the 1x scaler and the logical
            # screen size pyautogui reports stops matching the real display,
            # which corrupts the ROI scale factor.
            "NSHighResolutionCapable": True,
            "NSAppleEventsUsageDescription":
                "Fisch Macro controls the Roblox window to automate fishing.",
        },
    )
