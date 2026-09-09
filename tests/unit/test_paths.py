"""A frozen build must not write its settings somewhere that gets deleted.

ConfigManager takes a base directory and writes settings.json and profiles/
straight into it. That is correct from a checkout and wrong in every packaged
form: a one-file build unpacks to a temp directory that is removed on exit, and
writing inside a macOS .app breaks its code signature. Either way a user would
calibrate a rod, quit, and find the profile gone.
"""

import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import paths  # noqa: E402


@pytest.fixture
def frozen(monkeypatch, tmp_path):
    """Pose as a PyInstaller one-file build with a private HOME."""
    bundle = tmp_path / "bundle"
    (bundle / "profiles").mkdir(parents=True)
    (bundle / "profiles" / "Daybreaker.json").write_text('{"name": "Daybreaker"}')
    (bundle / "profiles" / "default.json").write_text('{"name": "default"}')
    (bundle / "settings.json").write_text('{"active_profile": "Daybreaker"}')

    home = tmp_path / "home"
    home.mkdir()

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("APPDATA", str(home / "AppData"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    return bundle


# -- checkout behaviour is unchanged ------------------------------------


def test_a_checkout_still_uses_the_project_root(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)

    assert not paths.is_frozen()
    assert paths.resolve_base_dir() == str(ROOT)


def test_the_checkout_root_is_where_settings_json_actually_lives():
    assert (ROOT / "settings.json").is_file()
    assert (ROOT / "profiles").is_dir()


# -- per-platform user data locations ------------------------------------


@pytest.mark.parametrize(
    "platform, expected_tail",
    [
        ("darwin", os.path.join("Library", "Application Support", "FischMacro")),
        ("win32", "FischMacro"),
        ("linux", "fisch-macro"),
    ],
)
def test_user_data_lands_in_the_platform_location(
    monkeypatch, tmp_path, platform, expected_tail
):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))

    assert paths.user_data_dir().endswith(expected_tail)


def test_linux_falls_back_to_dot_config_without_xdg(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert paths.user_data_dir() == str(tmp_path / ".config" / "fisch-macro")


# -- seeding -------------------------------------------------------------


def test_a_frozen_build_writes_outside_the_bundle(frozen):
    base = paths.resolve_base_dir()

    assert base != str(frozen), "settings must not be written into the bundle"
    assert os.path.isdir(base)


def test_first_run_seeds_the_bundled_profiles(frozen):
    base = pathlib.Path(paths.resolve_base_dir())

    assert (base / "settings.json").is_file()
    assert {p.name for p in (base / "profiles").glob("*.json")} == {
        "Daybreaker.json",
        "default.json",
    }


def test_seeding_never_clobbers_a_calibrated_profile(frozen):
    base = pathlib.Path(paths.resolve_base_dir())
    (base / "profiles" / "Daybreaker.json").write_text('{"name": "MINE"}')
    (base / "settings.json").write_text('{"active_profile": "MINE"}')

    paths.resolve_base_dir()  # second launch

    assert "MINE" in (base / "profiles" / "Daybreaker.json").read_text()
    assert "MINE" in (base / "settings.json").read_text()


def test_a_bundle_without_seed_data_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "empty"), raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))

    assert os.path.isdir(paths.resolve_base_dir())


# -- logs ----------------------------------------------------------------


def test_log_dir_is_created_under_the_base_dir(tmp_path):
    target = paths.log_dir(str(tmp_path))

    assert target == str(tmp_path / "logs")
    assert os.path.isdir(target)


def test_log_dir_does_not_depend_on_the_working_directory(frozen, monkeypatch):
    """A .app opened from Finder runs with cwd "/", where "logs" is unwritable."""
    monkeypatch.chdir("/")

    target = paths.log_dir()

    assert os.path.isdir(target)
    assert not target.startswith("/logs")
