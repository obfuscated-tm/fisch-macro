"""A build has to be able to say which release it is.

The version used to be a literal in the spec file, which meant a release could
be tagged v1.2.0 and ship a bundle still calling itself 1.0.0. It now comes
from the VERSION file, and CI overrides it for the tag it is releasing, so
these are the two paths that have to keep working — plus the case where neither
is available, which must not invent a number.

`--version` itself is deliberately not exercised here: importing main pulls in
src.controller, which imports pyautogui at module scope, and this suite runs on
a headless runner with no display. The build jobs check the flag on all four
platforms against the version they were built with, which tests the whole chain
rather than just the argparse wiring.
"""

import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import version  # noqa: E402


@pytest.fixture(autouse=True)
def no_inherited_override(monkeypatch):
    """CI sets this for real when it builds; it must not leak into the tests."""
    monkeypatch.delenv("FISCH_MACRO_VERSION", raising=False)


def test_a_checkout_reports_the_version_file():
    declared = (ROOT / "VERSION").read_text(encoding="utf-8").strip()

    assert version.resolve_version() == declared


def test_the_version_file_exists_and_is_a_release_number():
    declared = (ROOT / "VERSION").read_text(encoding="utf-8").strip()

    # CI refuses to release a tag that is not vMAJOR.MINOR.PATCH, so a VERSION
    # file that cannot be tagged is a file that will surprise someone later.
    assert declared.count(".") == 2
    assert all(part.isdigit() for part in declared.split("."))


def test_ci_can_override_the_version_it_is_releasing(monkeypatch):
    monkeypatch.setenv("FISCH_MACRO_VERSION", "9.8.7")

    assert version.resolve_version() == "9.8.7"


def test_a_blank_override_falls_back_rather_than_reporting_nothing(monkeypatch):
    monkeypatch.setenv("FISCH_MACRO_VERSION", "   ")

    assert version.resolve_version() == (ROOT / "VERSION").read_text().strip()


def test_a_frozen_build_reads_the_version_shipped_inside_it(monkeypatch, tmp_path):
    """The spec writes the resolved version into the bundle; read that."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / version.VERSION_FILE).write_text("2.5.0\n", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)

    assert version.resolve_version() == "2.5.0"


def test_a_tree_with_no_version_file_says_so_instead_of_guessing(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert version.resolve_version() == version.UNKNOWN_VERSION


def test_an_empty_version_file_is_not_reported_as_a_version(monkeypatch, tmp_path):
    (tmp_path / version.VERSION_FILE).write_text("\n", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert version.resolve_version() == version.UNKNOWN_VERSION
