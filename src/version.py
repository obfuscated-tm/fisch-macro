"""The version, resolved from one place rather than written down in several.

`VERSION` at the project root is that place. The spec file reads it when it
packages a build, CI reads it to decide what a release is called, and this
module reads it so a binary someone downloaded can say what it is.

A release is not built from a working tree whose VERSION file has been bumped —
the tag is chosen when the workflow runs — so CI passes the version it is
releasing in FISCH_MACRO_VERSION and the spec bakes that into the bundle. At
runtime the environment variable is long gone, which is why the packaged copy
of the file, not the environment, is what a frozen build reads.
"""

import os

from src.paths import bundle_dir

#: Name of the file holding the version, at the project root and, for a
#: packaged build, at the root of the bundle.
VERSION_FILE = "VERSION"

#: Reported when the file is missing, which means neither a checkout nor a
#: build produced by the spec — someone's hand-assembled tree. Better to say so
#: than to claim a version number that was never released.
UNKNOWN_VERSION = "0.0.0+unknown"


def resolve_version() -> str:
    """Return the version this copy of the macro should report.

    The environment variable is a build-time override, set by CI so that a
    release carries the version of the tag it is being published under. Nothing
    sets it at runtime.
    """
    override = os.environ.get("FISCH_MACRO_VERSION", "").strip()
    if override:
        return override

    try:
        with open(os.path.join(bundle_dir(), VERSION_FILE), encoding="utf-8") as f:
            return f.read().strip() or UNKNOWN_VERSION
    except OSError:
        return UNKNOWN_VERSION


__version__ = resolve_version()
