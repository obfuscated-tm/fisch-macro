#!/usr/bin/env python3
"""Extract frames from a screen recording into tests/frames/<clipname>/.

Usage:
    python3 scripts/extract_frames.py tests/clips/reel_basic.mov [--fps 15]

Frames are written as zero-padded PNGs so they sort chronologically, which
matters for the tracking/physics checks in test_detection.py.
"""

import argparse
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=pathlib.Path)
    ap.add_argument("--fps", type=float, default=15.0,
                    help="frames per second to sample (default 15)")
    ap.add_argument("--start", type=float, default=None, help="start seconds")
    ap.add_argument("--duration", type=float, default=None, help="seconds to take")
    ap.add_argument("--width", type=int, default=1512,
                    help="scale frames to this width (0 = native). Retina captures are "
                         "~3024px wide and produce gigabytes of PNG at native size.")
    args = ap.parse_args()

    if not args.clip.exists():
        print(f"no such clip: {args.clip}", file=sys.stderr)
        return 1
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH", file=sys.stderr)
        return 1

    out_dir = ROOT / "tests" / "frames" / args.clip.stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if args.start is not None:
        cmd += ["-ss", str(args.start)]
    cmd += ["-i", str(args.clip)]
    if args.duration is not None:
        cmd += ["-t", str(args.duration)]
    vf = f"fps={args.fps}"
    if args.width:
        vf += f",scale={args.width}:-2:flags=lanczos"
    cmd += ["-vf", vf, str(out_dir / "f%05d.png")]

    subprocess.run(cmd, check=True)

    frames = sorted(out_dir.glob("*.png"))
    size = sum(f.stat().st_size for f in frames) / 1e6
    print(f"{len(frames)} frames → {out_dir}  "
          f"({args.fps} fps, width {args.width or 'native'}, {size:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
