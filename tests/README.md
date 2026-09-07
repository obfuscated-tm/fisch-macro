# Offline detection test data

## Recording a clip

Record your screen (QuickTime → File → New Screen Recording, or Cmd+Shift+5)
while doing a full reel cycle. Useful clips to capture:

1. `reel_basic.mov`     — a normal fight, fish moving left and right
2. `reel_rod2.mov`      — the same, with a DIFFERENT rod (different bar/fish colors)
3. `reel_ui.mov`        — a fight with chat / UI overlays on screen
4. `physics_hold.mov`   — hold the mouse from the far LEFT and let the bar run to
                          the far RIGHT without releasing. Used to measure the
                          game's rightward acceleration.
5. `physics_release.mov`— the reverse: from the far RIGHT, release and let it fall
                          all the way LEFT. Measures leftward acceleration.

Drop the files in `tests/clips/`. Then:

    python3 scripts/extract_frames.py tests/clips/reel_basic.mov

Frames land in `tests/frames/<clipname>/`.

Individual screenshots can also just be dropped straight into `tests/frames/`.

## Running detection over them

    python3 scripts/test_detection.py

Writes annotated images to `tests/out/` and a per-frame CSV of
bar_left / bar_right / fish_x / on_target.
