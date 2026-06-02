## [2026-06-01 20:13:42] Task 1+2: Cast transition guardrails + hysteresis
- REELING_GUARD_SECONDS=1.5 blocks completion/failure exits early in reeling
- success_confirm_count requires 3 consecutive frames at >0.98 progress
- fail_confirm_count requires 5 consecutive frames at <0.01+bite
- bar_gone_confirm_count requires 15 consecutive frames of !bar_active
- All counters reset to 0 on reeling entry

## [2026-06-01 20:40:01] Task 3: Control-loop damping
- Action dwell: min_action_dwell=0.08s prevents switching actions faster than 80ms
- Stable hysteresis: once in rapid_click, deadzone expands by 1.5x before leaving
- PD-score deadband: 0.02 deadband prevents tiny oscillations from flipping direction
- All branches check cooldown before switching action types
- Action dwell state resets on reeling entry so old-session cooldown doesn't leak

## [2026-06-01 21:10:00] Task 4: Config-backed stabilization defaults
- Settings now carries reeling guard, confirmation-frame counts, dwell, hysteresis, and PD deadband defaults
- Missing JSON keys fall back to dataclass defaults, preserving backward compatibility
- Macro reeling logic now reads these values from `settings` instead of hardcoded literals

## [2026-06-01 21:25:00] Task 5: Smoke verification harness
- Smoke harness uses fake detector/controller objects and direct `_do_reeling()` calls
- Harness verifies premature finish blocking, flicker hysteresis, and stable-target anti-thrash behavior
- Evidence files are written under `.sisyphus/evidence/` for each scenario plus an overall pass file
- Added separate finish-capture debounce counter (_finish_confirm_count) to prevent VFX flashes from triggering premature completion

## [2026-06-01 21:40:00] GUI fix: readable settings panel + slider stability
- Settings tab now uses a light card surface with black text so labels remain readable
- `_save_settings()` no longer rebinds the killswitch listener on every slider change
- Hotkey rebinding stays in the dedicated rebind flow only

## [2026-06-01 22:00:00] Task 6: GUI readability + slider crash fix
- `_save_settings()` previously called `setup_killswitch()` on every slider trace write, tearing
  down and re-registering the global hotkey listener on every slider drag frame → crash.
  Fix: removed `setup_killswitch()` and `_refresh_killswitch_labels()` from `_save_settings()`.
  The rebind flow (`_rebind_killswitch()`) still calls `setup_killswitch()` and label refresh.
- Settings tab had poor contrast (white text) because some ttk Frames/Labels lacked explicit
  backgrounds, potentially defaulting to native light surfaces on macOS.
  Fix: added `Settings.TFrame`, `Settings.TLabel`, `SettingsDim.TLabel`, `SettingsStat.TLabel`,
  `Settings.TScale`, `Settings.TCheckbutton` styles with a deliberate light card surface
  (`#eef1f5` background, `#000000` black text) — rest of dark theme untouched.
- New colour constants: `settings_bg`, `settings_text`, `settings_dim` in COLORS dict.
- Added bar-velocity compensation term to PD control: velocity_compensation = -self._bar_velocity * k_vc where k_vc=0.15. This creates a feed-forward bias opposite to bar motion (left drift -> right bias) to prevent overshoot.

## [2026-06-01 22:20:00] Final verification
- `python3 -m py_compile src/gui.py src/macro.py src/config.py scripts/smoke_fisch_macro.py` passed with no syntax errors.
- `python3 scripts/smoke_fisch_macro.py` passed all scenarios and wrote evidence files under `.sisyphus/evidence/`.
- Final checklist items in the plan were marked complete after verification.

## [2026-06-01 22:35:00] GUI crash fix: Settings.TScale layout registration
- `ttk.Scale(style="Settings.TScale")` needs an explicit `Horizontal.Settings.TScale` layout on Tk/clam; otherwise instantiation raises `TclError: Layout Horizontal.Settings.TScale not found`.
- Fix: copied the existing `Horizontal.TScale` and `Vertical.TScale` layouts onto the settings-prefixed style names inside `_configure_styles()`.
- Verified by direct widget instantiation plus `python3 -m py_compile src/gui.py src/macro.py src/config.py scripts/smoke_fisch_macro.py` and `python3 scripts/smoke_fisch_macro.py`.
