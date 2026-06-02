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
