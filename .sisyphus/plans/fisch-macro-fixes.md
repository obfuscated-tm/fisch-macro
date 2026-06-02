# Fisch Macro Stability Fixes

## TL;DR

> Stabilize the reeling loop so finish/completion signals do not fire at cast start, reduce over-aggressive stop behavior with hysteresis, and damp the positive feedback loop in bar control.
>
> Deliverables:
> - Reeling state guards with minimum-age / consecutive-confirmation logic
> - Less hyperactive stop/completion handling
> - Dampened control loop with rate limits and hysteresis
> - Small smoke-verification setup for the new behavior

**Estimated Effort**: Short
**Parallel Execution**: YES - 2 waves
**Critical Path**: Task 1 → Task 3 → Task 5

---

## Context

### Original Request
User reported the macro still has issues in Fisch: the thing that means a cast is finished seems to fire at the start, cast stopping is too hyperactive, and there is a large positive feedback issue that needs mitigation.

### Interview Summary
**Key Discussions**:
- The repo is Python and the main behavior lives in `src/macro.py`.
- Detection is in `src/detector.py`; controller safety/debounce already exists in `src/controller.py`.
- There is no existing automated test suite.
- User wants a small verification/smoke-test setup included.

**Research Findings**:
- `MacroEngine._do_reeling()` contains immediate completion/failure exits based on `progress` and `bar_active`.
- The positive feedback issue is likely in the PD/control branch in `_do_reeling()` where rapid click / hold / release decisions react too quickly.
- `detect_all()` may still report progress or active state during transitions, so the macro needs local hysteresis rather than trusting a single frame.

### Metis Review
**Identified Gaps** (addressed):
- Timing thresholds were not explicit: plan adds minimum-age / consecutive-frame confirmation.
- Scope creep risk: plan limits changes to targeted stabilization, not a detector redesign.
- Missing acceptance criteria: plan adds agent-verifiable smoke checks for the two failure modes plus a stable reeling run.

---

## Work Objectives

### Core Objective
Make the macro stop misfiring on cast-start transitions and prevent reeling from amplifying its own corrections.

### Concrete Deliverables
- Stabilized completion/failure detection in `src/macro.py`
- Dampened control behavior in `src/macro.py`
- Minimal config hooks for the new guard timings/gains, if needed
- A small smoke-verification path

### Definition of Done
- [ ] A cast-start transition no longer immediately trips completion/failure logic in smoke verification.
- [ ] Reeling no longer toggles rapidly between hold/release in the positive-feedback scenario.
- [ ] Smoke checks pass with the new guard timing and control damping.

### Must Have
- Fix the premature completion signal.
- Reduce hyperactive stopping.
- Mitigate the positive feedback loop.

### Must NOT Have (Guardrails)
- No full rewrite of the detector pipeline.
- No new runtime dependencies.
- No major GUI redesign.
- No broad refactor outside the targeted reeling / transition logic.

---

## Verification Strategy (MANDATORY)

> ZERO HUMAN INTERVENTION for verification. The executor must run checks and capture evidence.

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: Small smoke-verification setup
- **Framework**: lightweight Python smoke checks / runtime assertions

### QA Policy
Each task includes an agent-executed scenario with concrete pass/fail criteria.

---

## Execution Strategy

### Parallel Execution Waves

Wave 1 (foundation + stabilization guards):
├── Task 1: Cast transition guard / debounce
├── Task 2: Completion/failure confirmation hysteresis
└── Task 3: Control-loop damping and rate limiting

Wave 2 (integration + verification):
├── Task 4: Config defaults and parameter wiring
└── Task 5: Smoke verification harness and runtime checks

Dependency note: Task 3 depends on the transition guards from Tasks 1-2; Task 5 depends on all prior changes.

---

## TODOs

- [x] 1. Add cast-transition guardrails

  **What to do**:
  - Inspect `src/macro.py` reeling entry/exit logic and add a minimum reeling age before completion/failure can trigger.
  - Track short-lived state such as “frames since reeling started” or “time since bite confirmed”.
  - Make the start-of-cast / start-of-reel window ignore ambiguous frames instead of finishing immediately.

  **Must NOT do**:
  - Do not change the detector’s public contract.
  - Do not add new macro states.

  **Recommended Agent Profile**:
  > Category: `quick`
    - Reason: localized behavioral fix in one module.
  > Skills: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 3, Task 5
  - **Blocked By**: None

  **References**:
  - `src/macro.py:364-433` - current reeling exit logic and the likely source of premature completion.
  - `src/macro.py:301-355` - entry into waiting/shaking/reeling and current transition setup.

  **Why These References Matter**:
  - They show where the macro currently trusts a single frame too early.

  **Acceptance Criteria**:
  - [ ] Reeling does not complete/fail on the first ambiguous frame after transition.
  - [ ] Guard logic is observable in logs or smoke output.

  **QA Scenarios**:
  ```
  Scenario: Early-transition false finish is blocked
    Tool: Python smoke check
    Preconditions: mocked detector reports transient progress/bar flicker immediately after reeling start
    Steps:
      1. Start reeling with a fresh session timestamp.
      2. Feed one or two ambiguous detections.
      3. Assert the engine remains in reeling and does not emit completion/failure.
    Expected Result: no completion/failure before the minimum-age gate is satisfied.
    Evidence: .sisyphus/evidence/task-1-early-transition-blocked.txt

  Scenario: Normal reeling still proceeds after the gate
    Tool: Python smoke check
    Preconditions: stable reeling detections after the guard window
    Steps:
      1. Advance time beyond the guard window.
      2. Feed a valid stable detection sequence.
      3. Assert the state can now resolve normally.
    Expected Result: the guard does not permanently block legitimate completion.
    Evidence: .sisyphus/evidence/task-1-normal-reeling-after-gate.txt
  ```

- [x] 2. Add completion/failure hysteresis

  **What to do**:
  - Replace single-frame completion/failure exits with consecutive-confirmation logic.
  - Require repeated evidence for zero-progress disappearance or full-progress success before changing state.
  - Favor a short grace period on flaky detections to avoid hyperactive stopping.

  **Must NOT do**:
  - Do not widen thresholds so much that real catches get ignored.
  - Do not add expensive image processing.

  **Recommended Agent Profile**:
  > Category: `quick`
    - Reason: small logic change in the same state machine.
  > Skills: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 3, Task 5
  - **Blocked By**: None

  **References**:
  - `src/macro.py:382-433` - immediate success/failure branches to harden.
  - `src/detector.py:524-580` - how active/bite/progress are produced and why a single frame can be noisy.

  **Why These References Matter**:
  - They define which inputs should be confirmed over time instead of trusted instantly.

  **Acceptance Criteria**:
  - [ ] No immediate completion/failure from a single flicker.
  - [ ] Legitimate catch/fail sequences still resolve after confirmation.

  **QA Scenarios**:
  ```
  Scenario: Flicker does not stop the cast too early
    Tool: Python smoke check
    Preconditions: alternating progress/bar-active frames around the threshold
    Steps:
      1. Feed alternating active/inactive detections.
      2. Observe the confirmation counter.
      3. Assert no state change until the required consecutive confirmations are met.
    Expected Result: hysteresis prevents hyperactive stop behavior.
    Evidence: .sisyphus/evidence/task-2-flicker-hysteresis.txt

  Scenario: True completion still wins
    Tool: Python smoke check
    Preconditions: sustained full-progress detections
    Steps:
      1. Feed repeated completion-level readings.
      2. Assert the engine transitions exactly once.
    Expected Result: completion still works, but only after confirmation.
    Evidence: .sisyphus/evidence/task-2-true-completion.txt
  ```

- [x] 3. Damp the positive feedback loop

  **What to do**:
  - Reduce oscillation in `_do_reeling()` by smoothing the position/error inputs or lowering reaction gain.
  - Add action-rate limits / minimum dwell times so hold/release cannot thrash every tick.
  - Keep the existing control approach, but add hysteresis around stable / panic / hover decisions.

  **Must NOT do**:
  - Do not introduce a new control architecture.
  - Do not remove emergency panic behavior entirely.

  **Recommended Agent Profile**:
  > Category: `unspecified-high`
    - Reason: this is the most behavior-sensitive part and may require careful tuning.
  > Skills: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 5
  - **Blocked By**: Tasks 1-2

  **References**:
  - `src/macro.py:446-527` - the PD-style decision path and rapid-click behavior.
  - `src/macro.py:457-492` - the current gain / deadzone setup that likely amplifies oscillation.

  **Why These References Matter**:
  - They show where to insert damping without changing the overall loop shape.

  **Acceptance Criteria**:
  - [ ] Hold/release decisions no longer alternate aggressively on stable input.
  - [ ] The macro still responds to real fish movement.

  **QA Scenarios**:
  ```
  Scenario: Stable target does not cause thrash
    Tool: Python smoke check
    Preconditions: fish/bar positions remain near center for several ticks
    Steps:
      1. Feed near-zero error inputs for many iterations.
      2. Capture controller actions.
      3. Assert actions stay sparse and do not alternate rapidly.
    Expected Result: no positive-feedback oscillation.
    Evidence: .sisyphus/evidence/task-3-stable-target-no-thrash.txt

  Scenario: Large deviation still triggers corrective action
    Tool: Python smoke check
    Preconditions: fish position shifts clearly beyond the deadzone
    Steps:
      1. Feed a sustained large error.
      2. Assert the controller responds with the intended correction.
    Expected Result: damping does not make the macro unresponsive.
    Evidence: .sisyphus/evidence/task-3-large-error-corrects.txt
  ```

- [x] 4. Wire small config knobs and defaults

  **What to do**:
  - Add/adjust small timing and hysteresis values in `src/config.py` if the implementation needs tunable defaults.
  - Keep defaults conservative so current users do not need to retune immediately.
  - Preserve compatibility with existing JSON configs.

  **Must NOT do**:
  - Do not break loading of old settings/profiles.
  - Do not expose a large new configuration surface.

  **Recommended Agent Profile**:
  > Category: `quick`
    - Reason: limited schema/default work.
  > Skills: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential after Tasks 1-3
  - **Blocks**: Task 5
  - **Blocked By**: Tasks 1-3

  **References**:
  - `src/config.py:56-82` - settings and profile schema.
  - `src/config.py:103-175` - JSON load/save compatibility patterns.

  **Why These References Matter**:
  - They define the safe place to add defaults without breaking existing saves.

  **Acceptance Criteria**:
  - [ ] New defaults load without errors.
  - [ ] Existing settings JSON still loads.

  **QA Scenarios**:
  ```
  Scenario: Existing config still loads
    Tool: Python smoke check
    Preconditions: old settings JSON shape with existing fields only
    Steps:
      1. Load the config file.
      2. Assert the returned object has valid defaults for new fields.
    Expected Result: backward compatibility preserved.
    Evidence: .sisyphus/evidence/task-4-backward-compatible-config.txt

  Scenario: New defaults round-trip cleanly
    Tool: Python smoke check
    Preconditions: config object with the new tuning values
    Steps:
      1. Save the settings/profile.
      2. Reload it.
      3. Assert the values survive round-trip serialization.
    Expected Result: no config regressions.
    Evidence: .sisyphus/evidence/task-4-config-roundtrip.txt
  ```

- [ ] 5. Add smoke verification harness

  **What to do**:
  - Add a lightweight Python smoke check that exercises the three bug scenarios with mocked detector/controller behavior.
  - Verify: premature finish is blocked, stop hysteresis works, and feedback damping avoids thrash.
  - Capture concise output/evidence for each scenario.

  **Must NOT do**:
  - Do not create a heavyweight test framework.
  - Do not require manual GUI interaction for the core checks.

  **Recommended Agent Profile**:
  > Category: `quick`
    - Reason: simple verification harness plus runtime assertions.
  > Skills: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Final integration
  - **Blocks**: None
  - **Blocked By**: Tasks 1-4

  **References**:
  - `src/macro.py` and `src/detector.py` - behavior under test.
  - `src/controller.py` - action semantics to observe in the smoke harness.

  **Why These References Matter**:
  - The harness needs to assert the exact state/action sequences that the bug reports describe.

  **Acceptance Criteria**:
  - [ ] Smoke check reports pass for all three scenarios.
  - [ ] Evidence files are produced for each scenario.

  **QA Scenarios**:
  ```
  Scenario: Premature finish, hysteresis, and feedback all pass
    Tool: Python smoke check
    Preconditions: all implementation tasks merged locally
    Steps:
      1. Run the smoke harness.
      2. Verify each scenario prints pass/fail clearly.
      3. Verify evidence output paths are created.
    Expected Result: all smoke scenarios pass.
    Evidence: .sisyphus/evidence/task-5-smoke-overall.txt

  Scenario: A forced bad input fails clearly
    Tool: Python smoke check
    Preconditions: harness can inject a failing mocked detection case
    Steps:
      1. Run the harness with an intentionally bad case.
      2. Assert it fails with a clear message and non-zero exit.
    Expected Result: the smoke harness catches regressions.
    Evidence: .sisyphus/evidence/task-5-smoke-failure.txt
  ```

---

## Final Verification Wave

After implementation, run a final pass that checks:
- premature completion no longer fires at cast start,
- stop behavior is no longer hyperactive,
- the positive feedback loop is materially reduced,
- smoke evidence exists for all scenarios.

## Commit Strategy

- 1: `fix(macro): stabilize reeling and cast transitions`

## Success Criteria

### Verification Commands
```bash
python -m py_compile src/*.py  # Expected: no syntax errors
python scripts/smoke_fisch_macro.py  # Expected: all scenarios pass
```

### Final Checklist
- [ ] Premature cast-completion signal fixed
- [ ] Cast-stop behavior debounced / hysteresis-added
- [ ] Positive feedback loop damped
- [ ] Smoke verification passes
