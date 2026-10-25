# Discount-only shared-versus-separate source rerun — October 24, 2026

The user explicitly authorized a diagnostic agent on the completed campaign and a
concurrent rerun with discount factor 0.95. This is a new predeclared experiment,
not an operational resume of the gamma=0.995 run. Original results and checkpoints
remain preserved at `outputs/dynamics-pilot/online-multi-source-2026-10-24`.

## Fixed protocol and one changed setting

Use `config/dynamics_online_multi_gamma095.yaml`. Its only setting difference from
`config/dynamics_online_multi.yaml` is **gamma: 0.995 -> 0.95**. Reuse the existing
unchanged runner and exact source implementation. Both planning and evaluation use
0.95. Dynamics-only loss itself has no discount; gamma influences learning through
the learned-MPC actions that generate real replay. No reward/value/teacher or
simulator-MPC fitting data is introduced.

Keep tetrahedron, henneberg_n6_2tube_1 and henneberg_n8_3tube_64; shared versus
separate models; training seeds 0/1/2; matched initialization and random warmups;
common fitting-only frozen normalization; architecture, losses, learning rates,
physics, exploration, action masks/routing/aliases and episode boundaries; CEM
H10/16 candidates/two iterations; all checkpoint/reset schedules exactly as before.
Start from the original seeded random weights, never a prior final checkpoint.

Each topology/condition/seed receives exactly 20,000 real training transitions and
20,000 topology-specific updates. Shared Adam receives 60,000 steps per seed;
each specialist receives 20,000. Each topology contributes 640,000 one-step draws
and 640,000 five-step recursive windows (3,200,000 recursive target exposures).
Total training interactions: **360,000**. Monitoring at 1,600/4,000/8,000/12,000/
20,000 per topology with three paired resets; five additional paired final resets.
Use the same reset manifest/seed formula as the prior run for paired comparison.
No held-out/final-test topology access or outcome-driven checkpoint selection.

## Comparison and evidence

Report every topology and training seed, all monitoring checkpoints and fresh
final resets, displacement, collapse-penalized return, collapses, prediction errors
and planning latency. Verify original versus rerun initialization/warmup/scales.
Report shared-minus-separate paired effects within the new gamma condition.
Additionally compare gamma=.95 versus gamma=.995 controllers using the same
physical initial states and rescore their saved executed reward traces at both
common discounts (.95 and .995), plus undiscounted task return. No new rollouts
are needed for this cross-scoring. This separates controller outcome changes from
changes to the reported return definition. Displacement and collapse counts retain
their unchanged definitions. No assumption of improvement is made in advance.

`scripts/report_online_dynamics_discount_comparison.py` performs this descriptive
comparison after completion. Original held-out/final-test robots remain untouched.

## Resources, stopping and persistence

Reuse the verified three-worker benchmark and completed campaign measurements:
4.08 elapsed / 12.05 summed group wall hours in the preceding full experiment;
prelaunch benchmark projected 4.68 / 14.04 hours. Allocate the same six groups,
three CPU workers with one Torch thread each. Plan approximately **4–5 elapsed
hours**, with bounded extra contention from a one-thread saved-data diagnostic.
Keep at least the existing approximately 12 GiB free before launch; generated
checkpoints/replay remain ignored. No dependency changes or new GPU allocation.

Stop exactly at the fixed interaction budget or preserve numerical/integrity
failure. No automatic tuning, added seeds, extra training, diagnostic-based changes,
or unseen evaluation. Latest checkpoints every 1,000 rounds and retained declared
milestones preserve complete simulator, RNG, replay and optimizer state. Resume only
exact matching sources/config/dependencies after operational interruption. The
diagnostic agent cannot change any hashed training sources or the active rerun.
Record any diagnostic findings separately; do not fold them into this gamma-only
comparison. Arrange a quiet completion follow-up for verification, measured
reporting, curves, comparison with the original run and Obsidian updates.

## Reproduce

```bash
uv run python -m unittest tests.test_dynamics_discount_rerun -v
uv run python scripts/run_online_dynamics_multi.py --config config/dynamics_online_multi_gamma095.yaml --output outputs/dynamics-pilot/online-multi-gamma095-source-2026-10-24
# Same command plus --resume only after an operational interruption.
uv run python scripts/report_online_dynamics_discount_comparison.py --baseline outputs/dynamics-pilot/online-multi-source-2026-10-24 --rerun outputs/dynamics-pilot/online-multi-gamma095-source-2026-10-24
```

## Launch and preflight outcome

Launched at 2026-10-25 02:43:42 UTC (October 24 evening Mountain time), process
44566, with three one-thread CPU workers. All original campaign source hashes match;
only gamma differs in settings. No sources used by training were edited. New native
collection/evaluation smoke, planner ranking/config-only-change and common-discount
rescoring tests pass. A saved-data comparison fixture exercises 720 episodes / 2,160
rescored rows / 1,080 paired differences: identical trajectories compare as zero
under each common discount. Fixture scores are software checks, not new gamma-.95
results. Logs and frozen sources are saved in the ignored rerun verification folder.
Quiet completion follow-up: `finish-gamma-0-95-dynamics-mpc-comparison`.

The concurrent diagnostic has now completed in 42.47 seconds with three passing
regression tests and a passed 16-check artifact audit. It used original .995 saved
data only; see `docs/plans/gnn_dynamics_multi_diagnostic.md` and its report. Shared
missed 4/4 selected tetrahedron H10 collapse endpoints versus specialists 2/4;
local gradient cosines were mixed (11/36 negative). These are post hoc diagnostics,
not causal proof or a change to this active gamma-only experiment. Parent verified
114 input artifacts remain unchanged and independently reran all three tests. Both
figures were visually inspected. Obsidian notes 15 (rerun) and 16 (diagnostic) and
the pilot index are saved with exact readback/asset-hash verification.

## Completed outcome — October 25, 2026

All six groups completed the fixed 360,000 fitting transitions in 4.01 elapsed
hours (11.95 summed group wall hours), without run failures or operational restarts.
Only gamma changed. All 732 automatic and 208 additional completion checks pass;
the cross-discount audit verifies paired actual starts and matching initialization,
random warmups and frozen fitting-only normalization across both campaigns.

Gamma=.95 was not consistently better for this dynamics-only learned-MPC setup.
At common .995 scoring, fresh return differences (.95 controller minus .995
controller) are +0.019381 / -0.284042 for tetrahedron shared/separate,
-0.305721 / +0.157069 for six-node shared/separate, and -0.081107 / -0.012404 for
eight-node shared/separate. Tetrahedron specialists, shared six-node and shared
eight-node returns decrease in all three training seeds at this scoring convention.
All per-seed and both common-discount/undiscounted outcomes remain reported.

Fresh collapses (.995 -> .95) are tetrahedron shared 4/15 -> 3/15 and separate
0/15 -> 5/15; six-node shared 9/15 -> 15/15 and separate 13/15 -> 11/15; eight-node
both 15/15 -> 15/15. Shared six-node movement remains lower than separate models in
all three seeds. Discount-only changes did not establish reliable shared control.

Both campaigns' saved executed traces were scored at .95, .995 and 1.0 without new
fitting or simulator evaluation: 720 episodes, 2,160 rescored rows, 1,080 paired
differences. Reports/figures: `outputs/dynamics-pilot/online-multi-gamma095-source-2026-10-24/report.md`
and `discount-comparison/report.md` beneath that root. The diagnostic on original
saved data remains separate; it did not modify the rerun. No checkpoint selection,
source change, held-out/final-test access, tuning or extra training occurred.

Recommendation: do not treat .95 as a general improvement here. Preserve its
six-node specialist benefit alongside the regressions, and validate near-collapse
prediction and candidate ranking on a new source validation manifest before further
method changes. No additional experiment is authorized or running.

Completion reporting is verified: all three plots were visually inspected, final
artifact hashes pass, and both Obsidian notes preserve their earlier content with
all appended chunks read back exactly and copied figure hashes matching. The
campaign process has exited. The completion monitor is removed after delivery.
