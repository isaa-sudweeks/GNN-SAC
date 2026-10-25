> Published measured report. Static figures are included here. Raw CSV/JSON, interactive HTML, replay and checkpoint filenames mentioned below refer to ignored local artifacts under `outputs/dynamics-pilot/online-multi-gamma095-source-2026-10-24/`; large generated artifacts are not part of this Git snapshot.

# Discount-only source dynamics-MPC comparison

Both campaigns are complete. Planning discount changes from .995 to .95; all other settings, initial weights, random warmups and fitting-only normalization match. Frozen checkpoint resets pair exactly. Returns below compare controllers under the same scoring convention using their saved executed reward traces. No new fitting or simulator evaluation.

[Common-discount learning curves](figures/discount-comparison-learning-curves.svg)

## Measured conclusion

**Gamma=.95 was not consistently better for this dynamics-only learned-MPC setup.** Tetrahedron specialists, shared six-node models and shared eight-node models have lower fresh-final mean return in all three seeds when both campaigns are scored at .995. Separate six-node models improve in two seeds and worsen in one. Shared tetrahedron outcomes are mixed. Changing the discount did not resolve eight-node survival, and it did not establish sharing without a control cost.

## Fresh outcomes: .95 controller versus .995 controller

Each row uses 15 paired fresh episodes (three training seeds, five resets each). The return columns rescore both controllers using the same discount. Collapse and movement definitions are unchanged. All per-seed differences are retained below.

| Topology | Model | Movement .995 → .95 | Collapses .995 → .95 | Return difference scored .95 | Return difference scored .995 | Undiscounted difference |
|---|---|---|---|---:|---:|---:|
| tetrahedron | shared | 0.491443 → 0.466622 | 4/15 → 3/15 | -0.005913 | +0.019381 | +0.042043 |
| tetrahedron | separate | 0.525859 → 0.463219 | 0/15 → 5/15 | -0.008280 | -0.284042 | -0.396039 |
| henneberg_n6_2tube_1 | shared | 0.330990 → 0.323092 | 9/15 → 15/15 | -0.049534 | -0.305721 | -0.407603 |
| henneberg_n6_2tube_1 | separate | 0.370765 → 0.425827 | 13/15 → 11/15 | +0.030366 | +0.157069 | +0.187962 |
| henneberg_n8_3tube_64 | shared | 0.330344 → 0.274485 | 15/15 → 15/15 | -0.048838 | -0.081107 | -0.055563 |
| henneberg_n8_3tube_64 | separate | 0.291398 → 0.291057 | 15/15 → 15/15 | -0.018641 | -0.012404 | -0.000208 |

Tetrahedron shared collapses fall from 4/15 to 3/15, but its specialist collapses rise from 0/15 to 5/15. Shared six-node collapses rise from 9/15 to 15/15; separate six-node collapses fall from 13/15 to 11/15. Both eight-node conditions remain at 15/15 collapses. Shared forward movement decreases in all three topology means; specialists gain movement only on the six-node mean. These results do not support adopting .95 as a general improvement across trained sources.

A concrete metric distinction: shared six-node seed 1 improves by +0.030223 when both trajectories are scored at .95, while it decreases by -0.229833 when both are scored at .995 and adds two collapses. An apparent native-return improvement can therefore coexist with worse survival. Both scoring conventions were declared before this rerun, and both are reported; no checkpoint or preferred discount was selected from these outcomes.

## Within the gamma=.95 campaign

Native .95 shared-minus-separate fresh return differences are -0.001534 (tetrahedron), -0.056924 (six-node), and -0.000761 (eight-node). Collapses are shared/separate 3/15 versus 5/15, 15/15 versus 11/15, and 15/15 versus 15/15. Shared six-node movement is lower in all three seeds. Tetrahedron sharing improves relative to its specialists here, alongside specialist regression versus the original campaign; this is not a uniform gain from sharing or a general reliability result.

## Verification, interpretation and next step

All 732 native automatic integrity checks, 208 additional completion checks and the cross-discount pairing audit pass. The cross-discount audit verifies gamma is the only changed setting, matching initial weights/warmups/fitting-only normalization, actual paired initial-state hashes, trace hashes and native-return recomputation. It covers 720 evaluation episodes across both campaigns, 2,160 rescored rows and 1,080 paired differences. All 360,000 rerun fitting transitions, 20,000 updates per topology, shared Adam 60,000 steps, specialist Adam 20,000 steps and all declared milestone/fresh checkpoints are retained. No run failures, operational restarts, source changes, new normalization or diagnostic-based intervention occurred.

Rerun time through six completed reports: 4.0101 hours elapsed and 11.9485 summed group wall hours. Post-run auditing/rendering/note updates are additional. The completed saved-data diagnostic is separate from this gamma-only experiment: shared dynamics missed 4/4 original tetrahedron H10 collapse endpoints versus specialists 2/4, with mixed final-weight gradients (11/36 negative). It neither modified this run nor proves a unique cause of its outcomes.

Recommendation: retain the .995 campaign as the stronger demonstrated reference for tetrahedron specialists and shared six-node control in this sample, while preserving the six-node specialist improvement at .95 as topology-specific evidence. Prioritize a separately predeclared source-only near-collapse prediction/candidate-ranking validation using independent validation states before another method change. Discount alone did not repair the survival problem. No additional experiment, tuning, unseen-topology evaluation or budget extension is running.

The native learning curves, fresh paired plot and common-discount curves were visually inspected. PNG/SVG browser export initially failed inside the sandbox; successful exports use the same plotted measurements through the available Chrome runtime. Original errors remain preserved. The original automatic reports are preserved in verification. Planning latencies reflect concurrent CPU load and are not an intrinsic speed comparison. These are three-seed source-development results after an outcome-informed user choice of gamma, not unseen-topology transfer or formal confirmation.

## Fresh final gamma=.95 controller minus gamma=.995 controller, every topology/seed

| Topology | Condition | Seed | Return difference scored .95 | Return difference scored .995 | Undiscounted difference | Movement difference | Collapse difference / 5 |
|---|---|---:|---:|---:|---:|---:|---:|
| tetrahedron | shared | 0 | +0.004203 | +0.133952 | +0.205231 | +0.006220 | -1 |
| tetrahedron | shared | 1 | +0.005863 | +0.259454 | +0.373552 | -0.027030 | -2 |
| tetrahedron | shared | 2 | -0.027805 | -0.335263 | -0.452654 | -0.053653 | +2 |
| tetrahedron | separate | 0 | -0.013413 | -0.334456 | -0.454914 | -0.054747 | +2 |
| tetrahedron | separate | 1 | -0.001290 | -0.188078 | -0.269533 | -0.069576 | +1 |
| tetrahedron | separate | 2 | -0.010138 | -0.329591 | -0.463670 | -0.063598 | +2 |
| henneberg_n6_2tube_1 | shared | 0 | -0.030721 | -0.219784 | -0.334181 | +0.065806 | +2 |
| henneberg_n6_2tube_1 | shared | 1 | +0.030223 | -0.229833 | -0.366104 | +0.034393 | +2 |
| henneberg_n6_2tube_1 | shared | 2 | -0.148105 | -0.467546 | -0.522524 | -0.123893 | +2 |
| henneberg_n6_2tube_1 | separate | 0 | +0.044470 | +0.187018 | +0.235308 | +0.035717 | -1 |
| henneberg_n6_2tube_1 | separate | 1 | +0.057782 | +0.351468 | +0.439256 | +0.040087 | -2 |
| henneberg_n6_2tube_1 | separate | 2 | -0.011154 | -0.067281 | -0.110677 | +0.089382 | +1 |
| henneberg_n8_3tube_64 | shared | 0 | -0.047444 | -0.126871 | -0.115209 | -0.115711 | +0 |
| henneberg_n8_3tube_64 | shared | 1 | -0.008540 | -0.006065 | +0.007019 | +0.007264 | +0 |
| henneberg_n8_3tube_64 | shared | 2 | -0.090530 | -0.110386 | -0.058499 | -0.059129 | +0 |
| henneberg_n8_3tube_64 | separate | 0 | -0.009171 | -0.013929 | -0.016683 | -0.016553 | +0 |
| henneberg_n8_3tube_64 | separate | 1 | -0.085762 | -0.118986 | -0.103918 | -0.104533 | +0 |
| henneberg_n8_3tube_64 | separate | 2 | +0.039009 | +0.095704 | +0.119978 | +0.120064 | +0 |

## Limits

Each controller collects its own real experience. These are paired reset comparisons, not identical training trajectories after warmup. The .95 setting was selected after observing the original campaign and at the user request; this is source development evidence, not independent confirmation or unseen-topology transfer. A discounted-return improvement must be interpreted alongside displacement, collapse counts and undiscounted task return. More aggressive discounting can reduce the numerical charge of a late failure without improving survival. Three seeds and five fresh resets do not establish reliability. Native per-campaign reports contain within-gamma shared-minus-separate effects and prediction/latency tables.
