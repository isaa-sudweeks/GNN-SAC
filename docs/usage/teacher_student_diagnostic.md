# Teacher versus separate/shared student diagnostic

`scripts/run_teacher_student_diagnostic.py` isolates distillation fidelity from
transfer to unseen topologies. Its three worker actions consume an explicit JSON
manifest and default to `SLURM_ARRAY_TASK_ID` as their index:

```bash
uv run python scripts/run_teacher_student_diagnostic.py prepare --manifest /absolute/path/manifest.json
uv run python scripts/run_teacher_student_diagnostic.py train --manifest /absolute/path/manifest.json
uv run python scripts/run_teacher_student_diagnostic.py evaluate --manifest /absolute/path/manifest.json
```

The `teacher-student-v1` experiment uses three previously seen training presets:
`henneberg_n6_1tube_1`, `octahedron`, and `henneberg_n8_3tube_34`. For seeds 1–3,
train one student per topology plus a shared student on all three (12 jobs).
Students use the signed-routing architecture with attention and a global node;
tube membership and physics features are disabled. Actor initialization hashes
and parameter counts are recorded to verify matched capacity and initialization.

Each student trains for 10,000 updates, sampling 256 observations **per topology
per update** from the same 16,384-transition clean teacher replay. The shared
objective averages the three topology losses. It therefore has three times the
total samples and compute per update, while exposure per topology is matched.
No SAC updates or teacher retraining occur. This comparison tests the cost of
sharing capacity, rather than equal total compute across separate/shared jobs.

Preparation collects independent 4,096-transition teacher replay using seed
271828 (training replay uses 314159). These validation observations are never
passed to the training worker. Each worker evaluates teacher rollouts on its own
topology using deterministic mean actions and reset seeds 1000–1004.

Student snapshots at 2,000 and 10,000 updates are evaluated on 512 observations
sampled from that fresh replay, with Gaussian forward KL and tanh-mean action
MSE. Student rollouts use the **same teacher-derived environment configuration**
and reset seeds as the teacher, with observation/action noise and geometry
scaling disabled. Other reset/dynamics randomization remains as configured by
the teacher. Results include all individual distances, lengths and termination
flags, rather than only aggregate reward.

The shared student's per-topology actor-gradient cosine similarities are computed
on the fresh evaluation batch at each snapshot. Negative values identify opposing
optimization directions at that snapshot; two snapshots cannot establish that
conflict occurs frequently, and conflicting gradients alone do not prove
incompatible gaits.

On ORC, the experiment lives under
`~/nobackup/autodelete/GNN-SAC/runs/teacher-student-v1`. The manifest, exact worker
copy/hash, base commit, Slurm scripts and submission IDs are retained there.
Preparation is a three-task CPU array; training is a 12-task GPU array capped at
three simultaneous workers; CPU evaluation depends on both arrays succeeding.
All caches, temporary files, checkpoints and offline W&B records use autodelete.
Training logs go to W&B; `evaluation.json` and `*.teacher.json` contain the
independent diagnostic measurements and require separate reporting.

Interpretation:

- Poor separate-student fidelity points toward distillation, coverage, capacity,
  or rollout distribution shift before a teacher-interference explanation.
- Strong separate students and weaker shared students support joint-learning
  interference; capacity and optimization remain alternative explanations.
- Strong teacher reproduction on these training topologies, together with poor
  unseen-topology results, points toward transfer rather than fitting failure.

This is a three-topology training-set diagnostic, not an independent test of
generalization or a final-test evaluation. No N7 final-test presets are used.
