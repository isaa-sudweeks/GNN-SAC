# Repository Guidelines

## Mandatory Experiment Notebook Workflow

- Before doing any repository work, including giving status or next-step advice, read the Obsidian experiment notebook in vault `My vault`, folder `Research/06 Experiment Notebook/`. Start with `GNN Dynamics and MPC Generalization Pilot.md`, list the folder for newer entries, and read the experiment notes relevant to the request. Use the official Obsidian CLI and the applicable Obsidian skills. Historical memory, another chat, and repository reports do not replace this check. Resolve historical pending/next-step statements against later completed entries.
- If the notebook cannot be read, explicitly report the access problem and do not present an experiment history or recommendation as notebook-verified. Restore access before launching new experiments or making changes that depend on that history.
- Before each new research experiment, record its question, relation to prior tests, fixed protocol, seeds/topologies, data roles, episode/planning/training horizons, metrics, compute estimate, stopping rules and claim limits in that same notebook folder. Link it from the parent index. Record material protocol changes and reviews before execution.
- After every experiment or diagnostic, append measured results, failures, verification, artifact paths, interpretation and the next unresolved question to its note and update the parent index in the same folder. Include relevant software-test results with the experiment; standalone code validation must also be recorded there when it changes the experimental evidence or method. Preserve existing notes/embeds and verify saved Markdown by CLI readback. Do not call a research task complete until this reporting is verified, or explicitly report the access blocker and preserve the pending entry locally.
- Keep raw replay, checkpoints and large generated outputs ignored in the repository; notebook entries should link to evidence and contain the readable protocol/results. Keep matching repository usage/design documentation current.
- The research goal is to test whether learned robot dynamics generalizes across morphologies more effectively than model-free control. Source-only prediction, shared-versus-separate fitting, and source locomotion are prerequisites, not evidence of unseen-topology transfer or comparative superiority. Such claims require a source-trained model-free generalist comparator with matched train/held-out topology access, task/physics, evaluation starts and explicit interaction/compute accounting. Never use held-out specialists as the zero-shot baseline or tune on final-test robots.

## Project Structure & Module Organization

- `sac/` contains the SAC implementations, graph and MLP actor-critic layers, replay buffers, trainers, and the `train.py` (all backends) and `gnn_infer.py` entry points. `gnn_train.py` is legacy (equivalent to `train.py sac_backend=gnn`); do not use it in new commands or docs.
- `env/` defines MuJoCo environments and wrappers. Generated-topology adapters are under `env/mujoco_gen/`; XML assets for hand-authored trusses live in `env/truss/assets/`.
- `config/` holds composable Hydra YAML for algorithms, environments, GNNs, physical parameters, domain randomization, inference, and cluster runs.
- `tests/` contains checkpoint, inference, and MuJoCo smoke tests. `scripts/` contains launchers, validators, and benchmarks.
- `docs/` is indexed in `docs/README.md`: `usage/` and `design/` must match the code, `plans/` holds pending plans, and `archive/` is frozen history. When a change alters documented behavior, update the matching `docs/usage/` or `docs/design/` page in the same change.
- Treat `outputs/`, `logs/`, `checkpoints/`, and W&B data as generated artifacts; do not commit them.

## Setup, Test, and Development Commands

Dependencies are managed only with uv (`pyproject.toml` + `uv.lock`); there is no `requirements.txt`. Add or change dependencies with `uv add`/`uv lock`, never by hand-editing the lock file.

```bash
uv sync --frozen
```

Run the full test suite with `uv run python -m unittest discover -s tests -v`, or target one module with `uv run python -m unittest tests.test_checkpointing -v`. Tests involving MuJoCo require a working rendering backend and `mujoco-truss-gen`.

Start a short local training run with:

```bash
uv run python sac/train.py sac_backend=gnn device=cpu steps=1000 enable_wandb=false
```

Hydra accepts command-line overrides. Quote list values in zsh, for example `'truss_topologies=[octahedron,tetrahedron]'`. Select the simulator with `sim_backend=mujoco|mjx` (not the lower-level `mujoco_backend` key). Run backend benchmarks with `uv run python scripts/benchmark_mujoco_backends.py`.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python conventions: `snake_case` for functions, variables, and modules; `PascalCase` for classes; and uppercase constants. Keep environment construction in `env/`, learning logic in `sac/`, and experiment defaults in YAML rather than hard-coded values. Add type hints to new public helpers and concise docstrings where behavior is not obvious. No formatter or linter is currently enforced; keep imports grouped as standard library, third-party, then local.

## Testing Guidelines

Tests use `unittest` classes and methods named `test_<behavior>`. Add deterministic unit tests for math, parsing, buffers, and checkpoint changes. Environment or trainer changes should include a small CPU smoke test with W&B, video, and artifact saving disabled. There is no declared coverage threshold; cover regression paths and configuration edge cases directly.

## Commit & Pull Request Guidelines

Recent history primarily uses Conventional Commit prefixes such as `feat:`, `fix:`, `refactor:`, and `docs:`. Write imperative, focused subjects and avoid mixing unrelated experiments. Pull requests should explain the behavioral change, list test commands, identify affected Hydra overrides/topologies, and link the relevant issue or experiment. Include logs, metrics, or screenshots only when they substantiate training, performance, or rendering changes; never include credentials or large generated model files.
