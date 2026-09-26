# Repository Guidelines

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
