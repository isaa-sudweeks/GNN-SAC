# Repository Guidelines

## Project Structure & Module Organization

- `sac/` contains the SAC implementations, graph and MLP actor-critic layers, replay buffers, trainers, and the `gnn_train.py`/`gnn_infer.py` entry points.
- `env/` defines MuJoCo environments and wrappers. Generated-topology adapters are under `env/mujoco_gen/`; XML assets for hand-authored trusses live in `env/truss/assets/`.
- `config/` holds composable Hydra YAML for algorithms, environments, GNNs, physical parameters, domain randomization, inference, and cluster runs.
- `tests/` contains checkpoint, inference, and MuJoCo smoke tests. `scripts/` contains performance benchmarks; `docs/` records design and experiment decisions.
- Treat `outputs/`, `logs/`, `checkpoints/`, and W&B data as generated artifacts; do not commit them.

## Setup, Test, and Development Commands

Create an isolated environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the full test suite with `python -m unittest discover -s tests -v`, or target one module with `python -m unittest tests.test_checkpointing -v`. Tests involving MuJoCo require a working rendering backend and `mujoco-truss-gen`.

Start a short local training run with:

```bash
python sac/gnn_train.py device=cpu steps=1000 enable_wandb=false
```

Hydra accepts command-line overrides. Quote list values in zsh, for example `'truss_topologies=[octahedron,tetrahedron]'`. Run backend benchmarks with `python scripts/benchmark_mujoco_backends.py`.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python conventions: `snake_case` for functions, variables, and modules; `PascalCase` for classes; and uppercase constants. Keep environment construction in `env/`, learning logic in `sac/`, and experiment defaults in YAML rather than hard-coded values. Add type hints to new public helpers and concise docstrings where behavior is not obvious. No formatter or linter is currently enforced; keep imports grouped as standard library, third-party, then local.

## Testing Guidelines

Tests use `unittest` classes and methods named `test_<behavior>`. Add deterministic unit tests for math, parsing, buffers, and checkpoint changes. Environment or trainer changes should include a small CPU smoke test with W&B, video, and artifact saving disabled. There is no declared coverage threshold; cover regression paths and configuration edge cases directly.

## Commit & Pull Request Guidelines

Recent history primarily uses Conventional Commit prefixes such as `feat:`, `fix:`, `refactor:`, and `docs:`. Write imperative, focused subjects and avoid mixing unrelated experiments. Pull requests should explain the behavioral change, list test commands, identify affected Hydra overrides/topologies, and link the relevant issue or experiment. Include logs, metrics, or screenshots only when they substantiate training, performance, or rendering changes; never include credentials or large generated model files.
