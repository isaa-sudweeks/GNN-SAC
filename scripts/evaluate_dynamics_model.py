"""Evaluate a frozen trusted local dynamics checkpoint on indexed source episodes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sac")]
from dynamics.dataset import load_dataset
from dynamics.training import evaluate_recursive, load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--role', choices=['validation', 'evaluation'], default='evaluation')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = load_model(args.checkpoint)
    report = evaluate_recursive(model, load_dataset(args.data, roles=[args.role]), [1, 5, 10, 20])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
