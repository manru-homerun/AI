from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "conditional_gru_decoder_experiment"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Conditional GRU decoder evaluation artifacts.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    with open(artifact_dir / "metrics.json", encoding="utf-8") as fp:
        metrics = json.load(fp)
    with open(artifact_dir / "onnx_export.json", encoding="utf-8") as fp:
        onnx_export = json.load(fp)

    summary = {
        "artifact_dir": str(artifact_dir.relative_to(ROOT)),
        "best_config": metrics["best_config"],
        "experiments": [
            {
                "config": experiment["config"],
                "teacher_forcing_test": experiment["teacher_forcing_test"],
                "greedy_route_test": experiment["greedy_route_test"],
            }
            for experiment in metrics["experiments"]
        ],
        "onnx_export": onnx_export,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
