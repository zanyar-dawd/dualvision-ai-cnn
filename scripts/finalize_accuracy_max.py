"""Select candidates by validation only, then open the official test set once."""
import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.train_v2 import evaluate_candidate
from src.utilities.runtime import write_json


def finalize(dataset: str, config: dict, experiment_id: str) -> dict:
    candidate_dir = Path("results") / experiment_id / dataset
    candidate_training_path = candidate_dir / "training.json"
    candidate_checkpoint = Path("checkpoints") / experiment_id / dataset / "best.pt"
    if not candidate_training_path.is_file() or not candidate_checkpoint.is_file():
        raise FileNotFoundError(f"Completed {dataset} candidate artifacts are missing.")

    candidate = json.loads(candidate_training_path.read_text(encoding="utf-8"))
    baseline_training = json.loads(
        (Path("results/baseline") / dataset / "training.json").read_text(encoding="utf-8")
    )
    baseline_evaluation = json.loads(
        (Path("results/baseline") / dataset / "evaluation.json").read_text(encoding="utf-8")
    )
    selection = {
        "dataset": dataset,
        "selection_basis": "validation_accuracy_then_validation_loss",
        "baseline_best_validation_accuracy": baseline_training["best_validation_accuracy"],
        "candidate_best_validation_accuracy": candidate["best_validation_accuracy"],
        "candidate_best_epoch": candidate["best_epoch"],
        "official_test_used_for_selection": False,
        "historical_baseline_test_was_known": True,
    }
    if candidate["best_validation_accuracy"] <= baseline_training["best_validation_accuracy"]:
        selection.update(
            selected="baseline",
            final_test_action="candidate_not_opened",
            reason="Candidate did not exceed the baseline on the identical validation split.",
        )
        write_json(candidate_dir / "selection.json", selection)
        return selection

    selection.update(selected="candidate", final_test_action="evaluate_once")
    write_json(candidate_dir / "selection.json", selection)
    evaluation = evaluate_candidate(dataset, config, experiment_id, "cpu")
    rows = [
        {
            "model": "historical baseline",
            "architecture": baseline_training.get("architecture", "baseline custom CNN"),
            "parameters": baseline_training["parameters"],
            "best_validation_accuracy": baseline_training["best_validation_accuracy"],
            "test_accuracy": baseline_evaluation["accuracy"],
            "test_loss": baseline_evaluation["loss"],
            "macro_f1": baseline_evaluation["macro_f1"],
            "training_seconds": baseline_training["training_seconds"],
        },
        {
            "model": "validation-selected candidate",
            "architecture": candidate["architecture"],
            "parameters": candidate["parameters"],
            "best_validation_accuracy": candidate["best_validation_accuracy"],
            "test_accuracy": evaluation["accuracy"],
            "test_loss": evaluation["loss"],
            "macro_f1": evaluation["macro_f1"],
            "training_seconds": candidate["training_seconds"],
        },
    ]
    pd.DataFrame(rows).to_csv(candidate_dir / "baseline_comparison.csv", index=False)
    write_json(candidate_dir / "baseline_comparison.json", rows)
    write_json(
        candidate_dir / "finalization.json",
        {
            **selection,
            "test_accuracy": evaluation["accuracy"],
            "test_loss": evaluation["loss"],
            "macro_f1": evaluation["macro_f1"],
        },
    )
    return {**selection, "test_accuracy": evaluation["accuracy"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("mnist", "cifar10", "both"),
        default="both",
        help="Finalize one dataset or both (default: both).",
    )
    args = parser.parse_args()
    configs = json.loads(args.config.read_text(encoding="utf-8"))
    datasets = ("cifar10", "mnist") if args.dataset == "both" else (args.dataset,)
    result = {
        dataset: finalize(dataset, configs[dataset], args.experiment_id)
        for dataset in datasets
    }
    write_json(Path("results") / args.experiment_id / "selection_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
