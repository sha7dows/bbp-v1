"""用途：重放已训练 checkpoint，在验证集选择阈值并冻结评估测试集。"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from tqdm.auto import tqdm

from bbp_behm.config import ExperimentConfig, load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset
from bbp_behm.hypergraph import HyperedgeBuilder
from bbp_behm.model import BBPBeHM
from bbp_behm.thresholding import binary_metrics_at_threshold, select_validation_threshold
from bbp_behm.training import class_priors, make_loader, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a binary threshold on validation predictions and freeze it for test evaluation."
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/vine_dedup_cv5_8020_seed42_pw1.5.ini"))
    parser.add_argument("--device", help="Evaluation device override, for example cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, help="Evaluation batch-size override.")
    parser.add_argument("--run-name", action="append", dest="run_names")
    parser.add_argument("--output-name", default="threshold_evaluation.json")
    parser.add_argument("--summary-name", default="threshold_summary.json")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_config(base: ExperimentConfig, resolved: Dict) -> ExperimentConfig:
    """保留本机数据路径，同时恢复服务器 run 的其余实际配置。"""
    data = resolved["data"]
    expected = {
        "processed_sha256": base.data.processed_sha256,
        "expected_sessions": base.data.expected_sessions,
        "expected_nodes": base.data.expected_nodes,
        "expected_features": base.data.expected_features,
    }
    for key, local_value in expected.items():
        if str(data[key]).upper() != str(local_value).upper():
            raise ValueError(f"Resolved data field {key} does not match the local config.")

    split_values = dict(resolved["split"])
    split_values["seeds"] = tuple(int(seed) for seed in split_values["seeds"])
    pruning_values = dict(resolved["pruning"])
    pruning_values["candidate_ratios"] = tuple(
        float(value) for value in pruning_values["candidate_ratios"]
    )
    training_values = dict(resolved["training"])
    # 设备由当前机器或 CLI 决定，避免复用服务器的 cuda:N 编号。
    training_values["device"] = base.training.device
    config = replace(
        base,
        split=replace(base.split, **split_values),
        hypergraph=replace(base.hypergraph, **resolved["hypergraph"]),
        model=replace(base.model, **resolved["model"]),
        pruning=replace(base.pruning, **pruning_values),
        training=replace(base.training, **training_values),
    )
    config.validate()
    return config


@torch.no_grad()
def collect_predictions(
    model: BBPBeHM,
    loader,
    builder: HyperedgeBuilder,
    device: torch.device,
    result: Dict,
    description: str,
    no_progress: bool,
) -> Tuple[List[int], List[int], List[float]]:
    model.eval()
    session_indices: List[int] = []
    labels: List[int] = []
    probabilities: List[float] = []
    progress = tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        disable=no_progress,
    )
    for cpu_batch in progress:
        batch = cpu_batch.to(device)
        hypergraph = builder.build_batch(batch)
        output = model(
            batch,
            hypergraph,
            retention_ratio=result["retention_ratio"],
            pruning_enabled=result["mode"] == "bbp",
            stochastic_selection=False,
            budget_mode=result["budget_mode"],
            coverage_threshold=result["coverage_threshold"],
            min_retention_ratio=result["min_retention_ratio"],
            max_retention_ratio=result["max_retention_ratio"],
            coverage_q=result["coverage_q"],
            coverage_tail_gamma=result["coverage_tail_gamma"],
            coverage_eta=result["coverage_eta"],
        )
        positive = torch.softmax(output.logits, dim=1)[:, 1]
        session_indices.extend(batch.session_indices.detach().cpu().tolist())
        labels.extend(batch.labels.detach().cpu().tolist())
        probabilities.extend(positive.detach().cpu().tolist())
    return session_indices, labels, probabilities


def _prediction_records(
    indices: Iterable[int], labels: Iterable[int], probabilities: Iterable[float]
) -> List[Dict[str, float]]:
    return [
        {
            "session_index": int(index),
            "label": int(label),
            "positive_probability": float(probability),
        }
        for index, label, probability in zip(indices, labels, probabilities)
    ]


def evaluate_run(
    run_dir: Path,
    base_config: ExperimentConfig,
    dataset: PreprocessedSessionDataset,
    device: torch.device,
    batch_size_override: int | None,
    output_name: str,
    no_progress: bool,
) -> Dict:
    result = _load_json(run_dir / "result.json")
    split = _load_json(run_dir / "split.json")
    config = _run_config(base_config, result["resolved_config"])
    if split["processed_sha256"].upper() != base_config.data.processed_sha256.upper():
        raise ValueError(f"Split data hash does not match local data for {run_dir.name}.")
    batch_size = config.training.batch_size if batch_size_override is None else batch_size_override
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    set_seed(int(result["seed"]))
    priors = class_priors(dataset, split["train"], config.model.num_classes)
    model = BBPBeHM(config, priors)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    builder = HyperedgeBuilder(config.hypergraph)
    validation_loader = make_loader(
        dataset, split["validation"], batch_size, False, int(result["seed"])
    )
    test_loader = make_loader(dataset, split["test"], batch_size, False, int(result["seed"]))

    val_indices, val_labels, val_probabilities = collect_predictions(
        model,
        validation_loader,
        builder,
        device,
        result,
        f"{run_dir.name} validation",
        no_progress,
    )
    test_indices, test_labels, test_probabilities = collect_predictions(
        model,
        test_loader,
        builder,
        device,
        result,
        f"{run_dir.name} test",
        no_progress,
    )
    selection = select_validation_threshold(val_labels, val_probabilities)
    selected_threshold = float(selection["threshold"])
    test_at_half = binary_metrics_at_threshold(test_labels, test_probabilities, 0.5)
    test_selected = binary_metrics_at_threshold(
        test_labels, test_probabilities, selected_threshold
    )
    replay_delta = {
        key: float(test_at_half[key] - result["test"][key])
        for key in ("macro_f1", "positive_f1", "precision", "recall", "auroc")
    }
    artifact = {
        "_purpose": "Validation-selected threshold frozen for BBP-BeHM test evaluation",
        "run_name": run_dir.name,
        "seed": int(result["seed"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "evaluation_device": str(device),
        "batch_size": int(batch_size),
        "logit_adjustment": result["logit_adjustment"],
        "positive_class_weight": float(result["positive_class_weight"]),
        "threshold_selection": selection,
        "validation": {
            "threshold_0_5": binary_metrics_at_threshold(val_labels, val_probabilities, 0.5),
            "selected_threshold": selection["metrics"],
        },
        "test": {
            "threshold_0_5": test_at_half,
            "selected_threshold": test_selected,
            "replay_delta_from_result": replay_delta,
        },
        "validation_predictions": _prediction_records(
            val_indices, val_labels, val_probabilities
        ),
        "test_predictions": _prediction_records(test_indices, test_labels, test_probabilities),
    }
    (run_dir / output_name).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main() -> None:
    args = parse_args()
    base_config = load_experiment_config(args.config)
    requested_device = base_config.training.device if args.device is None else args.device
    device = resolve_device(requested_device)
    dataset = PreprocessedSessionDataset(
        base_config.data.processed_path,
        base_config.data.processed_sha256,
        base_config.data.expected_sessions,
        base_config.data.expected_nodes,
        base_config.data.expected_features,
    )
    if args.run_names:
        run_dirs = [args.runs_root / name for name in args.run_names]
    else:
        run_dirs = sorted(
            path
            for path in args.runs_root.iterdir()
            if path.is_dir()
            and (path / "result.json").is_file()
            and (path / "split.json").is_file()
            and (path / "best_model.pt").is_file()
        )
    if not run_dirs:
        raise ValueError("No complete run directories were found.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    artifacts = []
    for run_dir in run_dirs:
        logging.info("Threshold evaluation started: %s", run_dir)
        artifact = evaluate_run(
            run_dir,
            base_config,
            dataset,
            device,
            args.batch_size,
            args.output_name,
            args.no_progress,
        )
        artifacts.append(artifact)
        logging.info(
            "Threshold evaluation completed: run=%s threshold=%.6f validation_f1=%.4f test_f1=%.4f",
            run_dir.name,
            artifact["threshold_selection"]["threshold"],
            artifact["validation"]["selected_threshold"]["positive_f1"],
            artifact["test"]["selected_threshold"]["positive_f1"],
        )

    summary = {
        "_purpose": "Aggregate validation-threshold summary for BBP-BeHM runs",
        "runs_root": str(args.runs_root),
        "evaluation_device": str(device),
        "runs": [
            {
                "run_name": artifact["run_name"],
                "logit_adjustment": artifact["logit_adjustment"],
                "positive_class_weight": artifact["positive_class_weight"],
                "threshold": artifact["threshold_selection"]["threshold"],
                "validation_at_0_5": artifact["validation"]["threshold_0_5"],
                "validation_selected": artifact["validation"]["selected_threshold"],
                "test_at_0_5": artifact["test"]["threshold_0_5"],
                "test_selected": artifact["test"]["selected_threshold"],
                "replay_delta_from_result": artifact["test"]["replay_delta_from_result"],
            }
            for artifact in artifacts
        ],
    }
    (args.runs_root / args.summary_name).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
