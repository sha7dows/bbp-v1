"""用途：用既有 checkpoint 在 validation split 上关闭剪枝并回放完整图，绝不评估 test。"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict

import torch

from bbp_behm.config import ExperimentConfig, load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset
from bbp_behm.hypergraph import HyperedgeBuilder
from bbp_behm.model import BBPBeHM
from bbp_behm.training import class_priors, evaluate, make_loader, resolve_device, set_seed
from scripts.evaluate_thresholds import _run_config


COMPARISON_KEYS = (
    "macro_f1",
    "positive_f1",
    "precision",
    "recall",
    "auroc",
    "edge_retention",
    "incidence_retention",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one checkpoint on validation with pruning disabled; never evaluate test."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/vine_dedup_cv5_8020_seed42_pw1.5.ini")
    )
    parser.add_argument("--device", help="Evaluation device override, for example cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, help="Evaluation batch-size override.")
    parser.add_argument("--output-name", default="validation_full_graph_replay.json")
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_deltas(full_graph: Dict, pruned: Dict) -> Dict[str, float]:
    """计算完整图 validation 指标相对源 checkpoint 剪枝 validation 的差值。"""
    return {
        key: float(full_graph[key]) - float(pruned[key])
        for key in COMPARISON_KEYS
        if key in full_graph and key in pruned
    }


def run_validation_replay(
    run_dir: Path,
    base_config: ExperimentConfig,
    device: torch.device,
    batch_size_override: int | None,
) -> Dict:
    result = _load_json(run_dir / "result.json")
    split = _load_json(run_dir / "split.json")
    if result["mode"] != "bbp":
        raise ValueError("Full-graph replay requires a BBP checkpoint.")
    if split["processed_sha256"].upper() != base_config.data.processed_sha256.upper():
        raise ValueError("Split data hash does not match the supplied config.")

    config = _run_config(base_config, result["resolved_config"])
    batch_size = config.training.batch_size if batch_size_override is None else batch_size_override
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    set_seed(int(result["seed"]))
    dataset = PreprocessedSessionDataset(
        base_config.data.processed_path,
        base_config.data.processed_sha256,
        base_config.data.expected_sessions,
        base_config.data.expected_nodes,
        base_config.data.expected_features,
    )
    priors = class_priors(dataset, split["train"], config.model.num_classes)
    model = BBPBeHM(config, priors)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    builder = HyperedgeBuilder(config.hypergraph)
    validation_loader = make_loader(
        dataset, split["validation"], batch_size, False, int(result["seed"])
    )

    started = time.perf_counter()
    full_graph = evaluate(
        model,
        validation_loader,
        builder,
        device,
        retention_ratio=float(result["retention_ratio"]),
        pruning_enabled=False,
        budget_mode=result["budget_mode"],
        coverage_threshold=float(result["coverage_threshold"]),
        min_retention_ratio=float(result["min_retention_ratio"]),
        max_retention_ratio=float(result["max_retention_ratio"]),
        coverage_q=float(result["coverage_q"]),
        coverage_tail_gamma=float(result["coverage_tail_gamma"]),
        coverage_eta=float(result["coverage_eta"]),
    )
    source_validation = result["best_validation"]
    return {
        "_purpose": "Validation full-graph replay with the source checkpoint",
        "protocol": "validation_only_full_graph_replay",
        "test_evaluated": False,
        "source_run": run_dir.name,
        "source_checkpoint_epoch": int(checkpoint["epoch"]),
        "seed": int(result["seed"]),
        "evaluation_device": str(device),
        "batch_size": int(batch_size),
        "pruning_enabled": False,
        "source_pruned_validation": source_validation,
        "full_graph_validation": full_graph,
        "full_graph_minus_source_pruned": metric_deltas(full_graph, source_validation),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base_config = load_experiment_config(args.config)
    requested_device = base_config.training.device if args.device is None else args.device
    device = resolve_device(requested_device)
    artifact = run_validation_replay(
        args.run_dir, base_config, device, args.batch_size
    )
    output_path = args.run_dir / args.output_name
    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics = artifact["full_graph_validation"]
    logging.info(
        "Validation full-graph replay completed: f1=%.4f macro_f1=%.4f "
        "edge_retention=%.4f test_evaluated=false output=%s",
        metrics["positive_f1"],
        metrics["macro_f1"],
        metrics["edge_retention"],
        output_path,
    )


if __name__ == "__main__":
    main()
