"""用途：无需重训地扫描 robust 剪枝策略，支持严格的 validation-only 选参。"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from bbp_behm.config import ExperimentConfig, load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset
from bbp_behm.hypergraph import HyperedgeBuilder
from bbp_behm.model import BBPBeHM
from bbp_behm.training import class_priors, evaluate, make_loader, resolve_device, set_seed
from scripts.evaluate_thresholds import _run_config


DEFAULT_GAMMAS = (0.85, 0.90, 0.95)
DEFAULT_MIN_RETENTIONS = (0.0, 0.3, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep robust pruning policies on validation; optionally skip test entirely."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/vine_dedup_cv5_8020_seed42_pw1.5.ini"))
    parser.add_argument("--device", help="Evaluation device override, for example cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, help="Evaluation batch-size override.")
    parser.add_argument("--gamma", type=float, nargs="+", default=list(DEFAULT_GAMMAS))
    parser.add_argument(
        "--min-retention",
        type=float,
        nargs="+",
        default=list(DEFAULT_MIN_RETENTIONS),
    )
    parser.add_argument(
        "--eta",
        type=float,
        nargs="+",
        help="One or more eta values; keep the source run value when omitted.",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Never construct or evaluate a test loader.",
    )
    parser.add_argument("--output-name", default="pruning_policy_sweep.json")
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def policy_grid(
    gammas: Iterable[float],
    min_retentions: Iterable[float],
    max_retention: float,
) -> List[Tuple[float, float]]:
    """校验并生成去重后的 gamma × 最小保留率笛卡尔积。"""
    unique_gammas = tuple(dict.fromkeys(float(value) for value in gammas))
    unique_minimums = tuple(dict.fromkeys(float(value) for value in min_retentions))
    if not unique_gammas or not unique_minimums:
        raise ValueError("The policy grid cannot be empty.")
    if any(not 0.0 < value <= 1.0 for value in unique_gammas):
        raise ValueError("Every gamma must be in (0, 1].")
    if any(not 0.0 <= value <= max_retention for value in unique_minimums):
        raise ValueError("Every minimum retention must be within [0, max_retention].")
    return [
        (gamma, minimum)
        for gamma in unique_gammas
        for minimum in unique_minimums
    ]


def eta_policy_grid(
    gammas: Iterable[float],
    min_retentions: Iterable[float],
    etas: Iterable[float],
    max_retention: float,
) -> List[Tuple[float, float, float]]:
    """校验并生成 gamma × 最小保留率 × eta 的去重笛卡尔积。"""
    base_grid = policy_grid(gammas, min_retentions, max_retention)
    unique_etas = tuple(dict.fromkeys(float(value) for value in etas))
    if not unique_etas:
        raise ValueError("The eta grid cannot be empty.")
    if any(not 0.0 <= value <= 1.0 for value in unique_etas):
        raise ValueError("Every eta must be in [0, 1].")
    return [
        (gamma, minimum, eta)
        for gamma, minimum in base_grid
        for eta in unique_etas
    ]


def _selection_metric(value: float) -> float:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else float("-inf")


def select_validation_policy(candidates: Sequence[Dict]) -> Dict:
    """严格按验证指标选策略；性能相同才偏向更强压缩。"""
    if not candidates:
        raise ValueError("At least one validation candidate is required.")

    def ranking(candidate: Dict) -> Tuple[float, ...]:
        metrics = candidate["validation"]
        return (
            _selection_metric(metrics["positive_f1"]),
            _selection_metric(metrics["macro_f1"]),
            _selection_metric(metrics["auroc"]),
            -float(metrics["edge_retention"]),
            -float(metrics["incidence_retention"]),
            -float(candidate.get("eta_distance_from_source", 0.0)),
            -float(candidate["coverage_tail_gamma"]),
            -float(candidate["min_retention_ratio"]),
        )

    return max(candidates, key=ranking)


def _evaluate_policy(
    model: BBPBeHM,
    loader,
    builder: HyperedgeBuilder,
    device: torch.device,
    result: Dict,
    gamma: float,
    minimum: float,
    eta: float,
) -> Dict[str, float]:
    return evaluate(
        model,
        loader,
        builder,
        device,
        retention_ratio=float(result["retention_ratio"]),
        pruning_enabled=True,
        budget_mode="robust_adaptive",
        coverage_threshold=float(result["coverage_threshold"]),
        min_retention_ratio=minimum,
        max_retention_ratio=float(result["max_retention_ratio"]),
        coverage_q=float(result["coverage_q"]),
        coverage_tail_gamma=gamma,
        coverage_eta=eta,
    )


def run_sweep(
    run_dir: Path,
    base_config: ExperimentConfig,
    device: torch.device,
    batch_size_override: int | None,
    gammas: Sequence[float],
    min_retentions: Sequence[float],
    eta_overrides: Sequence[float] | None,
    validation_only: bool = False,
) -> Dict:
    result = _load_json(run_dir / "result.json")
    split = _load_json(run_dir / "split.json")
    if result["mode"] != "bbp" or result["budget_mode"] != "robust_adaptive":
        raise ValueError("Policy sweep requires a robust_adaptive BBP checkpoint.")
    if split["processed_sha256"].upper() != base_config.data.processed_sha256.upper():
        raise ValueError("Split data hash does not match the local config.")

    config = _run_config(base_config, result["resolved_config"])
    batch_size = config.training.batch_size if batch_size_override is None else batch_size_override
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    source_eta = float(result["coverage_eta"])
    requested_etas = [source_eta] if eta_overrides is None else eta_overrides
    grid = eta_policy_grid(
        gammas,
        min_retentions,
        requested_etas,
        float(result["max_retention_ratio"]),
    )

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
    candidates: List[Dict] = []
    for gamma, minimum, eta in grid:
        candidate_started = time.perf_counter()
        metrics = _evaluate_policy(
            model,
            validation_loader,
            builder,
            device,
            result,
            gamma,
            minimum,
            eta,
        )
        candidate = {
            "coverage_tail_gamma": gamma,
            "min_retention_ratio": minimum,
            "coverage_eta": eta,
            "eta_distance_from_source": abs(eta - source_eta),
            "elapsed_seconds": time.perf_counter() - candidate_started,
            "validation": metrics,
        }
        candidates.append(candidate)
        logging.info(
            "Validation policy completed: gamma=%.4f min_retention=%.4f eta=%.4f "
            "f1=%.4f macro_f1=%.4f auroc=%.4f edge_retention=%.4f",
            gamma,
            minimum,
            eta,
            metrics["positive_f1"],
            metrics["macro_f1"],
            metrics["auroc"],
            metrics["edge_retention"],
        )

    selected = select_validation_policy(candidates)
    test_metrics = None
    if not validation_only:
        # 只有验证集选择结束后才构造并运行一次测试 loader。
        test_loader = make_loader(
            dataset, split["test"], batch_size, False, int(result["seed"])
        )
        test_metrics = _evaluate_policy(
            model,
            test_loader,
            builder,
            device,
            result,
            float(selected["coverage_tail_gamma"]),
            float(selected["min_retention_ratio"]),
            float(selected["coverage_eta"]),
        )
    artifact = {
        "_purpose": (
            "Strict validation-only robust policy sweep"
            if validation_only
            else "Validation robust policy sweep with one frozen test evaluation"
        ),
        "protocol": "validation_only" if validation_only else "validation_then_frozen_test",
        "test_evaluated": not validation_only,
        "source_run": run_dir.name,
        "source_checkpoint_epoch": int(checkpoint["epoch"]),
        "seed": int(result["seed"]),
        "evaluation_device": str(device),
        "batch_size": int(batch_size),
        "grid": {
            "coverage_tail_gamma": list(dict.fromkeys(float(value) for value in gammas)),
            "min_retention_ratio": list(
                dict.fromkeys(float(value) for value in min_retentions)
            ),
            "coverage_eta": list(
                dict.fromkeys(float(value) for value in requested_etas)
            ),
        },
        "selection_rule": [
            "higher_validation_positive_f1",
            "higher_validation_macro_f1",
            "higher_validation_auroc",
            "lower_validation_edge_retention",
            "lower_validation_incidence_retention",
            "eta_closer_to_source_on_exact_tie",
        ],
        "validation_candidates": candidates,
        "selected_policy": {
            "coverage_tail_gamma": selected["coverage_tail_gamma"],
            "min_retention_ratio": selected["min_retention_ratio"],
            "coverage_eta": selected["coverage_eta"],
            "validation": selected["validation"],
        },
        "source_policy": {
            "coverage_tail_gamma": float(result["coverage_tail_gamma"]),
            "min_retention_ratio": float(result["min_retention_ratio"]),
            "coverage_eta": float(result["coverage_eta"]),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    if test_metrics is not None:
        artifact["selected_policy"]["test"] = test_metrics
        artifact["source_policy"]["test_from_training_result"] = result["test"]
    return artifact


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base_config = load_experiment_config(args.config)
    requested_device = base_config.training.device if args.device is None else args.device
    device = resolve_device(requested_device)
    artifact = run_sweep(
        args.run_dir,
        base_config,
        device,
        args.batch_size,
        args.gamma,
        args.min_retention,
        args.eta,
        args.validation_only,
    )
    output_path = args.run_dir / args.output_name
    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    selected = artifact["selected_policy"]
    if artifact["test_evaluated"]:
        logging.info(
            "Policy sweep completed: gamma=%.4f min_retention=%.4f eta=%.4f "
            "validation_f1=%.4f test_f1=%.4f output=%s",
            selected["coverage_tail_gamma"],
            selected["min_retention_ratio"],
            selected["coverage_eta"],
            selected["validation"]["positive_f1"],
            selected["test"]["positive_f1"],
            output_path,
        )
    else:
        logging.info(
            "Validation-only policy sweep completed: gamma=%.4f min_retention=%.4f "
            "eta=%.4f validation_f1=%.4f output=%s",
            selected["coverage_tail_gamma"],
            selected["min_retention_ratio"],
            selected["coverage_eta"],
            selected["validation"]["positive_f1"],
            output_path,
        )


if __name__ == "__main__":
    main()
