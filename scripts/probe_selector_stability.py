"""用途：在 validation split 上隔离 selector dropout，诊断 robust 剪枝掩码与边排序的重复稳定性；不访问 test。"""

from __future__ import annotations

import argparse
import json
import logging
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import Tensor
from tqdm.auto import tqdm

from bbp_behm.config import ExperimentConfig, load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset
from bbp_behm.hypergraph import BehaviorType, HyperedgeBuilder
from bbp_behm.model import BBPBeHM
from bbp_behm.training import (
    class_priors,
    confidence_statistics,
    make_loader,
    resolve_device,
    set_seed,
)
from scripts.evaluate_thresholds import _run_config


STABILITY_METRIC_KEYS = (
    "mean_jaccard",
    "min_jaccard",
    "mean_flip_rate",
    "max_flip_rate",
    "mean_js_divergence",
    "max_js_divergence",
    "retention_mean",
    "retention_std",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat only the dropout-enabled selector on validation; never build a test loader."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/vine_dedup_cv5_8020_seed42_pw1.5.ini")
    )
    parser.add_argument("--device", help="Evaluation device override, for example cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, help="Validation batch-size override.")
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--probe-seed", type=int)
    parser.add_argument("--output-name", default="selector_stability_validation.json")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validation_indices(split: Dict) -> Tuple[int, ...]:
    """只提取 validation 索引；该 helper 不要求 split 中存在 test 字段。"""
    if "validation" not in split:
        raise ValueError("Split manifest has no validation indices.")
    indices = tuple(int(index) for index in split["validation"])
    if not indices:
        raise ValueError("Validation split cannot be empty.")
    return indices


def mask_pair_metrics(left: Tensor, right: Tensor) -> Dict[str, float]:
    """计算一对同长度 hard mask 的 Jaccard 与候选边翻转比例。"""
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Masks must be one-dimensional and share the same shape.")
    left_bool = left.bool()
    right_bool = right.bool()
    if left_bool.numel() == 0:
        return {"jaccard": 1.0, "flip_rate": 0.0}
    intersection = torch.logical_and(left_bool, right_bool).sum().item()
    union = torch.logical_or(left_bool, right_bool).sum().item()
    jaccard = 1.0 if union == 0 else float(intersection / union)
    flip_rate = float(torch.ne(left_bool, right_bool).float().mean().item())
    return {"jaccard": jaccard, "flip_rate": flip_rate}


def score_js_divergence(left: Tensor, right: Tensor) -> float:
    """计算同一 session/type 内两组 edge score softmax 分布的 JS divergence。"""
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Scores must be one-dimensional and share the same shape.")
    if left.numel() <= 1:
        return 0.0
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise ValueError("Scores must be finite.")
    left_log = torch.log_softmax(left.float(), dim=0)
    right_log = torch.log_softmax(right.float(), dim=0)
    left_prob = left_log.exp()
    right_prob = right_log.exp()
    middle_prob = 0.5 * (left_prob + right_prob)
    middle_log = middle_prob.clamp_min(1e-12).log()
    divergence = 0.5 * (
        torch.sum(left_prob * (left_log - middle_log))
        + torch.sum(right_prob * (right_log - middle_log))
    )
    return float(divergence.clamp_min(0.0).item())


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return float(sum(materialized) / len(materialized)) if materialized else 0.0


def summarize_view_group(masks: Tensor, scores: Tensor) -> Dict[str, float]:
    """把一个 session 或 session/type 的多次 selector 结果汇总为单条记录。"""
    if masks.ndim != 2 or scores.ndim != 2 or masks.shape != scores.shape:
        raise ValueError("masks and scores must share shape [passes, edges].")
    if masks.size(0) < 2:
        raise ValueError("At least two selector passes are required.")

    pair_masks: List[Dict[str, float]] = []
    pair_js: List[float] = []
    for left_index, right_index in combinations(range(masks.size(0)), 2):
        pair_masks.append(mask_pair_metrics(masks[left_index], masks[right_index]))
        pair_js.append(score_js_divergence(scores[left_index], scores[right_index]))

    if masks.size(1) == 0:
        retentions = masks.new_ones((masks.size(0),), dtype=torch.float32)
    else:
        retentions = masks.float().mean(dim=1)
    return {
        "candidate_edges": int(masks.size(1)),
        "mean_jaccard": _mean(item["jaccard"] for item in pair_masks),
        "min_jaccard": min((item["jaccard"] for item in pair_masks), default=1.0),
        "mean_flip_rate": _mean(item["flip_rate"] for item in pair_masks),
        "max_flip_rate": max((item["flip_rate"] for item in pair_masks), default=0.0),
        "mean_js_divergence": _mean(pair_js),
        "max_js_divergence": max(pair_js, default=0.0),
        "retention_mean": float(retentions.mean().item()),
        "retention_std": float(retentions.std(unbiased=False).item()),
    }


def distribution_summary(values: Iterable[float]) -> Dict[str, float]:
    materialized = list(float(value) for value in values)
    if not materialized:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p10": 0.0, "median": 0.0, "max": 0.0}
    tensor = torch.tensor(materialized, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "median": float(torch.quantile(tensor, 0.50).item()),
        "max": float(tensor.max().item()),
    }


def aggregate_stability_records(records: List[Dict]) -> Dict:
    """按 session 等权汇总，避免候选边较多的会话主导诊断。"""
    result = {"count": len(records)}
    for key in STABILITY_METRIC_KEYS:
        result[key] = distribution_summary(record[key] for record in records)
    result["unstable_fraction"] = (
        _mean(
            float(
                record["mean_jaccard"] < 0.95
                or record["mean_flip_rate"] > 0.05
            )
            for record in records
        )
        if records
        else 0.0
    )
    return result


def stability_decision(summary: Dict) -> Dict:
    mean_jaccard = float(summary["mean_jaccard"]["mean"])
    mean_flip_rate = float(summary["mean_flip_rate"]["mean"])
    worth_training = mean_jaccard < 0.95 or mean_flip_rate > 0.05
    highly_stable = mean_jaccard >= 0.98 and mean_flip_rate <= 0.02
    if worth_training:
        recommendation = "train_selector_consistency"
    elif highly_stable:
        recommendation = "stop_selector_consistency"
    else:
        recommendation = "borderline_review_subgroups"
    return {
        "thresholds": {
            "mean_jaccard_below": 0.95,
            "mean_flip_rate_above": 0.05,
            "high_stability_jaccard_at_least": 0.98,
            "high_stability_flip_rate_at_most": 0.02,
        },
        "worth_training": worth_training,
        "highly_stable": highly_stable,
        "recommendation": recommendation,
    }


def _batch_records(batch, hypergraph, masks: Tensor, scores: Tensor) -> Tuple[List[Dict], List[Dict]]:
    session_records: List[Dict] = []
    type_records: List[Dict] = []
    for session_position in range(batch.num_graphs):
        session_mask = hypergraph.edge_to_session == session_position
        session_indices = torch.nonzero(session_mask, as_tuple=False).flatten()
        record = {
            "session_index": int(batch.session_indices[session_position].item()),
            "label": int(batch.labels[session_position].item()),
            **summarize_view_group(masks[:, session_indices], scores[:, session_indices]),
        }
        session_records.append(record)

        present_types = torch.unique(hypergraph.type_ids[session_indices]).tolist()
        for type_id in present_types:
            type_indices = session_indices[
                hypergraph.type_ids[session_indices] == int(type_id)
            ]
            type_records.append(
                {
                    "session_index": record["session_index"],
                    "label": record["label"],
                    "type_id": int(type_id),
                    "type": BehaviorType(int(type_id)).label,
                    **summarize_view_group(masks[:, type_indices], scores[:, type_indices]),
                }
            )
    return session_records, type_records


@torch.no_grad()
def run_selector_probe(
    run_dir: Path,
    base_config: ExperimentConfig,
    device: torch.device,
    batch_size_override: int | None,
    passes: int,
    probe_seed_override: int | None,
    no_progress: bool,
) -> Dict:
    if passes < 2:
        raise ValueError("passes must be at least 2.")
    result = _load_json(run_dir / "result.json")
    split = _load_json(run_dir / "split.json")
    if result["mode"] != "bbp" or result["budget_mode"] != "robust_adaptive":
        raise ValueError("Selector stability probe requires a robust_adaptive BBP run.")
    if split["processed_sha256"].upper() != base_config.data.processed_sha256.upper():
        raise ValueError("Split data hash does not match the supplied local config.")

    config = _run_config(base_config, result["resolved_config"])
    batch_size = config.training.batch_size if batch_size_override is None else batch_size_override
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    validation = validation_indices(split)

    dataset = PreprocessedSessionDataset(
        base_config.data.processed_path,
        base_config.data.processed_sha256,
        base_config.data.expected_sessions,
        base_config.data.expected_nodes,
        base_config.data.expected_features,
    )
    priors = class_priors(dataset, split["train"], config.model.num_classes)
    confidence_class_means, confidence_global_mean = confidence_statistics(
        dataset, split["train"], config.model.num_classes
    )
    model = BBPBeHM(
        config,
        priors,
        confidence_class_means=confidence_class_means,
        confidence_global_mean=confidence_global_mean,
    )
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    # 只打开 selector scorer dropout；backbone projection 与其他模块保持 eval。
    model.selector.train()
    if model.training or model.backbone.training or not model.selector.training:
        raise RuntimeError("Selector-only dropout isolation was not established.")

    builder = HyperedgeBuilder(config.hypergraph)
    validation_loader = make_loader(
        dataset, validation, batch_size, False, int(result["seed"])
    )
    probe_seed = int(result["seed"]) if probe_seed_override is None else probe_seed_override
    set_seed(probe_seed)
    started = time.perf_counter()
    session_records: List[Dict] = []
    type_records: List[Dict] = []
    progress = tqdm(
        validation_loader,
        desc="Selector stability validation",
        unit="batch",
        dynamic_ncols=True,
        disable=no_progress,
    )
    for cpu_batch in progress:
        batch = cpu_batch.to(device)
        hypergraph = builder.build_batch(batch)
        projected_nodes = model.backbone.project_nodes(batch.x)
        selections = [
            model.selector(
                projected_nodes=projected_nodes,
                batch_x=batch.x,
                graph_ids=batch.graph_ids,
                num_sessions=batch.num_graphs,
                hypergraph=hypergraph,
                retention_ratio=float(result["retention_ratio"]),
                gumbel_temperature=float(config.pruning.gumbel_temperature),
                stochastic=False,
                budget_mode=result["budget_mode"],
                coverage_threshold=float(result["coverage_threshold"]),
                min_retention_ratio=float(result["min_retention_ratio"]),
                max_retention_ratio=float(result["max_retention_ratio"]),
                coverage_q=float(result["coverage_q"]),
                coverage_tail_gamma=float(result["coverage_tail_gamma"]),
                coverage_eta=float(result["coverage_eta"]),
            )
            for _ in range(passes)
        ]
        # 保持与 hypergraph 索引同设备，汇总函数只输出 Python 数值，不长期保留 GPU tensor。
        masks = torch.stack([selection.hard_mask > 0.5 for selection in selections])
        scores = torch.stack([selection.scores for selection in selections])
        batch_session_records, batch_type_records = _batch_records(
            batch, hypergraph, masks, scores
        )
        session_records.extend(batch_session_records)
        type_records.extend(batch_type_records)

    overall = aggregate_stability_records(session_records)
    by_label = {
        "negative": aggregate_stability_records(
            [record for record in session_records if record["label"] == 0]
        ),
        "positive": aggregate_stability_records(
            [record for record in session_records if record["label"] == 1]
        ),
    }
    by_type = {
        behavior_type.label: aggregate_stability_records(
            [record for record in type_records if record["type_id"] == int(behavior_type)]
        )
        for behavior_type in BehaviorType
    }
    return {
        "_purpose": "Selector-only dropout stability probe on the validation split",
        "protocol": "validation_only_selector_stability",
        "test_evaluated": False,
        "source_run": run_dir.name,
        "source_checkpoint_epoch": int(checkpoint["epoch"]),
        "source_model_strategy": checkpoint.get("model_strategy", "best_single"),
        "seed": int(result["seed"]),
        "probe_seed": int(probe_seed),
        "passes": int(passes),
        "pair_count_per_group": int(passes * (passes - 1) // 2),
        "evaluation_device": str(device),
        "batch_size": int(batch_size),
        "validation_sessions": len(validation),
        "budget_mode": result["budget_mode"],
        "coverage_q": float(result["coverage_q"]),
        "coverage_tail_gamma": float(result["coverage_tail_gamma"]),
        "coverage_eta": float(result["coverage_eta"]),
        "selector_dropout": float(config.model.dropout),
        "isolation": {
            "model_training": bool(model.training),
            "backbone_training": bool(model.backbone.training),
            "selector_training": bool(model.selector.training),
            "stochastic_gumbel": False,
            "backbone_forward_executed": False,
        },
        "overall": overall,
        "by_label": by_label,
        "by_type": by_type,
        "decision": stability_decision(overall),
        "session_records": session_records,
        "session_type_records": type_records,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base_config = load_experiment_config(args.config)
    requested_device = base_config.training.device if args.device is None else args.device
    device = resolve_device(requested_device)
    artifact = run_selector_probe(
        args.run_dir,
        base_config,
        device,
        args.batch_size,
        args.passes,
        args.probe_seed,
        args.no_progress,
    )
    output_path = args.run_dir / args.output_name
    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    overall = artifact["overall"]
    logging.info(
        "Selector stability completed: mean_jaccard=%.6f mean_flip_rate=%.6f "
        "unstable_fraction=%.6f recommendation=%s test_evaluated=false output=%s",
        overall["mean_jaccard"]["mean"],
        overall["mean_flip_rate"]["mean"],
        overall["unstable_fraction"],
        artifact["decision"]["recommendation"],
        output_path,
    )


if __name__ == "__main__":
    main()
