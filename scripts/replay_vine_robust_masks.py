"""用途：只读重放 Vine 检查点，对比当前均值覆盖、分类型均值和鲁棒尾部覆盖掩码。"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch import Tensor
from tqdm.auto import tqdm

from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset
from bbp_behm.hypergraph import HyperedgeBatch, HyperedgeBuilder
from bbp_behm.model import BBPBeHM
from bbp_behm.training import make_loader, resolve_device, set_seed


@dataclass(frozen=True)
class VariantSpec:
    """描述一种只读掩码重放策略。"""

    name: str
    mode: str
    gamma: float
    tail_fraction: float = 0.10
    task_weight: float = 0.50


@dataclass
class TypeContext:
    """保存单个会话内一种实际存在类型的候选边和代表矩阵。"""

    edge_ids: Tensor
    representative: Tensor


def parse_float_list(value: str) -> List[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one floating-point value.")
    return values


def parse_mode_list(value: str) -> List[str]:
    allowed = {"current", "type_mean", "robust_prefix", "robust_greedy"}
    modes = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(modes) - allowed)
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported modes: {', '.join(invalid)}")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay robust Vine pruning masks without training.")
    parser.add_argument("--config", type=Path, default=Path("configs/vine_dedup_cv5_8020_seed42_pw1.5.ini"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-name", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--modes", type=parse_mode_list, default=parse_mode_list("current,type_mean,robust_prefix,robust_greedy"))
    parser.add_argument("--tail-fractions", type=parse_float_list, default=parse_float_list("0.05,0.10,0.20"))
    parser.add_argument("--gamma-values", type=parse_float_list, default=parse_float_list("0.70,0.80,0.90"))
    parser.add_argument("--task-weights", type=parse_float_list, default=parse_float_list("0.50"))
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def lower_tail_mean(values: Tensor, tail_fraction: float) -> Tensor:
    """计算最低覆盖 q% 的均值；空集合按覆盖 1 处理。"""
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1].")
    if values.numel() == 0:
        return values.new_tensor(1.0)
    count = max(1, int(math.ceil(tail_fraction * int(values.numel()))))
    return torch.topk(values, k=count, largest=False, sorted=False).values.mean()


def present_type_coverage(
    best_by_type: Dict[int, Tensor],
    tail_fraction: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """返回实际存在类型的当前均值、最差类型均值和最差类型尾部覆盖。"""
    if not best_by_type:
        one = torch.tensor(1.0)
        return one, one, one
    means = torch.stack([values.mean() for values in best_by_type.values()])
    tails = torch.stack([lower_tail_mean(values, tail_fraction) for values in best_by_type.values()])
    return means.mean(), means.min(), tails.min()


def normalize_scores(scores: Tensor) -> Tensor:
    """在单会话内把任务分数缩放到 [0, 1]，常量分数退化为零。"""
    if scores.numel() == 0:
        return scores
    lower = scores.min()
    span = scores.max() - lower
    if float(span.detach().cpu()) <= 1e-12:
        return torch.zeros_like(scores)
    return (scores - lower) / span


def build_type_contexts(
    model: BBPBeHM,
    hypergraph: HyperedgeBatch,
    stable_embeddings: Tensor,
    session_edge_ids: Tensor,
) -> Dict[int, TypeContext]:
    """只为当前会话实际存在的类型构造代表矩阵。"""
    contexts: Dict[int, TypeContext] = {}
    if session_edge_ids.numel() == 0:
        return contexts
    present_types = torch.unique(hypergraph.type_ids[session_edge_ids]).tolist()
    for type_id in present_types:
        local = session_edge_ids[hypergraph.type_ids[session_edge_ids] == int(type_id)]
        representative = model.selector._representativeness_matrix(
            hypergraph.incidence[local],
            stable_embeddings[local],
            hypergraph.structural_stats[local],
        )
        contexts[int(type_id)] = TypeContext(local, representative)
    return contexts


def initialize_best(contexts: Dict[int, TypeContext]) -> Dict[int, Tensor]:
    return {
        type_id: context.representative.new_zeros((context.representative.size(0),))
        for type_id, context in contexts.items()
    }


def edge_locations(contexts: Dict[int, TypeContext]) -> Dict[int, tuple[int, int]]:
    result: Dict[int, tuple[int, int]] = {}
    for type_id, context in contexts.items():
        for position, edge_id in enumerate(context.edge_ids.tolist()):
            result[int(edge_id)] = (type_id, position)
    return result


def criterion_value(
    best_by_type: Dict[int, Tensor],
    mode: str,
    tail_fraction: float,
) -> Tensor:
    current_mean, type_mean, robust_tail = present_type_coverage(best_by_type, tail_fraction)
    if mode == "current":
        return current_mean
    if mode == "type_mean":
        return type_mean
    if mode in {"robust_prefix", "robust_greedy"}:
        return robust_tail
    raise ValueError(f"Unsupported selection mode: {mode}")


def keep_bounds(count: int, min_ratio: float, max_ratio: float) -> tuple[int, int]:
    if count == 0:
        return 0, 0
    minimum = max(1, min(count, int(math.ceil(min_ratio * count))))
    maximum = max(minimum, min(count, int(math.ceil(max_ratio * count))))
    return minimum, maximum


def select_score_prefix(
    scores: Tensor,
    session_edge_ids: Tensor,
    contexts: Dict[int, TypeContext],
    mode: str,
    gamma: float,
    tail_fraction: float,
    min_ratio: float,
    max_ratio: float,
) -> tuple[Tensor, float, str]:
    """沿任务评分前缀选择，分别支持当前、类型均值和鲁棒尾部停止准则。"""
    count = int(session_edge_ids.numel())
    mask = torch.zeros_like(scores)
    if count == 0:
        return mask, 1.0, "no_edges"
    minimum, maximum = keep_bounds(count, min_ratio, max_ratio)
    ranked = session_edge_ids[torch.argsort(scores[session_edge_ids], descending=True)]
    locations = edge_locations(contexts)
    best = initialize_best(contexts)
    value = scores.new_tensor(0.0)
    kept = maximum
    reason = "maximum"
    for prefix, edge_id in enumerate(ranked.tolist(), start=1):
        type_id, position = locations[int(edge_id)]
        best[type_id] = torch.maximum(
            best[type_id], contexts[type_id].representative[:, position]
        )
        value = criterion_value(best, mode, tail_fraction)
        if prefix >= minimum and float(value.detach().cpu()) >= gamma:
            kept = prefix
            reason = "coverage_full" if prefix == count else "coverage"
            break
        if prefix >= maximum:
            kept = maximum
            reason = "all_edges" if maximum == count else "maximum"
            break
    mask[ranked[:kept]] = 1.0
    return mask, float(value.detach().cpu()), reason


def candidate_robust_gains(
    contexts: Dict[int, TypeContext],
    best: Dict[int, Tensor],
    tail_fraction: float,
    gamma: float,
    total_edges: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """计算每条候选边对分类型尾部覆盖缺口的边际改善。"""
    gains = torch.zeros((total_edges,), dtype=dtype, device=device)
    type_count = max(len(contexts), 1)
    for type_id, context in contexts.items():
        current = lower_tail_mean(best[type_id], tail_fraction)
        updated = torch.maximum(best[type_id][:, None], context.representative)
        tail_count = max(1, int(math.ceil(tail_fraction * updated.size(0))))
        candidate_tail = torch.topk(
            updated, k=tail_count, dim=0, largest=False, sorted=False
        ).values.mean(dim=0)
        current_progress = torch.clamp(current / gamma, max=1.0)
        candidate_progress = torch.clamp(candidate_tail / gamma, max=1.0)
        gains[context.edge_ids] = (candidate_progress - current_progress) / type_count
    return gains


def select_robust_greedy(
    scores: Tensor,
    session_edge_ids: Tensor,
    contexts: Dict[int, TypeContext],
    gamma: float,
    tail_fraction: float,
    task_weight: float,
    min_ratio: float,
    max_ratio: float,
) -> tuple[Tensor, float, str]:
    """联合任务评分和尾部覆盖边际，构造满足鲁棒约束的代表集合。"""
    if not 0.0 <= task_weight <= 1.0:
        raise ValueError("task_weight must be in [0, 1].")
    count = int(session_edge_ids.numel())
    mask = torch.zeros_like(scores)
    if count == 0:
        return mask, 1.0, "no_edges"
    minimum, maximum = keep_bounds(count, min_ratio, max_ratio)
    locations = edge_locations(contexts)
    best = initialize_best(contexts)
    local_task = normalize_scores(scores[session_edge_ids])
    selected = torch.zeros((count,), dtype=torch.bool, device=scores.device)
    value = scores.new_tensor(0.0)
    reason = "maximum"

    for step in range(1, maximum + 1):
        raw_gains = candidate_robust_gains(
            contexts,
            best,
            tail_fraction,
            gamma,
            scores.numel(),
            scores.dtype,
            scores.device,
        )[session_edge_ids]
        gain_score = normalize_scores(raw_gains)
        utility = task_weight * local_task + (1.0 - task_weight) * gain_score
        utility = utility.masked_fill(selected, -torch.inf)
        chosen_local = int(torch.argmax(utility).item())
        selected[chosen_local] = True
        chosen_edge = int(session_edge_ids[chosen_local].item())
        type_id, position = locations[chosen_edge]
        best[type_id] = torch.maximum(
            best[type_id], contexts[type_id].representative[:, position]
        )
        value = criterion_value(best, "robust_greedy", tail_fraction)
        if step >= minimum and float(value.detach().cpu()) >= gamma:
            reason = "coverage_full" if step == count else "coverage"
            break
        if step == maximum:
            reason = "all_edges" if maximum == count else "maximum"
    mask[session_edge_ids[selected]] = 1.0
    return mask, float(value.detach().cpu()), reason


def mask_diagnostics(
    contexts: Dict[int, TypeContext],
    mask: Tensor,
    tail_fraction: float,
) -> dict:
    """统计一种掩码的类型覆盖和逐边最佳代表相似度。"""
    if not contexts:
        return {
            "current_mean": 1.0,
            "type_mean_min": 1.0,
            "robust_tail_min": 1.0,
            "below_0_5": 0,
            "below_0_7": 0,
            "below_0_9": 0,
        }
    best: Dict[int, Tensor] = {}
    all_values: List[Tensor] = []
    for type_id, context in contexts.items():
        kept = mask[context.edge_ids] > 0.5
        values = (
            context.representative[:, kept].max(dim=1).values
            if bool(kept.any())
            else context.representative.new_zeros((context.representative.size(0),))
        )
        best[type_id] = values
        all_values.append(values)
    current_mean, type_mean, robust_tail = present_type_coverage(best, tail_fraction)
    flat = torch.cat(all_values)
    return {
        "current_mean": float(current_mean.detach().cpu()),
        "type_mean_min": float(type_mean.detach().cpu()),
        "robust_tail_min": float(robust_tail.detach().cpu()),
        "below_0_5": int((flat < 0.5).sum().item()),
        "below_0_7": int((flat < 0.7).sum().item()),
        "below_0_9": int((flat < 0.9).sum().item()),
    }


def make_variants(args: argparse.Namespace, configured_gamma: float) -> List[VariantSpec]:
    variants: List[VariantSpec] = []
    if "current" in args.modes:
        variants.append(VariantSpec("current_mean", "current", configured_gamma))
    if "type_mean" in args.modes:
        variants.extend(
            VariantSpec(f"type_mean_gamma{gamma:g}", "type_mean", gamma)
            for gamma in args.gamma_values
        )
    for tail_fraction in args.tail_fractions:
        for gamma in args.gamma_values:
            if "robust_prefix" in args.modes:
                variants.append(
                    VariantSpec(
                        f"robust_prefix_q{tail_fraction:g}_gamma{gamma:g}",
                        "robust_prefix",
                        gamma,
                        tail_fraction,
                    )
                )
            if "robust_greedy" in args.modes:
                variants.extend(
                    VariantSpec(
                        f"robust_greedy_q{tail_fraction:g}_gamma{gamma:g}_eta{task_weight:g}",
                        "robust_greedy",
                        gamma,
                        tail_fraction,
                        task_weight,
                    )
                    for task_weight in args.task_weights
                )
    names = [variant.name for variant in variants]
    if len(names) != len(set(names)):
        raise ValueError("Variant names must be unique.")
    return variants


def classification_summary(labels: Sequence[int], probabilities: Sequence[float]) -> dict:
    if not labels:
        return {}
    predictions = [int(value >= 0.5) for value in probabilities]
    result = {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
    }
    result["auroc"] = (
        float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None
    )
    return result


def summarize_variant(accumulator: dict, structure_only: bool) -> dict:
    sessions = max(accumulator["sessions"], 1)
    candidates = max(accumulator["candidate_edges"], 1)
    result = {
        "sessions": accumulator["sessions"],
        "candidate_edges": accumulator["candidate_edges"],
        "kept_edges": accumulator["kept_edges"],
        "edge_retention": accumulator["kept_edges"] / candidates,
        "session_retention_mean": float(np.mean(accumulator["session_retentions"])),
        "current_mean_coverage": float(np.mean(accumulator["current_means"])),
        "minimum_type_mean_coverage": float(np.mean(accumulator["type_mean_mins"])),
        "minimum_tail_coverage": float(np.mean(accumulator["tail_mins"])),
        "coverage_satisfied_fraction": accumulator["satisfied"] / sessions,
        "best_similarity_below_0_5_fraction": accumulator["below_0_5"] / candidates,
        "best_similarity_below_0_7_fraction": accumulator["below_0_7"] / candidates,
        "best_similarity_below_0_9_fraction": accumulator["below_0_9"] / candidates,
        "stopping_reasons": dict(accumulator["reasons"]),
    }
    if not structure_only:
        result.update(classification_summary(accumulator["labels"], accumulator["probabilities"]))
    return result


def new_accumulator() -> dict:
    from collections import Counter

    return {
        "sessions": 0,
        "candidate_edges": 0,
        "kept_edges": 0,
        "session_retentions": [],
        "current_means": [],
        "type_mean_mins": [],
        "tail_mins": [],
        "below_0_5": 0,
        "below_0_7": 0,
        "below_0_9": 0,
        "satisfied": 0,
        "reasons": Counter(),
        "labels": [],
        "probabilities": [],
    }


def replay(args: argparse.Namespace) -> dict:
    config = load_experiment_config(args.config)
    set_seed(args.seed)
    device = resolve_device(args.device or config.training.device)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    indices = list(split[args.split_name])
    if args.max_sessions is not None:
        if args.max_sessions <= 0:
            raise ValueError("max_sessions must be positive.")
        indices = indices[: args.max_sessions]
    dataset = PreprocessedSessionDataset(
        config.data.processed_path,
        config.data.processed_sha256,
        config.data.expected_sessions,
        config.data.expected_nodes,
        config.data.expected_features,
    )
    loader = make_loader(dataset, indices, args.batch_size, False, args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model = BBPBeHM(config, torch.tensor([0.5, 0.5])).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    builder = HyperedgeBuilder(config.hypergraph)
    variants = make_variants(args, config.pruning.coverage_threshold)
    accumulators = {variant.name: new_accumulator() for variant in variants}
    logging.info(
        "Replay started: split=%s sessions=%d variants=%d device=%s structure_only=%s",
        args.split_name,
        len(indices),
        len(variants),
        device,
        args.structure_only,
    )

    iterator: Iterable = tqdm(
        loader,
        desc="robust-mask replay",
        unit="batch",
        disable=args.no_progress,
    )
    with torch.no_grad():
        for cpu_batch in iterator:
            batch = cpu_batch.to(device)
            hypergraph = builder.build_batch(batch)
            projected_nodes = model.backbone.project_nodes(batch.x)
            scores, _, _, _ = model.selector.score_edges(
                projected_nodes,
                batch.graph_ids,
                batch.num_graphs,
                hypergraph,
            )
            stable_embeddings = model.selector.encode_edges_raw(batch.x, hypergraph.incidence)
            masks = {
                variant.name: hypergraph.incidence.new_zeros((hypergraph.num_edges,))
                for variant in variants
            }
            values: Dict[str, Dict[int, tuple[float, str]]] = {
                variant.name: {} for variant in variants
            }
            contexts_by_session: Dict[int, Dict[int, TypeContext]] = {}

            for session in range(batch.num_graphs):
                session_edges = torch.nonzero(
                    hypergraph.edge_to_session == session, as_tuple=False
                ).flatten()
                contexts = build_type_contexts(
                    model, hypergraph, stable_embeddings, session_edges
                )
                contexts_by_session[session] = contexts
                for variant in variants:
                    if variant.mode == "robust_greedy":
                        mask, value, reason = select_robust_greedy(
                            scores,
                            session_edges,
                            contexts,
                            variant.gamma,
                            variant.tail_fraction,
                            variant.task_weight,
                            config.pruning.min_retention_ratio,
                            config.pruning.max_retention_ratio,
                        )
                    else:
                        mask, value, reason = select_score_prefix(
                            scores,
                            session_edges,
                            contexts,
                            variant.mode,
                            variant.gamma,
                            variant.tail_fraction,
                            config.pruning.min_retention_ratio,
                            config.pruning.max_retention_ratio,
                        )
                    masks[variant.name] += mask
                    values[variant.name][session] = (value, reason)

            for variant in variants:
                mask = masks[variant.name]
                output = None
                if not args.structure_only:
                    masked_incidence = hypergraph.incidence * mask[:, None]
                    output = model.backbone.forward_projected(
                        projected_nodes,
                        masked_incidence,
                        batch.graph_ids,
                        hypergraph.edge_to_session,
                        mask,
                        batch.num_graphs,
                        labels=None,
                        confidences=None,
                    )
                    probabilities = F.softmax(output.logits, dim=1)[:, 1]

                accumulator = accumulators[variant.name]
                for session in range(batch.num_graphs):
                    session_edges = torch.nonzero(
                        hypergraph.edge_to_session == session, as_tuple=False
                    ).flatten()
                    diagnostics = mask_diagnostics(
                        contexts_by_session[session], mask, variant.tail_fraction
                    )
                    candidate_count = int(session_edges.numel())
                    kept_count = int((mask[session_edges] > 0.5).sum().item())
                    value, reason = values[variant.name][session]
                    accumulator["sessions"] += 1
                    accumulator["candidate_edges"] += candidate_count
                    accumulator["kept_edges"] += kept_count
                    accumulator["session_retentions"].append(
                        kept_count / candidate_count if candidate_count else 1.0
                    )
                    accumulator["current_means"].append(diagnostics["current_mean"])
                    accumulator["type_mean_mins"].append(diagnostics["type_mean_min"])
                    accumulator["tail_mins"].append(diagnostics["robust_tail_min"])
                    accumulator["below_0_5"] += diagnostics["below_0_5"]
                    accumulator["below_0_7"] += diagnostics["below_0_7"]
                    accumulator["below_0_9"] += diagnostics["below_0_9"]
                    accumulator["satisfied"] += int(value >= variant.gamma)
                    accumulator["reasons"][reason] += 1
                    if output is not None:
                        accumulator["labels"].append(int(batch.labels[session].item()))
                        accumulator["probabilities"].append(
                            float(probabilities[session].detach().cpu())
                        )

    report = {
        "_purpose": "Read-only Vine robust-mask replay for approval gate A",
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "split_name": args.split_name,
        "seed": args.seed,
        "sessions": len(indices),
        "structure_only": args.structure_only,
        "important_scope": (
            "This reuses a checkpoint trained with current mean coverage. It is a mask counterfactual, "
            "not a retrained robust-selector result. Validation may be used for selection; test must not be tuned."
        ),
        "variants": {
            variant.name: {
                "mode": variant.mode,
                "gamma": variant.gamma,
                "tail_fraction": variant.tail_fraction,
                "task_weight": variant.task_weight,
                **summarize_variant(accumulators[variant.name], args.structure_only),
            }
            for variant in variants
        },
    }
    logging.info("Replay completed: split=%s variants=%d", args.split_name, len(variants))
    return report


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = replay(args)
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        logging.info("Replay report written: %s", args.output)


if __name__ == "__main__":
    main()
