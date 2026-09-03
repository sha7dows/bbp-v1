"""用途：用旧 Vine 检查点诊断正类召回不足，比较阈值、logit 调整和推理时剪枝反事实。"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset
from bbp_behm.hypergraph import BehaviorType, HyperedgeBuilder
from bbp_behm.model import BBPBeHM
from bbp_behm.training import make_loader, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose recall of a trained Vine BBP-BeHM checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/vine_dedup_cv5_8020_seed42_pw1.5.ini"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/vine_bbp_seed42_rho0.5/final_model.pt")
    )
    parser.add_argument("--split", type=Path, default=Path("outputs/vine_bbp_seed42_rho0.5/split.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/vine_bbp_seed42_recall_diagnostics.json"),
    )
    parser.add_argument("--retention-ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def classification_metrics(labels: list[int], probabilities: list[float], threshold: float) -> dict:
    predictions = [int(probability >= threshold) for probability in probabilities]
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "predicted_positive": int(sum(predictions)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_diagnostics(labels: list[int], probabilities: list[float]) -> dict:
    candidates = np.linspace(0.05, 0.95, 181)
    rows = [classification_metrics(labels, probabilities, float(threshold)) for threshold in candidates]
    best_macro = max(rows, key=lambda row: (row["macro_f1"], row["threshold"]))
    best_positive = max(rows, key=lambda row: (row["positive_f1"], row["threshold"]))

    def best_with_recall(target: float) -> dict | None:
        feasible = [row for row in rows if row["recall"] >= target]
        return max(feasible, key=lambda row: (row["precision"], row["threshold"])) if feasible else None

    return {
        "fixed_0.5": classification_metrics(labels, probabilities, 0.5),
        # 以下最优阈值使用测试标签，仅用于定位问题，不能作为最终报告结果。
        "test_oracle_best_macro_f1": best_macro,
        "test_oracle_best_positive_f1": best_positive,
        "test_oracle_recall_at_least_0.7": best_with_recall(0.7),
        "test_oracle_recall_at_least_0.8": best_with_recall(0.8),
    }


def calibration(labels: list[int], probabilities: list[float], bins: int = 10) -> dict:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    ece = 0.0
    bin_rows = []
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (p >= lower) & (p < upper if upper < 1.0 else p <= upper)
        if not mask.any():
            continue
        confidence = float(p[mask].mean())
        positive_rate = float(y[mask].mean())
        fraction = float(mask.mean())
        ece += fraction * abs(confidence - positive_rate)
        bin_rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(mask.sum()),
                "mean_probability": confidence,
                "positive_rate": positive_rate,
            }
        )
    return {"brier": brier, "ece_10_bin": float(ece), "bins": bin_rows}


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
    }


def collect_predictions(
    model: BBPBeHM,
    loader,
    builder: HyperedgeBuilder,
    device: torch.device,
    retention_ratio: float,
    pruning_enabled: bool,
) -> list[dict]:
    model.eval()
    rows = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            hypergraph = builder.build_batch(batch)
            output = model(
                batch,
                hypergraph,
                retention_ratio=retention_ratio,
                pruning_enabled=pruning_enabled,
                stochastic_selection=False,
                # 该脚本专门重放上一版固定 rho 检查点，不能继承新版 adaptive 默认值。
                budget_mode="fixed",
            )
            adjusted_probability = F.softmax(output.logits, dim=1)[:, 1]
            scale = F.softplus(model.backbone.raw_adjustment)
            if model.backbone.logit_adjustment == "paper":
                raw_logits = output.logits + scale * torch.log(model.backbone.class_priors.clamp_min(1e-8))
            elif model.backbone.logit_adjustment == "legacy":
                raw_logits = output.logits - scale * model.backbone.class_priors
            else:
                raw_logits = output.logits
            raw_probability = F.softmax(raw_logits, dim=1)[:, 1]
            hard_mask = (
                torch.ones(hypergraph.num_edges, device=device)
                if output.selection is None
                else output.selection.hard_mask
            )

            for position in range(batch.num_graphs):
                edge_ids = torch.nonzero(
                    hypergraph.edge_to_session == position, as_tuple=False
                ).flatten()
                kept_ids = edge_ids[hard_mask[edge_ids] > 0.5]
                node_start = int(batch.ptr[position])
                node_end = int(batch.ptr[position + 1])
                type_candidate = {}
                type_kept = {}
                for behavior in BehaviorType:
                    local = edge_ids[hypergraph.type_ids[edge_ids] == int(behavior)]
                    type_candidate[behavior.label] = int(local.numel())
                    type_kept[behavior.label] = int((hard_mask[local] > 0.5).sum())
                stc_records = [
                    hypergraph.records[int(edge_id)]
                    for edge_id in edge_ids.tolist()
                    if hypergraph.records[int(edge_id)].behavior_type == BehaviorType.STC
                ]
                kept_stc_sizes = [
                    hypergraph.records[int(edge_id)].size
                    for edge_id in kept_ids.tolist()
                    if hypergraph.records[int(edge_id)].behavior_type == BehaviorType.STC
                ]
                candidate_coverage = int(
                    (hypergraph.incidence[edge_ids].sum(dim=0) > 0).sum().cpu()
                ) if edge_ids.numel() else 0
                kept_coverage = int(
                    (hypergraph.incidence[kept_ids].sum(dim=0) > 0).sum().cpu()
                ) if kept_ids.numel() else 0
                rows.append(
                    {
                        "session_index": int(batch.session_indices[position].cpu()),
                        "session_id": str(batch.session_ids[position]),
                        "label": int(batch.labels[position].cpu()),
                        "confidence": float(batch.confidences[position].cpu()),
                        "probability": float(adjusted_probability[position].cpu()),
                        "raw_probability_without_logit_adjustment": float(raw_probability[position].cpu()),
                        "prediction": int(adjusted_probability[position] >= 0.5),
                        "nodes": node_end - node_start,
                        "candidate_edges": int(edge_ids.numel()),
                        "kept_edges": int(kept_ids.numel()),
                        "candidate_incidence": float(hypergraph.incidence[edge_ids].sum().cpu()),
                        "kept_incidence": float(hypergraph.incidence[kept_ids].sum().cpu()),
                        "candidate_node_coverage": candidate_coverage,
                        "kept_node_coverage": kept_coverage,
                        "candidate_by_type": type_candidate,
                        "kept_by_type": type_kept,
                        "candidate_stc_mean_size": (
                            float(np.mean([record.size for record in stc_records])) if stc_records else 0.0
                        ),
                        "candidate_stc_max_size": max((record.size for record in stc_records), default=0),
                        "kept_stc_mean_size": float(np.mean(kept_stc_sizes)) if kept_stc_sizes else 0.0,
                        "kept_stc_max_size": max(kept_stc_sizes, default=0),
                    }
                )
    return rows


def group_comparison(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if row["label"] == 1:
            group = "TP" if row["prediction"] == 1 else "FN"
        else:
            group = "FP" if row["prediction"] == 1 else "TN"
        groups[group].append(row)
    numeric_fields = [
        "probability",
        "raw_probability_without_logit_adjustment",
        "confidence",
        "nodes",
        "candidate_edges",
        "kept_edges",
        "candidate_incidence",
        "kept_incidence",
        "candidate_node_coverage",
        "kept_node_coverage",
        "candidate_stc_mean_size",
        "candidate_stc_max_size",
        "kept_stc_mean_size",
        "kept_stc_max_size",
    ]
    summaries = {
        group: {field: summarize([float(row[field]) for row in members]) for field in numeric_fields}
        for group, members in groups.items()
    }
    tests = {}
    for field in numeric_fields:
        tp_values = [float(row[field]) for row in groups["TP"]]
        fn_values = [float(row[field]) for row in groups["FN"]]
        if tp_values and fn_values:
            test = mannwhitneyu(tp_values, fn_values, alternative="two-sided")
            tests[field] = {"u": float(test.statistic), "p_uncorrected": float(test.pvalue)}
    return {"summaries": summaries, "tp_vs_fn_mann_whitney": tests}


def pruning_counterfactual(pruned: list[dict], unpruned: list[dict]) -> dict:
    unpruned_by_id = {row["session_index"]: row for row in unpruned}
    transitions = defaultdict(int)
    positive_deltas = []
    recovered = []
    harmed = []
    for row in pruned:
        other = unpruned_by_id[row["session_index"]]
        transitions[f"{row['prediction']}->{other['prediction']}"] += 1
        delta = other["probability"] - row["probability"]
        if row["label"] == 1:
            positive_deltas.append(delta)
            if row["prediction"] == 0 and other["prediction"] == 1:
                recovered.append({"session_index": row["session_index"], "delta": delta})
            if row["prediction"] == 1 and other["prediction"] == 0:
                harmed.append({"session_index": row["session_index"], "delta": delta})
    return {
        "transition_counts_all": dict(transitions),
        "positive_probability_delta_unpruned_minus_pruned": summarize(positive_deltas),
        "positive_fn_recovered_without_pruning": recovered,
        "positive_tp_lost_without_pruning": harmed,
    }


def correlations_on_positive(rows: list[dict]) -> dict:
    positives = [row for row in rows if row["label"] == 1]
    missed = np.asarray([1 - row["prediction"] for row in positives], dtype=float)
    fields = [
        "confidence",
        "nodes",
        "candidate_edges",
        "kept_edges",
        "candidate_incidence",
        "kept_incidence",
        "candidate_stc_mean_size",
        "candidate_stc_max_size",
        "kept_stc_mean_size",
        "kept_stc_max_size",
    ]
    result = {}
    for field in fields:
        values = np.asarray([row[field] for row in positives], dtype=float)
        statistic = float(spearmanr(values, missed).statistic)
        result[field] = statistic if math.isfinite(statistic) else None
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_experiment_config(args.config)
    set_seed(42)
    device = resolve_device(config.training.device)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    dataset = PreprocessedSessionDataset(
        config.data.processed_path,
        config.data.processed_sha256,
        config.data.expected_sessions,
        config.data.expected_nodes,
        config.data.expected_features,
    )
    loader = make_loader(dataset, split["test"], args.batch_size, False, 42)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = BBPBeHM(config, torch.tensor([0.5, 0.5])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    builder = HyperedgeBuilder(config.hypergraph)

    logging.info("Collecting pruned predictions")
    pruned = collect_predictions(model, loader, builder, device, args.retention_ratio, True)
    # 重新创建加载器，避免依赖迭代器状态。
    loader = make_loader(dataset, split["test"], args.batch_size, False, 42)
    logging.info("Collecting inference-time unpruned counterfactual predictions")
    unpruned = collect_predictions(model, loader, builder, device, 1.0, False)

    labels = [row["label"] for row in pruned]
    probabilities = [row["probability"] for row in pruned]
    raw_probabilities = [row["raw_probability_without_logit_adjustment"] for row in pruned]
    unpruned_probabilities = [row["probability"] for row in unpruned]
    scale = float(F.softplus(model.backbone.raw_adjustment).detach().cpu())
    priors = model.backbone.class_priors.detach().cpu().tolist()
    relative_positive_shift = scale * math.log(priors[0] / priors[1])
    report = {
        "_purpose": "Recall-centered diagnosis of the old 40-epoch Vine BBP-BeHM checkpoint",
        "important_scope": (
            "Threshold optima use test labels only for diagnosis. The unpruned path uses a pruning-trained model "
            "and is an inference counterfactual, not a retrained BeHM baseline."
        ),
        "class_priors": priors,
        "logit_adjustment_scale": scale,
        "relative_positive_logit_shift": relative_positive_shift,
        "pruned_adjusted": {
            "auroc": float(roc_auc_score(labels, probabilities)),
            "thresholds": threshold_diagnostics(labels, probabilities),
            "calibration": calibration(labels, probabilities),
        },
        "pruned_without_logit_adjustment": {
            "auroc": float(roc_auc_score(labels, raw_probabilities)),
            "thresholds": threshold_diagnostics(labels, raw_probabilities),
            "calibration": calibration(labels, raw_probabilities),
        },
        "unpruned_adjusted_counterfactual": {
            "auroc": float(roc_auc_score(labels, unpruned_probabilities)),
            "thresholds": threshold_diagnostics(labels, unpruned_probabilities),
            "calibration": calibration(labels, unpruned_probabilities),
        },
        "pruned_group_comparison": group_comparison(pruned),
        "positive_miss_correlations": correlations_on_positive(pruned),
        "pruning_counterfactual": pruning_counterfactual(pruned, unpruned),
        "sessions": pruned,
        "unpruned_sessions": unpruned,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    logging.info("Recall diagnostics written: %s", args.output)


if __name__ == "__main__":
    main()
