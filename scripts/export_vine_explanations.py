"""用途：导出关键边 Top-K 可解释性表（论文案例素材）。

口径（用户确认）：
- 目标会话：验证集中预测为正类且 positive_probability 最高的前 N 个会话（默认 20）；
- 每会话：对全部保留边计算 deletion contribution，目标类别 = 真实标签（边支撑正确分类的程度），取贡献绝对值 Top-K（默认 15）；
- 性能：按会话逐个处理（单会话 batch），Louvain 逐会话跑，避免全批量逐边前向；
- 每行补充超边唯一信息量 u(e)=1-max Jaccard（结构事实，与贡献互补）。

不读取测试集，不训练。输出：outputs/explain_vine/<name>.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset, collate_sessions
from bbp_behm.explain import build_explanation_rows, deletion_contributions, export_jsonl, representative_assignments
from bbp_behm.hypergraph import BehaviorType, HyperedgeBuilder
from bbp_behm.model import BBPBeHM
from bbp_behm.training import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出关键边 Top-K 可解释性表（验证集，不读测试集）。")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--processed-path", type=Path, required=True)
    parser.add_argument("--n-sessions", type=int, default=20, help="预测正类概率最高的会话数。")
    parser.add_argument("--top-k", type=int, default=15, help="每会话贡献绝对值 Top-K 条边。")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/explain_vine/explain_topk.jsonl"))
    return parser.parse_args()


def session_uniqueness(records, incidence: torch.Tensor) -> dict[int, float]:
    """会话内每条边的 u(e)=1-max Jaccard；返回 {hyperedge_id: u}。"""
    if incidence.size(0) == 0:
        return {}
    intersection = incidence @ incidence.t()
    sizes = incidence.sum(dim=1, keepdim=True)
    union = (sizes + sizes.t() - intersection).clamp_min(1e-8)
    jaccard = intersection / union
    jaccard = jaccard - torch.diag_embed(torch.diagonal(jaccard))
    max_other = jaccard.max(dim=1).values.clamp(0.0, 1.0)
    u_values = (1.0 - max_other).clamp(0.0, 1.0)
    return {record.hyperedge_id: float(u_values[i].item()) for i, record in enumerate(records)}


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    config = replace(config, data=replace(config.data, processed_path=args.processed_path))
    set_seed(42)
    device = resolve_device(args.device)

    split = json.loads(args.split.read_text(encoding="utf-8"))
    validation_indices = list(split["validation"])
    dataset = PreprocessedSessionDataset(
        config.data.processed_path,
        config.data.processed_sha256,
        config.data.expected_sessions,
        config.data.expected_nodes,
        config.data.expected_features,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model = BBPBeHM(config, torch.tensor([0.5, 0.5])).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    builder = HyperedgeBuilder(config.hypergraph)

    # 阶段 1：全验证集前向，收集每个会话的正类概率与标签，选 Top-N 正类概率会话。
    scored = []  # (positive_prob, session_index, label)
    for index in validation_indices:
        batch = collate_sessions([dataset[index]]).to(device)
        hypergraph = builder.build_batch(batch)
        with torch.no_grad():
            output = model(
                batch,
                hypergraph,
                pruning_enabled=True,
                stochastic_selection=False,
                budget_mode="robust_adaptive",
                coverage_q=config.pruning.coverage_q,
                coverage_tail_gamma=config.pruning.coverage_tail_gamma,
                coverage_eta=config.pruning.coverage_eta,
            )
        prob = float(torch.softmax(output.logits, dim=1)[0, 1].item())
        label = int(batch.labels[0].item())
        scored.append((prob, index, label))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[: args.n_sessions]
    print(
        f"selected {len(selected)} sessions by positive probability: "
        f"prob range [{selected[-1][0]:.4f}, {selected[0][0]:.4f}], "
        f"true-positive count = {sum(1 for _, _, lab in selected if lab == 1)}"
    )

    # 阶段 2：对每个选中会话，逐边计算真实标签概率差贡献，取 Top-K。
    all_rows: list[dict] = []
    timing = {"stage1_seconds": 0.0, "stage2_seconds": 0.0}
    type_counter: Counter = Counter()
    kept_edges_total = 0
    contributions_all: list[float] = []

    t0 = time.perf_counter()
    for rank, (prob, index, label) in enumerate(selected):
        batch = collate_sessions([dataset[index]]).to(device)
        hypergraph = builder.build_batch(batch)
        with torch.no_grad():
            base_output = model(
                batch,
                hypergraph,
                pruning_enabled=True,
                stochastic_selection=False,
                budget_mode="robust_adaptive",
                coverage_q=config.pruning.coverage_q,
                coverage_tail_gamma=config.pruning.coverage_tail_gamma,
                coverage_eta=config.pruning.coverage_eta,
            )
        kept_ids = torch.nonzero(base_output.selection.hard_mask > 0.5, as_tuple=False).flatten().tolist()
        kept_edges_total += len(kept_ids)
        u_map = session_uniqueness(hypergraph.records, hypergraph.incidence)
        contributions = deletion_contributions(
            model,
            batch,
            hypergraph,
            base_output,
            edge_ids=kept_ids,
            target_classes=batch.labels,  # 真实标签语义
        )
        ranked = sorted(
            [(edge_id, abs(value), value) for edge_id, value in contributions.items()],
            key=lambda item: item[1],
            reverse=True,
        )[: args.top_k]
        top_ids = {edge_id for edge_id, _, _ in ranked}

        rows = build_explanation_rows(batch, hypergraph, base_output, model, contributions)
        for row in rows:
            row["uniqueness_u"] = u_map.get(row["hyperedge_id"])
            row["session_rank_by_positive_prob"] = rank
            row["session_positive_probability"] = prob
            row["is_selected_topk"] = row["hyperedge_id"] in top_ids
            if row["is_selected_topk"]:
                type_counter[row["behavior_type"]] += 1
                if row["deletion_contribution"] is not None:
                    contributions_all.append(row["deletion_contribution"])
        all_rows.extend(rows)
        if (rank + 1) % 5 == 0:
            print(f"  processed {rank + 1}/{len(selected)} sessions, kept_edges_so_far={kept_edges_total}")
    timing["stage2_seconds"] = time.perf_counter() - t0

    metadata = {
        "purpose": "Validation-only Top-K key-edge explanation export (no test access)",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "n_sessions": args.n_sessions,
        "top_k": args.top_k,
        "target_semantics": "true_label_probability_difference",
        "selection": "top_positive_probability_sessions",
        "dedup_mode": config.hypergraph.dedup_mode,
        "selected_sessions": [
            {"rank": rank, "session_index": index, "positive_prob": prob, "label": label}
            for rank, (prob, index, label) in enumerate(selected)
        ],
        "selected_true_positive": sum(1 for _, _, lab in selected if lab == 1),
        "kept_edges_total": kept_edges_total,
        "topk_type_distribution": dict(type_counter),
        "topk_contribution_stats": (
            {
                "min": float(np.min(contributions_all)),
                "median": float(np.median(contributions_all)),
                "mean": float(np.mean(contributions_all)),
                "max": float(np.max(contributions_all)),
            }
            if contributions_all
            else None
        ),
        "timing": timing,
    }
    export_jsonl(args.output, all_rows, metadata=metadata)
    print(f"\nrows exported: {len(all_rows)} (candidate rows incl. non-topk)")
    print(f"top-k rows: {sum(1 for r in all_rows if r['is_selected_topk'])}")
    print(f"topk type distribution: {dict(type_counter)}")
    if contributions_all:
        print(
            f"topk contribution: min={metadata['topk_contribution_stats']['min']:.5f} "
            f"median={metadata['topk_contribution_stats']['median']:.5f} "
            f"max={metadata['topk_contribution_stats']['max']:.5f}"
        )
    print(f"timing: stage2={timing['stage2_seconds']:.1f}s")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
