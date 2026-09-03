"""作用：E0 只读结构盘存 —— 在 CV5-80/20 开发集(776 会话)口径下量化"超图规模过大"的结构症状。

范围纪律：
- 只读开发集会话(development_indices);不读取/构建 final_test_indices 中任何会话。
- 不训练、不改模型;只调用 HyperedgeBuilder(构图是确定性的、与标签无关)。
- 输出开发集整体 + 每个 fold-train(620 会话)的聚合统计,可直接与
  outputs/cv5/vine_dedup_cv5_8020_seed42_pw1.5/result.json 的 per-fold 数字交叉核对。

口径说明(全部针对"候选边集合",不依赖训练权重)：
1. raw = dedup_mode=none 的候选;sub = dedup_mode=subset 之后。
2. 残余冗余只统计 sub 之后、同会话同类型的 STC 边：
   - 两两 Jaccard>0 对数(IBG 有向边数上界 = 2×无序对数);
   - 每条边与同类型其它边的 max Jaccard 及其 >=0.7/0.9/0.95 占比;
   - 严格包含残余(应为 0,验证 subset 语义);
   - cover 贪心代表核(按 strength 降序、size 降序)在 tau∈{0.85,0.90,0.95} 下的保留边数
     —— 结构层面"互不冗余"的压缩下界,供方向 A 机制设计参考。
3. 巨边负担：在 fold-train 内取 STC 尺寸 p90/p95,统计 >= 阈值边/incidence 占比、受影响会话数。

用法示例：
python scripts/diagnose_vine_cv5_scale_census.py \
  --config configs/vine_dedup_cv5_8020_seed42_pw1.5.ini \
  --split-json outputs/cv5/vine_dedup_cv5_8020_seed42_pw1.5/split.json \
  --output outputs/diagnostics/vine_cv5_scale_census_dev.json [--limit-sessions N]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np

import torch  # noqa: F401

from bbp_behm.config import load_experiment_config  # noqa: E402
from bbp_behm.data import PreprocessedSessionDataset  # noqa: E402
from bbp_behm.hypergraph import BehaviorType, HyperedgeBuilder, HyperedgeRecord  # noqa: E402

LOCAL_PROCESSED = Path(
    r"E:\Paper\Code\BeHM\data\vine_with_5\KbertConfidence\processed\KbertConfidence_data_bert.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CV5-80/20 开发集超图规模只读盘存(E0)。")
    parser.add_argument("--config", type=Path, required=True, help="实验 INI(提供超图参数与数据校验)。")
    parser.add_argument("--split-json", type=Path, required=True, help="CV5 run 的 split.json。")
    parser.add_argument("--processed-path", type=Path, default=LOCAL_PROCESSED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-sessions", type=int, default=0, help="调试用:只处理前 N 个开发集会话。")
    return parser.parse_args()


def psum(values: Sequence[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def session_local_incidence(records: Sequence[HyperedgeRecord]) -> torch.Tensor:
    member_to_local: dict[int, int] = {}
    local_members: List[List[int]] = []
    for record in records:
        local: List[int] = []
        for member in record.members:
            if member not in member_to_local:
                member_to_local[member] = len(member_to_local)
            local.append(member_to_local[member])
        local_members.append(local)
    incidence = torch.zeros((len(records), len(member_to_local)), dtype=torch.float32)
    for edge_id, local in enumerate(local_members):
        incidence[edge_id, local] = 1.0
    return incidence


def stc_jaccard_matrix(records: Sequence[HyperedgeRecord]) -> torch.Tensor:
    typed = [r for r in records if r.behavior_type == BehaviorType.STC]
    if len(typed) < 2:
        return None, typed
    incidence = session_local_incidence(typed)
    if incidence.size(1) == 0:
        return None, typed
    intersection = incidence @ incidence.t()
    sizes = incidence.sum(dim=1, keepdim=True)
    union = (sizes + sizes.t() - intersection).clamp_min(1e-8)
    jaccard = intersection / union
    jaccard = jaccard - torch.diag_embed(torch.diagonal(jaccard))
    return jaccard, typed


def cover_core_counts(
    typed: Sequence[HyperedgeRecord], jaccard: torch.Tensor, taus: Sequence[float]
) -> dict[str, int]:
    """strength 降序(size 降序并列)贪心;与已保留边 Jaccard>=tau 则跳过。"""
    if len(typed) < 2:
        return {f"tau_{tau:.2f}": len(typed) for tau in taus}
    order = sorted(
        range(len(typed)),
        key=lambda i: (-typed[i].construction_strength, -typed[i].size),
    )
    result: dict[str, int] = {}
    matrix = jaccard.numpy()
    for tau in taus:
        kept: List[int] = []
        for idx in order:
            if kept and bool(np.any(matrix[idx, kept] >= tau)):
                continue
            kept.append(idx)
        result[f"tau_{tau:.2f}"] = len(kept)
    return result


def analyze_session(
    raw: Sequence[HyperedgeRecord], sub: Sequence[HyperedgeRecord]
) -> dict[str, Any]:
    n_raw_by_type = {bt.label: 0 for bt in BehaviorType}
    n_sub_by_type = {bt.label: 0 for bt in BehaviorType}
    size_raw_by_type = {bt.label: [] for bt in BehaviorType}
    size_sub_by_type = {bt.label: [] for bt in BehaviorType}
    for record in raw:
        n_raw_by_type[record.behavior_type.label] += 1
        size_raw_by_type[record.behavior_type.label].append(record.size)
    for record in sub:
        n_sub_by_type[record.behavior_type.label] += 1
        size_sub_by_type[record.behavior_type.label].append(record.size)

    jaccard, typed_stc = stc_jaccard_matrix(sub)
    row: dict[str, Any] = {
        "n_raw": {k: v for k, v in n_raw_by_type.items()},
        "n_sub": {k: v for k, v in n_sub_by_type.items()},
        "size_sub_stc": size_sub_by_type["STC"],
    }
    if jaccard is None:
        row.update(
            {
                "stc_pairs_gt0": 0,
                "stc_ibg_directed_ub": 0,
                "stc_edge_max_jac_ge_0_7": 0.0,
                "stc_edge_max_jac_ge_0_9": 0.0,
                "stc_subset_contained_residual": 0,
                "cover": {f"tau_{tau:.2f}": len(typed_stc) for tau in (0.85, 0.90, 0.95)},
            }
        )
        return row

    triu = torch.triu(jaccard, diagonal=1)
    pairs_gt0 = int((triu > 0.0).sum().item())
    max_jac = jaccard.max(dim=1).values
    # 严格包含残余 = 边 A 是边 B 的真子集(成员层面),subset 之后应为 0。
    typed_by_type = [r for r in sub if r.behavior_type == BehaviorType.STC]
    incidence = session_local_incidence(typed_by_type)
    cont = (incidence[:, None, :] <= incidence[None, :, :]).all(dim=2).float()
    cont = cont - torch.eye(len(typed_by_type), dtype=cont.dtype)
    subset_residual = int(cont.any(dim=1).sum().item())

    m = max_jac.numpy()
    row.update(
        {
            "stc_pairs_gt0": pairs_gt0,
            "stc_ibg_directed_ub": 2 * pairs_gt0,
            "stc_edge_max_jac_ge_0_7": float((m >= 0.7).mean()) if len(m) else 0.0,
            "stc_edge_max_jac_ge_0_9": float((m >= 0.9).mean()) if len(m) else 0.0,
            "stc_edge_max_jac_p90": float(np.quantile(m, 0.90)) if len(m) else 0.0,
            "stc_subset_contained_residual": subset_residual,
            "cover": cover_core_counts(typed_stc, jaccard, (0.85, 0.90, 0.95)),
        }
    )
    return row


def summarize_sessions(
    rows: Sequence[dict], indices: Sequence[int], labels: Sequence[int]
) -> dict[str, Any]:
    n_sessions = len(rows)
    session_total_sub = [sum(r["n_sub"].values()) for r in rows]
    session_total_raw = [sum(r["n_raw"].values()) for r in rows]
    stc_sizes: List[float] = []
    for r in rows:
        stc_sizes.extend(r["size_sub_stc"])
    pairs = [r["stc_pairs_gt0"] for r in rows]
    ibg_ub = [r["stc_ibg_directed_ub"] for r in rows]
    stc_total_sub = int(sum(r["n_sub"].get("STC", 0) for r in rows))
    stc_total_raw = int(sum(r["n_raw"].get("STC", 0) for r in rows))

    def per_type_totals(key: str) -> dict[str, int]:
        return {
            behavior_type.label: int(sum(r[key].get(behavior_type.label, 0) for r in rows))
            for behavior_type in BehaviorType
        }

    def edge_frac(key: str) -> float:
        denom = max(stc_total_sub, 1)
        return float(sum(r[key] for r in rows) / denom)

    def cover_frac(tau_key: str) -> float:
        if not rows:
            return 0.0
        per_session = [
            r["cover"][tau_key] / max(int(r["n_sub"].get("STC", 0)), 1) for r in rows
        ]
        return float(np.mean(per_session))

    return {
        "session_count": n_sessions,
        "edges_raw_total": int(sum(session_total_raw)),
        "edges_sub_total": int(sum(session_total_sub)),
        "edges_per_session_sub": psum(session_total_sub),
        "edges_per_session_raw": psum(session_total_raw),
        "stc_edges_sub_total": int(stc_total_sub),
        "stc_edges_raw_total": int(stc_total_raw),
        "per_type_raw": per_type_totals("n_raw"),
        "per_type_sub": per_type_totals("n_sub"),
        "stc_size_sub": psum(stc_sizes),
        "stc_pairs_gt0_per_session": psum(pairs),
        "stc_ibg_directed_ub_per_session": psum(ibg_ub),
        "stc_edge_max_jac_ge_0_7_frac": edge_frac("stc_edge_max_jac_ge_0_7"),
        "stc_edge_max_jac_ge_0_9_frac": edge_frac("stc_edge_max_jac_ge_0_9"),
        "stc_subset_contained_residual_total": int(sum(r["stc_subset_contained_residual"] for r in rows)),
        "cover_tau_0_85": {"kept_total": int(sum(r["cover"]["tau_0.85"] for r in rows)), "session_frac": cover_frac("tau_0.85")},
        "cover_tau_0_90": {"kept_total": int(sum(r["cover"]["tau_0.90"] for r in rows)), "session_frac": cover_frac("tau_0.90")},
        "cover_tau_0_95": {"kept_total": int(sum(r["cover"]["tau_0.95"] for r in rows)), "session_frac": cover_frac("tau_0.95")},
        "by_label": {
            str(label): {
                "session_count": int(np.sum(np.asarray(labels) == label)),
                "edges_sub_total": int(
                    sum(sum(r["n_sub"].values()) for r, y in zip(rows, labels) if y == label)
                ),
            }
            for label in (0, 1)
        },
    }


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    config = replace(config, data=replace(config.data, processed_path=args.processed_path))
    dataset = PreprocessedSessionDataset(
        config.data.processed_path,
        config.data.processed_sha256,
        config.data.expected_sessions,
        config.data.expected_nodes,
        config.data.expected_features,
    )
    split_data = json.loads(args.split_json.read_text(encoding="utf-8"))
    dev_indices = [int(i) for i in split_data["development_indices"]]
    folds_detail = split_data["folds_detail"]
    if args.limit_sessions:
        dev_indices = dev_indices[: args.limit_sessions]

    builder_none = HyperedgeBuilder(replace(config.hypergraph, dedup_mode="none"))
    builder_subset = HyperedgeBuilder(config.hypergraph)

    rows: List[dict] = []
    labels: List[int] = []
    for position, index in enumerate(dev_indices):
        sample = dataset[index]
        raw = builder_none.build_session(
            sample.x, sample.node_metadata, session_position=0, session_id=sample.session_id
        )
        sub = builder_subset.build_session(
            sample.x, sample.node_metadata, session_position=0, session_id=sample.session_id
        )
        rows.append(analyze_session(raw, sub))
        labels.append(int(sample.label))
        if position % 100 == 0:
            print(f"built {position + 1}/{len(dev_indices)}", flush=True)

    dev_set = set(dev_indices)
    overall = summarize_sessions(rows, dev_indices, labels)

    by_fold: List[dict] = []
    for fold in folds_detail:
        fold_no = int(fold["fold"])
        train_idx = [int(i) for i in fold["train"] if int(i) in dev_set]
        val_idx = [int(i) for i in fold["validation"] if int(i) in dev_set]
        pos_map = {index: position for position, index in enumerate(dev_indices)}
        train_rows = [rows[pos_map[i]] for i in train_idx]
        train_labels = [labels[pos_map[i]] for i in train_idx]
        train_stats = summarize_sessions(train_rows, train_idx, train_labels)

        # 巨边负担:在 fold-train 内按 STC 尺寸取 p90/p95。
        stc_sizes = [s for r in train_rows for s in r["size_sub_stc"]]
        arr = np.asarray(stc_sizes, dtype=np.float64)
        giant: dict[str, Any] = {}
        if arr.size:
            for qkey, q in (("p90", 0.90), ("p95", 0.95)):
                th = float(np.quantile(arr, q))
                edge_burden = float((arr >= th).mean())
                inc_burden = float(arr[arr >= th].sum() / max(arr.sum(), 1))
                sessions_with = int(
                    sum(any(s >= th for s in r["size_sub_stc"]) for r in train_rows)
                )
                giant[qkey] = {
                    "threshold": th,
                    "edge_fraction_ge": edge_burden,
                    "incidence_fraction_ge": inc_burden,
                    "session_count_with": sessions_with,
                }
        by_fold.append(
            {
                "fold": fold_no,
                "train_session_count": len(train_rows),
                "train": train_stats,
                "val_sanity_candidate_edges_sub": int(
                    sum(sum(r["n_sub"].values()) for i in val_idx for r in [rows[pos_map[i]]])
                ),
                "giant_stc": giant,
            }
        )

    result = {
        "_purpose": "E0 read-only structural census on CV5-80/20 development sessions (test sessions never built).",
        "split_json": str(args.split_json),
        "hypergraph_config": {
            "tau": config.hypergraph.tau,
            "threshold": config.hypergraph.threshold,
            "semantic_gate": config.hypergraph.semantic_gate,
            "time_weight": config.hypergraph.time_weight,
            "time_window_seconds": config.hypergraph.time_window_seconds,
            "dedup_mode_sub": config.hypergraph.dedup_mode,
        },
        "overall_dev": overall,
        "by_fold": by_fold,
        "session_rows": [
            {
                "dev_position": position,
                "label": int(labels[position]),
                "n_raw": rows[position]["n_raw"],
                "n_sub": rows[position]["n_sub"],
                "stc_ibg_directed_ub": rows[position]["stc_ibg_directed_ub"],
                "stc_edge_max_jac_ge_0_7": rows[position]["stc_edge_max_jac_ge_0_7"],
                "stc_edge_max_jac_ge_0_9": rows[position]["stc_edge_max_jac_ge_0_9"],
                "cover": rows[position]["cover"],
                "session_node_count": None,  # 占位:可选后处理补充
            }
            for position in range(len(rows))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "dev_sessions": len(dev_indices)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
