"""作用：在不访问测试集、不训练模型的条件下，量化 Vine 超边集合中的结构冗余。

冗余口径（全部基于候选边集合，不依赖任何训练好的模型）：
1. duplicate_source_count：完全相同的节点集合被多少个中心/来源重复构造（构建阶段已合并，count>1 表示多源重复）。
2. 会话内两两 Jaccard 重叠：近似重复（>=0.9）、高度重叠（>=0.7、>=0.5）的边对占比。
3. 每条边的 max Jaccard（同类型 / 任意类型）：该边能被同会话其它边代表的程度。
4. 子集包含：每条边被同会话另一条边完全包含（F ⊇ E），即严格冗余。

输出：outputs/diagnostics/vine_hyperedge_redundancy_seed42.json
只扫描 seed42 的训练集与验证集；不读取测试集索引。
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def install_pyg_deserialization_stub() -> None:
    """仅为读取可信快照提供最小 PyG 类型占位，不参与任何图计算。"""
    try:
        import torch_geometric  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    package = types.ModuleType("torch_geometric")
    package.__path__ = []
    data_package = types.ModuleType("torch_geometric.data")
    data_package.__path__ = []
    data_module = types.ModuleType("torch_geometric.data.data")

    class PyGDataPlaceholder:
        pass

    def resolve_data_attribute(name: str) -> type[PyGDataPlaceholder]:
        if name.startswith("__"):
            raise AttributeError(name)
        placeholder = type(name, (PyGDataPlaceholder,), {})
        placeholder.__module__ = "torch_geometric.data.data"
        setattr(data_module, name, placeholder)
        return placeholder

    data_module.__getattr__ = resolve_data_attribute
    data_module.__file__ = __file__
    data_module.Data = resolve_data_attribute("Data")
    storage_module = types.ModuleType("torch_geometric.data.storage")
    storage_module.__file__ = __file__
    storage_module.__getattr__ = resolve_data_attribute
    package.data = data_package
    data_package.data = data_module
    data_package.storage = storage_module
    sys.modules["torch_geometric"] = package
    sys.modules["torch_geometric.data"] = data_package
    sys.modules["torch_geometric.data.data"] = data_module
    sys.modules["torch_geometric.data.storage"] = storage_module


install_pyg_deserialization_stub()

import torch  # noqa: E402

from bbp_behm.config import load_experiment_config  # noqa: E402
from bbp_behm.data import PreprocessedSessionDataset  # noqa: E402
from bbp_behm.hypergraph import BehaviorType, HyperedgeBuilder, HyperedgeRecord  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="诊断 Vine 训练/验证集候选超边的结构冗余；不读取测试集。"
    )
    parser.add_argument("--config", type=Path, required=True, help="当前实验 INI 配置。")
    parser.add_argument("--processed-path", type=Path, required=True, help="Vine processed .pt 文件。")
    parser.add_argument("--seed", type=int, default=42, help="沿用训练脚本的分层划分随机种子。")
    parser.add_argument("--output", type=Path, required=True, help="诊断 JSON 输出路径。")
    return parser.parse_args()


def percentile_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "p25": None, "p50": None, "p75": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _session_incidence(records: list[HyperedgeRecord]) -> torch.Tensor:
    """会话内 incidence：行=边，列=该会话成员节点（按出现顺序局部编号）。"""
    member_to_local: dict[int, int] = {}
    local_members: list[list[int]] = []
    for record in records:
        local = []
        for member in record.members:
            if member not in member_to_local:
                member_to_local[member] = len(member_to_local)
            local.append(member_to_local[member])
        local_members.append(local)
    incidence = torch.zeros((len(records), len(member_to_local)), dtype=torch.float32)
    for edge_id, local in enumerate(local_members):
        incidence[edge_id, local] = 1.0
    return incidence


def _jaccard_matrix(incidence: torch.Tensor) -> torch.Tensor:
    intersection = incidence @ incidence.t()
    sizes = incidence.sum(dim=1, keepdim=True)
    union = (sizes + sizes.t() - intersection).clamp_min(1e-8)
    jaccard = intersection / union
    return jaccard - torch.diag_embed(torch.diagonal(jaccard))


def analyze_session_records(records: list[HyperedgeRecord]) -> dict[str, Any]:
    """对单个会话的候选边集合计算冗余统计（不依赖模型）。"""
    if len(records) < 2:
        return None
    incidence = _session_incidence(records)
    if incidence.size(1) == 0:
        return None
    jaccard = _jaccard_matrix(incidence)
    triu = torch.triu(jaccard, diagonal=1)
    nonzero = triu[triu > 0.0]
    pair_jaccards = nonzero.tolist() if nonzero.numel() else []
    max_jaccard = jaccard.max(dim=1).values.tolist()
    containment = (incidence[:, None, :] <= incidence[None, :, :]).all(dim=2).float()
    containment = containment - torch.eye(len(records), dtype=containment.dtype)
    subset_contained = containment.any(dim=1).tolist()
    return {
        "session_edges": len(records),
        "pair_jaccard": percentile_summary(pair_jaccards),
        "edge_max_jaccard": percentile_summary(max_jaccard),
        "subset_contained": percentile_summary(subset_contained),
    }


def pair_fraction_stats(pair_jaccards: list[float]) -> dict[str, float | None]:
    if not pair_jaccards:
        return {"count": 0, "ge_0_5": None, "ge_0_7": None, "ge_0_9": None}
    array = np.asarray(pair_jaccards, dtype=np.float64)
    return {
        "count": int(array.size),
        "ge_0_5": float((array >= 0.5).mean()),
        "ge_0_7": float((array >= 0.7).mean()),
        "ge_0_9": float((array >= 0.9).mean()),
    }


def summarize_split(sessions: list[tuple[int, int, list[HyperedgeRecord]]]) -> dict[str, Any]:
    """聚合训练或验证集：按类型、按类别、按会话。"""
    total_edges = sum(len(records) for _, _, records in sessions)
    total_incidence = sum(record.size for _, _, records in sessions for record in records)

    per_type: dict[str, dict[str, Any]] = {}
    for behavior_type in BehaviorType:
        label = behavior_type.label
        typed = [
            record for _, _, records in sessions
            for record in records if record.behavior_type == behavior_type
        ]
        sizes = [record.size for record in typed]
        dup = [record.duplicate_count for record in typed]

        # 同类型会话内统计。
        max_jac_values: list[float] = []
        subset_values: list[float] = []
        pair_values: list[float] = []
        for _, _, records in sessions:
            session_typed = [r for r in records if r.behavior_type == behavior_type]
            if not session_typed:
                continue
            max_jac_values.extend(_recompute_edge_max(session_typed))
            subset_values.extend(_recompute_subset(session_typed))
            pair_values.extend(_recompute_pair_jaccards(session_typed))

        per_type[label] = {
            "edge_count": len(typed),
            "edge_fraction_all_types": float(len(typed) / max(total_edges, 1)),
            "incidence": sum(sizes),
            "incidence_fraction_all_types": float(sum(sizes) / max(total_incidence, 1)),
            "edge_size": percentile_summary(sizes),
            "duplicate_source_count": percentile_summary(dup),
            "duplicate_edge_fraction": float(sum(1 for value in dup if value > 1) / max(len(dup), 1)),
            "edge_max_jaccard_same_type": percentile_summary(max_jac_values),
            "edge_subset_contained_fraction": percentile_summary(subset_values),
            "session_pair_stats": pair_fraction_stats(pair_values),
        }

    per_label: dict[str, dict[str, Any]] = {}
    for label in (0, 1):
        label_sessions = [(index, records) for index, current_label, records in sessions if current_label == label]
        values: dict[str, Any] = {"session_count": len(label_sessions), "by_type": {}}
        for behavior_type in BehaviorType:
            label_type = behavior_type.label
            typed = [
                record for _, records in label_sessions
                for record in records if record.behavior_type == behavior_type
            ]
            dup = [record.duplicate_count for record in typed]
            max_jac_values: list[float] = []
            subset_values: list[float] = []
            pair_values: list[float] = []
            for _, records in label_sessions:
                session_typed = [r for r in records if r.behavior_type == behavior_type]
                max_jac_values.extend(_recompute_edge_max(session_typed))
                subset_values.extend(_recompute_subset(session_typed))
                pair_values.extend(_recompute_pair_jaccards(session_typed))
            values["by_type"][label_type] = {
                "edge_count": len(typed),
                "duplicate_edge_fraction": float(sum(1 for value in dup if value > 1) / max(len(dup), 1)),
                "edge_max_jaccard_same_type": percentile_summary(max_jac_values),
                "edge_subset_contained_fraction": percentile_summary(subset_values),
                "session_pair_stats": pair_fraction_stats(pair_values),
            }
        per_label[str(label)] = values

    # 会话级：任意类型组合的两两 Jaccard 对。
    session_rows = []
    all_pair_values: list[float] = []
    for _, label, records in sessions:
        analysis = analyze_session_records(records)
        if analysis is None:
            session_rows.append({"session_edges": len(records), "session_pair_stats": pair_fraction_stats([])})
            continue
        session_pair_values = _recompute_pair_jaccards(records)
        all_pair_values.extend(session_pair_values)
        session_rows.append(
            {
                "session_edges": len(records),
                "session_pair_stats": pair_fraction_stats(session_pair_values),
            }
        )

    return {
        "session_count": len(sessions),
        "edge_count": total_edges,
        "incidence": total_incidence,
        "by_type": per_type,
        "by_label": per_label,
        "all_types_session_pair_stats": pair_fraction_stats(all_pair_values),
        "session_rows": session_rows,
    }


def _recompute_edge_max(records: list[HyperedgeRecord]) -> list[float]:
    if len(records) < 2:
        return []
    incidence = _session_incidence(records)
    if incidence.size(1) == 0:
        return []
    return _jaccard_matrix(incidence).max(dim=1).values.tolist()


def _recompute_subset(records: list[HyperedgeRecord]) -> list[float]:
    if len(records) < 2:
        return []
    incidence = _session_incidence(records)
    if incidence.size(1) == 0:
        return []
    containment = (incidence[:, None, :] <= incidence[None, :, :]).all(dim=2).float()
    containment = containment - torch.eye(len(records), dtype=containment.dtype)
    return containment.any(dim=1).tolist()


def _recompute_pair_jaccards(records: list[HyperedgeRecord]) -> list[float]:
    if len(records) < 2:
        return []
    incidence = _session_incidence(records)
    if incidence.size(1) == 0:
        return []
    jaccard = _jaccard_matrix(incidence)
    triu = torch.triu(jaccard, diagonal=1)
    nonzero = triu[triu > 0.0]
    return nonzero.tolist() if nonzero.numel() else []


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
    split = dataset.make_split(
        seed=args.seed,
        test_fraction=config.split.test_fraction,
        validation_fraction=config.split.validation_fraction,
    )
    builder = HyperedgeBuilder(config.hypergraph)

    def build_sessions(indices: Iterable[int]) -> list[tuple[int, int, list[HyperedgeRecord]]]:
        sessions = []
        for index in indices:
            sample = dataset[index]
            records = builder.build_session(
                sample.x,
                sample.node_metadata,
                session_position=0,
                session_id=sample.session_id,
            )
            sessions.append((int(index), int(sample.label), records))
        return sessions

    train_sessions = build_sessions(split.train)
    validation_sessions = build_sessions(split.validation)

    result = {
        "_purpose": "Train/validation-only Vine hyperedge redundancy diagnostic; test split is not read.",
        "seed": args.seed,
        "hypergraph_config": {
            "tau": config.hypergraph.tau,
            "threshold": config.hypergraph.threshold,
            "semantic_gate": config.hypergraph.semantic_gate,
            "time_weight": config.hypergraph.time_weight,
            "time_window_seconds": config.hypergraph.time_window_seconds,
        },
        "redundancy_definitions": {
            "duplicate_edge": "完全相同节点集合被多个来源构造（构建阶段已合并，count>1 为多源重复）",
            "approx_duplicate_pair": "会话内两两 Jaccard >= 0.9 的边对",
            "edge_max_jaccard": "该边与同会话同类型其它边的最大 Jaccard（可代表程度）",
            "edge_subset_contained": "该边成员被同会话另一条边完全包含（严格冗余）",
        },
        "train": summarize_split(train_sessions),
        "validation": summarize_split(validation_sessions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train_sessions": len(train_sessions),
                "validation_sessions": len(validation_sessions),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
