"""作用：只读模拟 STC 候选边去冗余规则（子集剔除 / Jaccard 合并），量化压缩率与覆盖保持。

不训练、不访问测试集。模拟口径：
- 规则 subset：会话内 STC 边中，若边 E 的成员集合是另一条 STC 边 F 的严格子集，则 E 可被剔除（F 完全覆盖 E）。
- 规则 jaccard_tau：会话内 STC 边两两 Jaccard>=tau 时，保留 construction_strength 更高（并列时尺寸更大）的一条，剔除另一条。
- 覆盖保持：被剔除边与其保留代表边的最大 Jaccard 分布（被代表程度）；以及被剔除边在类型内是否仍被保留集合完全包含。
输出：outputs/diagnostics/vine_stc_dedup_simulation_seed42.json
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

from bbp_behm.config import load_experiment_config  # noqa: E402
from bbp_behm.data import PreprocessedSessionDataset  # noqa: E402
from bbp_behm.hypergraph import BehaviorType, HyperedgeBuilder, HyperedgeRecord  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="模拟 STC 去冗余规则的压缩率与覆盖保持；不读取测试集。")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--processed-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jaccard-taus", type=str, default="0.7,0.8,0.9", help="逗号分隔的 Jaccard 阈值列表。")
    return parser.parse_args()


def percentile_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _jaccard_matrix_from_sets(sets: list[frozenset[int]]) -> np.ndarray:
    n = len(sets)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            matrix[i, j] = matrix[j, i] = inter / union if union else 0.0
    return matrix


def apply_subset_dedup(records: list[HyperedgeRecord]) -> tuple[list[HyperedgeRecord], dict[str, Any]]:
    """剔除成员集合是另一条 STC 边严格子集的边。"""
    sets = [frozenset(record.members) for record in records]
    n = len(records)
    contained_by_any = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if len(sets[j]) > len(sets[i]) and sets[i] < sets[j]:
                contained_by_any[i] = True
                break
    kept = [record for i, record in enumerate(records) if not contained_by_any[i]]
    stats = {
        "kept": len(kept),
        "removed": n - len(kept),
        "removal_fraction": float(n - len(kept)) / max(n, 1),
    }
    return kept, stats


def apply_jaccard_dedup(records: list[HyperedgeRecord], tau: float) -> tuple[list[HyperedgeRecord], dict[str, Any]]:
    """Jaccard>=tau 的边对中保留 construction_strength 更高（并列尺寸更大、再并列索引更小）者。"""
    n = len(records)
    if n < 2:
        return records, {"kept": n, "removed": 0, "removal_fraction": 0.0}
    sets = [frozenset(record.members) for record in records]
    matrix = _jaccard_matrix_from_sets(sets)
    removed = [False] * n
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] >= tau:
                key_i = (records[i].construction_strength, len(records[i].members), -i)
                key_j = (records[j].construction_strength, len(records[j].members), -j)
                if key_i >= key_j:
                    removed[j] = True
                else:
                    removed[i] = True
    kept = [record for i, record in enumerate(records) if not removed[i]]
    return kept, {"kept": len(kept), "removed": n - len(kept), "removal_fraction": float(n - len(kept)) / max(n, 1)}


def simulate_split(sessions: list[tuple[int, int, list[HyperedgeRecord]]], taus: list[float]) -> dict[str, Any]:
    total_stc = sum(1 for _, _, records in sessions for record in records if record.behavior_type == BehaviorType.STC)
    result: dict[str, Any] = {"stc_edges": total_stc}

    # subset 规则。
    kept_all = 0
    removed_all = 0
    for _, label, records in sessions:
        stc = [record for record in records if record.behavior_type == BehaviorType.STC]
        if len(stc) < 2:
            kept_all += len(stc)
            continue
        kept, stats = apply_subset_dedup(stc)
        kept_all += stats["kept"]
        removed_all += stats["removed"]
    # 逐条收集被剔除边的包含标记与最大 Jaccard。
    subset_cov_best: list[float] = []
    subset_cov_contained: list[float] = []
    for _, label, records in sessions:
        stc = [record for record in records if record.behavior_type == BehaviorType.STC]
        if len(stc) < 2:
            continue
        kept, _ = apply_subset_dedup(stc)
        removed = [record for record in stc if record not in kept]
        kept_sets = [frozenset(r.members) for r in kept]
        for record in removed:
            rset = frozenset(record.members)
            best = 0.0
            is_contained = False
            for kept_set in kept_sets:
                inter = len(rset & kept_set)
                union = len(rset | kept_set)
                jac = inter / union if union else 0.0
                best = max(best, jac)
                if rset <= kept_set and rset != kept_set:
                    is_contained = True
            subset_cov_best.append(best)
            subset_cov_contained.append(1.0 if is_contained else 0.0)
    result["subset"] = {
        "kept": kept_all,
        "removed": removed_all,
        "removal_fraction": float(removed_all) / max(total_stc, 1),
        "removed_max_jaccard": percentile_summary(subset_cov_best),
        "removed_contained_fraction": float(np.mean(subset_cov_contained)) if subset_cov_contained else None,
    }

    # Jaccard 规则（各 tau）。
    result["jaccard"] = {}
    for tau in taus:
        kept_all = 0
        removed_all = 0
        cov_best: list[float] = []
        cov_contained: list[float] = []
        for _, label, records in sessions:
            stc = [record for record in records if record.behavior_type == BehaviorType.STC]
            if len(stc) < 2:
                kept_all += len(stc)
                continue
            kept, stats = apply_jaccard_dedup(stc, tau)
            kept_all += stats["kept"]
            removed_all += stats["removed"]
            removed = [record for record in stc if record not in kept]
            kept_sets = [frozenset(r.members) for r in kept]
            for record in removed:
                rset = frozenset(record.members)
                best = 0.0
                is_contained = False
                for kept_set in kept_sets:
                    inter = len(rset & kept_set)
                    union = len(rset | kept_set)
                    jac = inter / union if union else 0.0
                    best = max(best, jac)
                    if rset <= kept_set and rset != kept_set:
                        is_contained = True
                cov_best.append(best)
                cov_contained.append(1.0 if is_contained else 0.0)
        result["jaccard"][str(tau)] = {
            "kept": kept_all,
            "removed": removed_all,
            "removal_fraction": float(removed_all) / max(total_stc, 1),
            "removed_max_jaccard": percentile_summary(cov_best),
            "removed_contained_fraction": float(np.mean(cov_contained)) if cov_contained else None,
        }
    return result


def main() -> None:
    args = parse_args()
    taus = [float(item.strip()) for item in args.jaccard_taus.split(",") if item.strip()]
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
                sample.x, sample.node_metadata, session_position=0, session_id=sample.session_id
            )
            sessions.append((int(index), int(sample.label), records))
        return sessions

    train_sessions = build_sessions(split.train)
    validation_sessions = build_sessions(split.validation)
    result = {
        "_purpose": "Train/validation-only STC dedup simulation; test split is not read.",
        "seed": args.seed,
        "hypergraph_config": {
            "tau": config.hypergraph.tau,
            "threshold": config.hypergraph.threshold,
            "semantic_gate": config.hypergraph.semantic_gate,
            "time_weight": config.hypergraph.time_weight,
        },
        "rules": {
            "subset": "剔除成员集合是另一条 STC 边严格子集的边",
            "jaccard": "Jaccard>=tau 的边对保留 construction_strength 更高（并列尺寸更大）者",
        },
        "train": simulate_split(train_sessions, taus),
        "validation": simulate_split(validation_sessions, taus),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
