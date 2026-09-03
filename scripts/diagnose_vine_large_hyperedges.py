"""作用：在不访问测试集、不训练模型的条件下，统计 Vine 训练集和验证集的超大超边分布。"""

from __future__ import annotations

import argparse
import json
import sys
import types
from collections import defaultdict
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

    # 诊断只使用保存对象上的张量和元数据属性；该占位类满足 pickle 的类型解析需求。
    package = types.ModuleType("torch_geometric")
    package.__path__ = []
    data_package = types.ModuleType("torch_geometric.data")
    data_package.__path__ = []
    data_module = types.ModuleType("torch_geometric.data.data")

    class PyGDataPlaceholder:
        """承接快照中仅保存属性的 PyG 数据类状态。"""

        pass

    def resolve_data_attribute(name: str) -> type[PyGDataPlaceholder]:
        # PyG 的 Data、DataEdgeAttr 等在本诊断中都只需可反序列化的属性容器。
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

from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset
from bbp_behm.hypergraph import BehaviorType, HyperedgeBuilder, HyperedgeRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="诊断 Vine 训练集和验证集中的超大超边；不会读取测试集。"
    )
    parser.add_argument("--config", type=Path, required=True, help="当前实验 INI 配置。")
    parser.add_argument("--processed-path", type=Path, required=True, help="Vine processed .pt 文件。")
    parser.add_argument("--seed", type=int, default=42, help="沿用训练脚本的分层划分随机种子。")
    parser.add_argument("--output", type=Path, required=True, help="诊断 JSON 输出路径。")
    return parser.parse_args()


def percentile_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "p50": None, "p75": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def build_records(
    dataset: PreprocessedSessionDataset,
    indices: Iterable[int],
    builder: HyperedgeBuilder,
) -> list[tuple[int, int, list[HyperedgeRecord]]]:
    """逐会话构边，避免把所有会话拼接为一个大批次。"""
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


def thresholds_from_train(
    sessions: list[tuple[int, int, list[HyperedgeRecord]]],
) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for behavior_type in BehaviorType:
        sizes = [record.size for _, _, records in sessions for record in records if record.behavior_type == behavior_type]
        if not sizes:
            thresholds[behavior_type.label] = {"p90": float("inf"), "p95": float("inf")}
            continue
        values = np.asarray(sizes, dtype=np.float64)
        thresholds[behavior_type.label] = {
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
        }
    return thresholds


def summarize_split(
    sessions: list[tuple[int, int, list[HyperedgeRecord]]],
    thresholds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    by_label: dict[str, dict[str, Any]] = {}
    total_edges = sum(len(records) for _, _, records in sessions)
    total_incidence = sum(record.size for _, _, records in sessions for record in records)

    for behavior_type in BehaviorType:
        records = [record for _, _, current in sessions for record in current if record.behavior_type == behavior_type]
        sizes = [record.size for record in records]
        incidence = sum(sizes)
        type_thresholds = thresholds[behavior_type.label]
        large_summary = {}
        for threshold_name, threshold in type_thresholds.items():
            large = [record for record in records if record.size >= threshold]
            large_incidence = sum(record.size for record in large)
            large_summary[threshold_name] = {
                "threshold": threshold,
                "edge_count": len(large),
                "edge_fraction_within_type": float(len(large) / max(len(records), 1)),
                "incidence": large_incidence,
                "incidence_fraction_within_type": float(large_incidence / max(incidence, 1)),
            }
        by_type[behavior_type.label] = {
            "edge_size": percentile_summary(sizes),
            "duplicate_source_count": percentile_summary(record.duplicate_count for record in records),
            "edge_count": len(records),
            "edge_fraction_all_types": float(len(records) / max(total_edges, 1)),
            "incidence": incidence,
            "incidence_fraction_all_types": float(incidence / max(total_incidence, 1)),
            "large_edge_summary": large_summary,
        }

    for label in (0, 1):
        label_sessions = [(index, records) for index, current_label, records in sessions if current_label == label]
        values: dict[str, Any] = {"session_count": len(label_sessions), "by_type": {}}
        for behavior_type in BehaviorType:
            type_records = [record for _, records in label_sessions for record in records if record.behavior_type == behavior_type]
            type_incidence = sum(record.size for record in type_records)
            p95 = thresholds[behavior_type.label]["p95"]
            large = [record for record in type_records if record.size >= p95]
            sessions_with_large = sum(
                any(record.behavior_type == behavior_type and record.size >= p95 for record in records)
                for _, records in label_sessions
            )
            values["by_type"][behavior_type.label] = {
                "edge_count": len(type_records),
                "edge_size": percentile_summary(record.size for record in type_records),
                "incidence": type_incidence,
                "p95_large_edge_count": len(large),
                "p95_large_incidence_fraction": float(sum(record.size for record in large) / max(type_incidence, 1)),
                "sessions_with_p95_large_edge": sessions_with_large,
                "session_fraction_with_p95_large_edge": float(sessions_with_large / max(len(label_sessions), 1)),
            }
        by_label[str(label)] = values

    session_burden = []
    for index, label, records in sessions:
        incidence = sum(record.size for record in records)
        large_incidence = sum(
            record.size
            for record in records
            if record.size >= thresholds[record.behavior_type.label]["p95"]
        )
        session_burden.append(
            {
                "session_index": index,
                "label": label,
                "edge_count": len(records),
                "incidence": incidence,
                "p95_large_incidence": large_incidence,
                "p95_large_incidence_fraction": float(large_incidence / max(incidence, 1)),
            }
        )
    return {
        "session_count": len(sessions),
        "edge_count": total_edges,
        "incidence": total_incidence,
        "by_type": by_type,
        "by_label": by_label,
        "session_burden": session_burden,
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
    split = dataset.make_split(
        seed=args.seed,
        test_fraction=config.split.test_fraction,
        validation_fraction=config.split.validation_fraction,
    )
    builder = HyperedgeBuilder(config.hypergraph)
    train_sessions = build_records(dataset, split.train, builder)
    validation_sessions = build_records(dataset, split.validation, builder)
    thresholds = thresholds_from_train(train_sessions)
    result = {
        "_purpose": "Train/validation-only Vine large-hyperedge diagnostic; test split is not read.",
        "seed": args.seed,
        "hypergraph_config": {
            "tau": config.hypergraph.tau,
            "threshold": config.hypergraph.threshold,
            "semantic_gate": config.hypergraph.semantic_gate,
            "time_weight": config.hypergraph.time_weight,
            "time_window_seconds": config.hypergraph.time_window_seconds,
        },
        "large_edge_thresholds_from_train": thresholds,
        "train": summarize_split(train_sessions, thresholds),
        "validation": summarize_split(validation_sessions, thresholds),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "train_sessions": len(train_sessions), "validation_sessions": len(validation_sessions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
