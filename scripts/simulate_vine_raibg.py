"""作用：只读模拟路径 B（唯一信息量驱动的 RA-IBG）在 train+validation 上的机制效果。

模拟内容（不训练、不读取测试集）：
1. 超边唯一信息量 u(e) = 1 - max_{e'!=e, same session} Jaccard(e, e') 的分布（按类型、按类别）；
2. 基线 IBG（纯 Jaccard 边权）与 RA-IBG（jaccard*min(u_i,u_j)、jaccard*u_i*u_j）的权重分布对比：
   - IBG 总边数（Jaccard>0 的边对数）
   - 有效高权边数（W>=0.5 / W>=0.7）
   - Louvain 社区数量
3. dedup subset 处理后的超图（当前方法状态）上重复以上统计。

输出：outputs/diagnostics/vine_raibg_simulation_seed42.json
"""

from __future__ import annotations

import argparse
import json
import sys
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
    import types

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
import networkx as nx  # noqa: E402
import community as community_louvain  # noqa: E402

from bbp_behm.config import load_experiment_config  # noqa: E402
from bbp_behm.data import PreprocessedSessionDataset  # noqa: E402
from bbp_behm.hypergraph import BehaviorType, HyperedgeBuilder, HyperedgeRecord  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="模拟 RA-IBG 唯一信息量降权效果；不读取测试集。")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--processed-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dedup-mode", choices=("none", "subset"), default="subset")
    return parser.parse_args()


def percentile_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def session_incidence(records: list[HyperedgeRecord]) -> torch.Tensor:
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


def session_jaccard(incidence: torch.Tensor) -> torch.Tensor:
    if incidence.size(0) < 2 or incidence.size(1) == 0:
        return torch.zeros((incidence.size(0), incidence.size(0)), dtype=torch.float32)
    intersection = incidence @ incidence.t()
    sizes = incidence.sum(dim=1, keepdim=True)
    union = (sizes + sizes.t() - intersection).clamp_min(1e-8)
    jaccard = intersection / union
    return jaccard - torch.diag_embed(torch.diagonal(jaccard))


def uniqueness_scores(jaccard: torch.Tensor) -> torch.Tensor:
    """u(e) = 1 - max_{e'!=e} Jaccard(e, e')；无边或单边时 u=1。"""
    if jaccard.size(0) == 0:
        return torch.empty((0,), dtype=torch.float32)
    max_other = jaccard.max(dim=1).values
    return (1.0 - max_other).clamp(0.0, 1.0)


def ibg_stats(jaccard: torch.Tensor, u: torch.Tensor, incidence: torch.Tensor) -> dict[str, Any]:
    """对给定会话内 Jaccard 矩阵与唯一性向量，统计多种边权形态下的 IBG 结构与社区。"""
    n = int(jaccard.size(0))
    empty = {
        "nodes": n,
        "jaccard_edges": 0,
        "w_jaccard_ge0_5": 0, "w_jaccard_ge0_7": 0,
        "w_min_ge0_5": 0, "w_min_ge0_7": 0,
        "w_prod_ge0_5": 0, "w_prod_ge0_7": 0,
        "w_softmin_ge0_5": 0, "w_softmin_ge0_7": 0,
        "w_softprod_ge0_5": 0, "w_softprod_ge0_7": 0,
        "w_jaccard_mean": 0.0, "w_min_mean": 0.0, "w_prod_mean": 0.0,
        "w_softmin_mean": 0.0, "w_softprod_mean": 0.0,
        "communities_jaccard": 0, "communities_min": 0, "communities_prod": 0,
        "communities_softmin": 0, "communities_softprod": 0,
    }
    if n < 2:
        return empty
    triu_mask = torch.triu(torch.ones_like(jaccard), diagonal=1) > 0
    positive = triu_mask & (jaccard > 0.0)
    if int(positive.sum().item()) == 0:
        return empty
    weights_j = jaccard[positive]
    src = positive.nonzero(as_tuple=False)[:, 0]
    dst = positive.nonzero(as_tuple=False)[:, 1]
    u_i = u[src]
    u_j = u[dst]
    u_min = torch.minimum(u_i, u_j)
    u_prod = u_i * u_j
    weights_min = weights_j * u_min
    weights_prod = weights_j * u_prod
    # 温和降权：0.5 + 0.5*x，使权重映射到 [0.5J, J]，只对高冗余边压低一半。
    weights_softmin = weights_j * (0.5 + 0.5 * u_min)
    weights_softprod = weights_j * (0.5 + 0.5 * u_prod)

    def community_count(weights: torch.Tensor) -> int:
        if int(weights.numel()) == 0:
            return n
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for position in range(int(weights.numel())):
            s = int(src[position].item())
            t = int(dst[position].item())
            if s <= t:
                graph.add_edge(s, t, weight=float(weights[position].detach().cpu().item()))
        partition = community_louvain.best_partition(graph, weight="weight", random_state=42)
        return len(set(partition.values()))

    # 硬形态：直接裁剪 IBG 边，改变拓扑。
    hard_results: dict[str, dict[str, Any]] = {}
    for tau in (0.05, 0.10, 0.15, 0.20):
        keep_mask = (weights_j * u_min) >= tau
        kept_weights = weights_j[keep_mask]
        kept_src = src[keep_mask]
        kept_dst = dst[keep_mask]

        def community_count_kept(weights: torch.Tensor, kept_src_: torch.Tensor, kept_dst_: torch.Tensor) -> int:
            if int(weights.numel()) == 0:
                return n
            graph = nx.Graph()
            graph.add_nodes_from(range(n))
            for position in range(int(weights.numel())):
                s = int(kept_src_[position].item())
                t = int(kept_dst_[position].item())
                if s <= t:
                    graph.add_edge(s, t, weight=float(weights[position].detach().cpu().item()))
            partition = community_louvain.best_partition(graph, weight="weight", random_state=42)
            return len(set(partition.values()))

        hard_results[f"hard_min_tau{tau:g}"] = {
            "ibg_edges_kept": int(keep_mask.sum().item()),
            "ibg_edges_total": int(weights_j.numel()),
            "edge_retention": float(keep_mask.float().mean().item()),
            "communities": community_count_kept(kept_weights, kept_src, kept_dst),
        }
    # u 阈值剪边：先删 u<theta 的边（节点），剩余边按 Jaccard>0 建 IBG。
    for theta in (0.10, 0.20, 0.30):
        node_keep = u >= theta
        keep_idx = node_keep.nonzero(as_tuple=False).flatten()
        if keep_idx.numel() < 2:
            hard_results[f"u_cut_theta{theta:g}"] = {"ibg_edges_kept": 0, "ibg_edges_total": int(weights_j.numel()), "edge_retention": 0.0, "communities": int(keep_idx.numel())}
            continue
        sub_incidence = incidence[keep_idx]
        sub_jaccard = session_jaccard(sub_incidence)
        sub_triu = torch.triu(torch.ones_like(sub_jaccard), diagonal=1) > 0
        sub_pos = sub_triu & (sub_jaccard > 0.0)
        sub_weights = sub_jaccard[sub_pos]
        sub_src = sub_pos.nonzero(as_tuple=False)[:, 0]
        sub_dst = sub_pos.nonzero(as_tuple=False)[:, 1]

        def community_count_sub(weights: torch.Tensor, sub_src_: torch.Tensor, sub_dst_: torch.Tensor, m: int) -> int:
            if int(weights.numel()) == 0:
                return m
            graph = nx.Graph()
            graph.add_nodes_from(range(m))
            for position in range(int(weights.numel())):
                s = int(sub_src_[position].item())
                t = int(sub_dst_[position].item())
                if s <= t:
                    graph.add_edge(s, t, weight=float(weights[position].detach().cpu().item()))
            partition = community_louvain.best_partition(graph, weight="weight", random_state=42)
            return len(set(partition.values()))

        hard_results[f"u_cut_theta{theta:g}"] = {
            "ibg_edges_kept": int(sub_weights.numel()),
            "ibg_edges_total": int(weights_j.numel()),
            "edge_retention": float(sub_weights.numel() / max(int(weights_j.numel()), 1)),
            "communities": community_count_sub(sub_weights, sub_src, sub_dst, int(keep_idx.numel())),
            "nodes_kept": int(keep_idx.numel()),
            "nodes_total": n,
        }
    return {
        "nodes": n,
        "jaccard_edges": int(positive.sum().item()),
        "hard_variants": hard_results,
        "w_jaccard_ge0_5": int((weights_j >= 0.5).sum().item()),
        "w_jaccard_ge0_7": int((weights_j >= 0.7).sum().item()),
        "w_min_ge0_5": int((weights_min >= 0.5).sum().item()),
        "w_min_ge0_7": int((weights_min >= 0.7).sum().item()),
        "w_prod_ge0_5": int((weights_prod >= 0.5).sum().item()),
        "w_prod_ge0_7": int((weights_prod >= 0.7).sum().item()),
        "w_softmin_ge0_5": int((weights_softmin >= 0.5).sum().item()),
        "w_softmin_ge0_7": int((weights_softmin >= 0.7).sum().item()),
        "w_softprod_ge0_5": int((weights_softprod >= 0.5).sum().item()),
        "w_softprod_ge0_7": int((weights_softprod >= 0.7).sum().item()),
        "w_jaccard_mean": float(weights_j.mean().item()),
        "w_min_mean": float(weights_min.mean().item()),
        "w_prod_mean": float(weights_prod.mean().item()),
        "w_softmin_mean": float(weights_softmin.mean().item()),
        "w_softprod_mean": float(weights_softprod.mean().item()),
        "communities_jaccard": community_count(weights_j),
        "communities_min": community_count(weights_min),
        "communities_prod": community_count(weights_prod),
        "communities_softmin": community_count(weights_softmin),
        "communities_softprod": community_count(weights_softprod),
    }


def analyze_session(records: list[HyperedgeRecord]) -> dict[str, Any]:
    incidence = session_incidence(records)
    if incidence.size(0) == 0:
        return {"edges": 0, "u": [], "ibg": None, "u_by_type": {}}
    jaccard = session_jaccard(incidence)
    u = uniqueness_scores(jaccard)
    u_by_type: dict[str, list[float]] = {}
    for behavior_type in BehaviorType:
        type_mask = [i for i, r in enumerate(records) if r.behavior_type == behavior_type]
        u_by_type[behavior_type.label] = u[type_mask].tolist() if type_mask else []
    return {
        "edges": len(records),
        "u": u.tolist(),
        "u_by_type": u_by_type,
        "ibg": ibg_stats(jaccard, u, incidence),
    }


def summarize_sessions(sessions: list[tuple[int, int, dict[str, Any]]]) -> dict[str, Any]:
    all_u: list[float] = []
    u_by_type: dict[str, list[float]] = {behavior_type.label: [] for behavior_type in BehaviorType}
    ibg_jaccard_edges: list[int] = []
    ibg_w_min_edges: list[int] = []
    ibg_w_prod_edges: list[int] = []
    ibg_jac_ge05: list[int] = []
    ibg_min_ge05: list[int] = []
    ibg_prod_ge05: list[int] = []
    ibg_softmin_ge05: list[int] = []
    ibg_softprod_ge05: list[int] = []
    ibg_w_mean: dict[str, list[float]] = {"jaccard": [], "min": [], "prod": [], "softmin": [], "softprod": []}
    communities: dict[str, list[int]] = {"jaccard": [], "min": [], "prod": [], "softmin": [], "softprod": []}
    nodes_list: list[int] = []
    by_label: dict[str, list[float]] = {"0": [], "1": []}
    hard_variants: dict[str, dict[str, list[float | int]]] = {}
    for key in ("hard_min_tau0.05", "hard_min_tau0.1", "hard_min_tau0.15", "hard_min_tau0.2",
                "u_cut_theta0.1", "u_cut_theta0.2", "u_cut_theta0.3"):
        hard_variants[key] = {"edge_retention": [], "communities": []}

    for _, label, analysis in sessions:
        all_u.extend(analysis["u"])
        for t, values in analysis["u_by_type"].items():
            u_by_type[t].extend(values)
        by_label[str(label)].extend(analysis["u"])
        if analysis["ibg"] is None:
            continue
        ibg = analysis["ibg"]
        nodes_list.append(ibg["nodes"])
        for key in hard_variants:
            if key in ibg.get("hard_variants", {}):
                hard_variants[key]["edge_retention"].append(ibg["hard_variants"][key]["edge_retention"])
                hard_variants[key]["communities"].append(ibg["hard_variants"][key]["communities"])
        ibg_jaccard_edges.append(ibg["jaccard_edges"])
        ibg_w_min_edges.append(int(ibg["jaccard_edges"] - ibg["w_min_ge0_5"]))
        ibg_w_prod_edges.append(int(ibg["jaccard_edges"] - ibg["w_prod_ge0_5"]))
        ibg_jac_ge05.append(ibg["w_jaccard_ge0_5"])
        ibg_min_ge05.append(ibg["w_min_ge0_5"])
        ibg_prod_ge05.append(ibg["w_prod_ge0_5"])
        ibg_softmin_ge05.append(ibg["w_softmin_ge0_5"])
        ibg_softprod_ge05.append(ibg["w_softprod_ge0_5"])
        for key in ibg_w_mean:
            ibg_w_mean[key].append(ibg[f"w_{key}_mean"])
        for key in communities:
            communities[key].append(ibg[f"communities_{key}"])

    return {
        "u_all": percentile_summary(all_u),
        "u_by_type": {t: percentile_summary(values) for t, values in u_by_type.items()},
        "u_by_label": {t: percentile_summary(values) for t, values in by_label.items()},
        "ibg": {
            "sessions_with_ibg": len(nodes_list),
            "nodes_per_session": percentile_summary(nodes_list),
            "jaccard_edges_per_session": percentile_summary(ibg_jaccard_edges),
            "min_u_edges_below_0_5_per_session": percentile_summary(ibg_w_min_edges),
            "prod_u_edges_below_0_5_per_session": percentile_summary(ibg_w_prod_edges),
            "jaccard_ge0_5_per_session": percentile_summary(ibg_jac_ge05),
            "min_u_ge0_5_per_session": percentile_summary(ibg_min_ge05),
            "prod_u_ge0_5_per_session": percentile_summary(ibg_prod_ge05),
            "softmin_u_ge0_5_per_session": percentile_summary(ibg_softmin_ge05),
            "softprod_u_ge0_5_per_session": percentile_summary(ibg_softprod_ge05),
            "weight_mean_per_session": {key: percentile_summary(values) for key, values in ibg_w_mean.items()},
            "communities_per_session": {key: percentile_summary(values) for key, values in communities.items()},
            "hard_variants": {
                key: {
                    "edge_retention_mean": float(np.mean(values["edge_retention"])) if values["edge_retention"] else None,
                    "edge_retention_median": float(np.median(values["edge_retention"])) if values["edge_retention"] else None,
                    "communities_median": float(np.median(values["communities"])) if values["communities"] else None,
                }
                for key, values in hard_variants.items()
            },
        },
    }


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    config = replace(config, data=replace(config.data, processed_path=args.processed_path))
    if args.dedup_mode != "none":
        config = replace(config, hypergraph=replace(config.hypergraph, dedup_mode=args.dedup_mode))
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

    def analyze_indices(indices: Iterable[int]) -> list[tuple[int, int, dict[str, Any]]]:
        results = []
        for index in indices:
            sample = dataset[index]
            records = builder.build_session(
                sample.x, sample.node_metadata, session_position=0, session_id=sample.session_id
            )
            results.append((int(index), int(sample.label), analyze_session(records)))
        return results

    train_analysis = analyze_indices(split.train)
    validation_analysis = analyze_indices(split.validation)

    result = {
        "_purpose": "Train/validation-only RA-IBG simulation; test split is not read.",
        "seed": args.seed,
        "dedup_mode": args.dedup_mode,
        "hypergraph_config": {
            "tau": config.hypergraph.tau,
            "threshold": config.hypergraph.threshold,
            "semantic_gate": config.hypergraph.semantic_gate,
            "time_weight": config.hypergraph.time_weight,
        },
        "metric_definition": {
            "u": "u(e) = 1 - max_{e'!=e, same session} Jaccard(e, e')；越低越冗余",
            "w_jaccard": "IBG 基线边权 = Jaccard(i,j)",
            "w_min": "RA-IBG 边权 = Jaccard(i,j) * min(u_i, u_j)",
            "w_prod": "RA-IBG 边权 = Jaccard(i,j) * u_i * u_j",
        },
        "train": summarize_sessions(train_analysis),
        "validation": summarize_sessions(validation_analysis),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
