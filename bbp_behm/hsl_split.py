"""用途：诊断大STC边，并在父边预算不变的前提下生成保留来源及共享锚点的传播分解。"""

from dataclasses import replace
import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from .hypergraph import BehaviorType, HyperedgeBatch
from .model import BBPBeHM


def dispersion(features):
    """以成员两两余弦距离的均值衡量语义混杂，不使用任务标签。"""
    n = len(features)
    if n < 2:
        return 0.0
    unit = F.normalize(features.double(), dim=1)
    return float((n / (n - 1) * (1 - unit.mean(0).square().sum())).clamp_min(0))


def propose_split(record, sample, other_degree):
    """平衡二分；最多复制四个高共享度锚点，额外关联不超过父边的10%。"""
    by_id = {int(g): i for i, g in enumerate(sample.global_node_indices.tolist())}
    members = list(record.members)
    features = F.normalize(sample.x[[by_id[g] for g in members]].double(), dim=1)
    centrality = features @ features.mean(0)
    anchor_count = min(4, int(len(members) * 0.1), sum(other_degree.get(g, 0) > 0 for g in members))
    ranked = sorted(range(len(members)), key=lambda i: (-other_degree.get(members[i], 0), -float(centrality[i]), members[i]))
    anchors = tuple(sorted(members[i] for i in ranked[:anchor_count]))
    core = [g for g in members if g not in anchors]
    unit = features[[members.index(g) for g in core]]
    first = int(torch.argmin(unit @ unit.mean(0)))
    second = int(torch.argmin(unit @ unit[first]))
    first = int(torch.argmin(unit @ unit[second]))
    direction = unit[first] - unit[second]
    margin = (unit @ direction).tolist()
    order = sorted(range(len(core)), key=lambda i: (-margin[i], core[i]))
    half = len(core) // 2
    groups = [tuple(sorted(anchors + tuple(core[i] for i in ids))) for ids in (order[:half], order[half:])]
    # 随机对照逐父边复用完全相同的子边规模与锚点，且不消耗训练随机数。
    digest = hashlib.sha256((record.session_id + ":" + str(record.hyperedge_id)).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    permutation = rng.permutation(len(core)).tolist()
    random_groups = [tuple(sorted(anchors + tuple(core[i] for i in ids))) for ids in (permutation[:half], permutation[half:])]
    parent_dispersion = dispersion(sample.x[[by_id[g] for g in members]])
    child_dispersion = sum(len(gs) * dispersion(sample.x[[by_id[g] for g in gs]]) for gs in groups) / sum(map(len, groups))
    return {"parent_id": record.hyperedge_id, "members": members, "anchors": anchors,
            "semantic": groups, "random": random_groups, "dispersion": parent_dispersion,
            "gain": parent_dispersion - child_dispersion, "size": len(members)}


def fit_thresholds(training_records):
    """仅在当前训练折上拟合规模与混杂度阈值；凝聚提升下限预先固定。"""
    size = max(16, int(np.quantile([r["size"] for r in training_records], 0.75)))
    large = [r["dispersion"] for r in training_records if r["size"] >= size]
    return {"size": size, "dispersion": float(np.median(large)), "minimum_gain": 0.01,
            "maximum_anchor_count": 4, "maximum_extra_incidence_fraction": 0.1}


def eligible(plan, thresholds):
    return (plan["size"] >= thresholds["size"] and
            plan["dispersion"] >= thresholds["dispersion"] and
            plan["gain"] >= thresholds["minimum_gain"])


def child_record(parent, members, metadata, child_number):
    by_id = {int(m["global_node_idx"]): m for m in metadata}
    stamps = [by_id[g]["create_time"].timestamp() for g in members]
    # 构造强度仅作为父边来源属性继承；子边统计不参与父边选择打分。
    return replace(parent, members=tuple(members),
                   source_ids=parent.source_ids + (f"split-parent:{parent.session_id}:{parent.hyperedge_id}", f"split-child:{child_number}"),
                   time_span_days=(max(stamps) - min(stamps)) / 86400,
                   unique_user_count=len({str(by_id[g].get("user_id", "")) for g in members}))


class SplitCachedBuilder:
    """缓存子边拓扑并修正批记录编号；父边与子边使用独立的索引空间。"""

    def __init__(self, cache, plans, dataset, mode="none"):
        self.cache, self.plans, self.mode = cache, plans, mode
        self.expanded = {}
        for index, original in cache.items():
            sample = dataset[index]
            mapping = {int(g): i for i, g in enumerate(sample.global_node_indices.tolist())}
            self.expanded[index] = {}
            for variant in ("random", "semantic"):
                records, parents = [], []
                for r in original.records:
                    plan = plans.get(index, {}).get(r.hyperedge_id)
                    groups = plan[variant] if plan is not None else None
                    children = [child_record(r, group, sample.node_metadata, k) for k, group in enumerate(groups)] if groups else [r]
                    for child in children:
                        parents.append(r.hyperedge_id)
                        records.append(replace(child, hyperedge_id=len(records)))
                incidence = original.incidence.new_zeros((len(records), original.num_nodes))
                for r in records:
                    incidence[r.hyperedge_id, [mapping[g] for g in r.members]] = 1
                parent_rows = torch.tensor(parents, dtype=torch.long)
                stats = torch.tensor([[math.log1p(r.size), r.construction_strength, math.log1p(r.time_span_days),
                                       math.log1p(r.unique_user_count)] for r in records], dtype=original.structural_stats.dtype).reshape(-1,4)
                self.expanded[index][variant] = (HyperedgeBatch(incidence, original.edge_to_session[parent_rows],
                    original.type_ids[parent_rows], stats, records), parent_rows)

    @staticmethod
    def merge(graphs, device):
        records = []
        for session, graph in enumerate(graphs):
            for r in graph.records:
                records.append(replace(r, session_position=session, hyperedge_id=len(records)))
        return HyperedgeBatch(torch.block_diag(*[g.incidence.to(device) for g in graphs]),
            torch.cat([torch.full((g.num_edges,), i, dtype=torch.long, device=device) for i,g in enumerate(graphs)]),
            torch.cat([g.type_ids.to(device) for g in graphs]),
            torch.cat([g.structural_stats.to(device) for g in graphs]), records)

    def build_batch(self, batch):
        indices = batch.session_indices.cpu().tolist()
        graphs = [self.cache[int(i)] for i in indices]
        graph = self.merge(graphs, batch.x.device)
        if self.mode != "none":
            expansion = [self.expanded[int(i)][self.mode] for i in indices]
            graph.propagation = self.merge([g for g, _ in expansion], batch.x.device)
            offsets, count = [], 0
            for original, (_, rows) in zip(graphs, expansion):
                offsets.append(rows + count)
                count += original.num_edges
            graph.parent_rows = torch.cat(offsets).to(batch.x.device)
            graph.split_mode = self.mode
        return graph


class SplitBBPBeHM(BBPBeHM):
    """父边选择保持原BBP；显式开关仅影响传入HGNN和IBG的传播结构。"""

    def __init__(self, *args, mode="none", **kwargs):
        super().__init__(*args, **kwargs)
        if mode not in ("none", "random", "semantic"):
            raise ValueError("Unknown split mode")
        self.split_mode = mode

    def prepare_propagation(self, graph, incidence, hard_mask):
        if self.split_mode == "none":
            self.last_propagation_graph = graph
            self.last_parent_rows = torch.arange(graph.num_edges, device=incidence.device)
            return graph, incidence, hard_mask
        if getattr(graph, "split_mode", None) != self.split_mode:
            raise ValueError("Propagation cache does not match model split mode")
        parent_rows = graph.parent_rows
        transformed = graph.propagation
        # 向量化继承父边直通权重；预缓存拓扑避免逐边GPU分配和同步。
        actual = incidence[parent_rows] * transformed.incidence
        self.last_propagation_graph = transformed
        self.last_parent_rows = parent_rows
        return transformed, actual, hard_mask[parent_rows]
