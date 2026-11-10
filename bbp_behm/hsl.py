"""用途：实验性类 HSL 表征反馈与受预算约束的 STC 关联精简；不改变原 BBP 默认路径。"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .hypergraph import BehaviorType
from .model import BBPBeHM


class HSLBBPBeHM(BBPBeHM):
    """在原候选集合重选，并只对足够大的 STC 边执行有限成员删除。"""

    def __init__(self, *args, mode="feedback", edit_fraction=0.1, large_edge_threshold=32, **kwargs):
        super().__init__(*args, **kwargs)
        if mode not in {"bbp", "control", "feedback", "feedback_edit"}:
            raise ValueError("Unsupported HSL mode")
        if not 0.0 <= edit_fraction <= 0.2:
            raise ValueError("edit_fraction must be in [0, 0.2]")
        self.hsl_mode = mode
        self.edit_fraction = edit_fraction
        self.large_edge_threshold = max(2, int(large_edge_threshold))
        hidden = self.config.model.hidden_size
        self.feedback_adapter = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.member_scorer = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # 零初始化使反馈从恒等映射开始，避免初始重选纯粹由额外随机参数驱动。
        nn.init.zeros_(self.feedback_adapter[-1].weight)
        nn.init.zeros_(self.feedback_adapter[-1].bias)
        self.last_hsl_stats = {}
        self._pilot_nodes = None

    def select_hyperedges(self, **kwargs):
        self.last_hsl_stats = {"selection_changed": 0, "candidate_edges": kwargs["hypergraph"].num_edges}
        self._pilot_nodes = None
        if self.hsl_mode == "bbp":
            return self.selector(**kwargs)
        nodes, graph = kwargs["projected_nodes"], kwargs["hypergraph"]
        # 试探评分不消耗最终前向的随机数，使各组 Dropout 随机流可比。
        devices = [nodes.device.index] if nodes.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            initial = self.selector(**kwargs)
        # 仅初选支持被冻结，传播参数仍接受最终分类梯度；重选始终面向全部原候选。
        pilot_incidence = graph.incidence * initial.hard_mask.detach()[:, None]
        pilot = F.relu(self.backbone.hgnn.first(nodes, pilot_incidence))
        self._pilot_nodes = pilot
        # 控制组执行同样的试探传播，但适配器只使用原节点，区分上下文与额外容量。
        context = nodes if self.hsl_mode == "control" else pilot
        refined_nodes = nodes + self.feedback_adapter(torch.cat([nodes, context], dim=1))
        selected = self.selector(**{**kwargs, "projected_nodes": refined_nodes})
        self.last_hsl_stats["selection_changed"] = int((initial.hard_mask != selected.hard_mask).sum().detach())
        return selected

    def refine_incidence(self, projected_nodes, hypergraph, incidence, hard_mask):
        self.last_hsl_stats.update({"edits": 0, "eligible_stc": 0})
        if self.hsl_mode != "feedback_edit" or self.edit_fraction == 0.0:
            return incidence
        original_hard = incidence.detach() > 0.5
        degree = original_hard.sum(dim=0)
        non_stc = (hypergraph.type_ids != int(BehaviorType.STC)) & (hard_mask.detach() > 0.5)
        # 保留与 User/Mention 相交的节点，以及最后一条有效关联，避免直接破坏桥接和节点可达性。
        bridge_nodes = original_hard[non_stc].any(dim=0)
        result = incidence.clone()
        pilot = projected_nodes if self._pilot_nodes is None else self._pilot_nodes
        for edge in torch.nonzero((hypergraph.type_ids == int(BehaviorType.STC)) & (hard_mask.detach() > 0.5)).flatten().tolist():
            members = torch.nonzero(original_hard[edge]).flatten()
            size = int(members.numel())
            if size < self.large_edge_threshold:
                continue
            self.last_hsl_stats["eligible_stc"] += 1
            editable = members[(~bridge_nodes[members]) & (degree[members] > 1)]
            remove = min(int(math.floor(self.edit_fraction * size)), int(editable.numel()), size - 2)
            if remove <= 0:
                continue
            edge_mean = projected_nodes[members].mean(dim=0).expand(editable.numel(), -1)
            scores = self.member_scorer(torch.cat([projected_nodes[editable], pilot[editable], edge_mean], dim=1)).flatten()
            keep = int(editable.numel()) - remove
            hard = torch.zeros_like(scores)
            if keep:
                hard[torch.topk(scores, keep).indices] = 1.0
            soft = self.selector._budget_consistent_soft_mask(scores, keep, self.config.pruning.gumbel_temperature)
            mask = soft + (hard - soft).detach()
            result[edge, editable] = incidence[edge, editable] * mask
            removed = editable[hard == 0]
            degree[removed] -= 1
            self.last_hsl_stats["edits"] += remove
        # 覆盖仍以原候选为锚；实验脚本另行统计实际编辑量与传播结构的覆盖变化。
        return result
