"""用途：实现冻结节点反馈和分类型 IBG 行为上下文反馈，支持容量控制与原 BBP 关闭路径。"""

import torch
from torch import nn
from torch.nn import functional as F

from .hsl import HSLBBPBeHM
from .selector import EdgeSelector


class ContextEdgeSelector(EdgeSelector):
    """在原任务评分上叠加有界的行为上下文残差，不改变覆盖几何。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.hidden_dim
        self.context_scorer = nn.Sequential(nn.Linear(5 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.context_scorer[-1].weight)
        nn.init.zeros_(self.context_scorer[-1].bias)
        self.context_features = None
        self.last_delta = None

    def score_edges(self, projected_nodes, graph_ids, num_sessions, hypergraph):
        scores, shared, session, embeddings = super().score_edges(projected_nodes, graph_ids, num_sessions, hypergraph)
        self.last_delta = None
        if self.context_features is not None and hypergraph.num_edges:
            features = F.layer_norm(self.context_features, (self.context_features.size(1),))
            delta = 0.25 * torch.tanh(self.context_scorer(features).flatten())
            self.last_delta = delta
            scores = scores + delta
            session = session + delta
        return scores, shared, session, embeddings


class ContextHSLBBPBeHM(HSLBBPBeHM):
    """先进行一次预算选择，再用节点或行为上下文对全部候选边重评分。"""

    MODES = {"bbp", "node_control", "node_detach", "edge_control", "edge_context"}

    def __init__(self, *args, mode="edge_context", **kwargs):
        if mode not in self.MODES:
            raise ValueError("Unsupported context mode")
        super().__init__(*args, mode="bbp", **kwargs)
        self.context_mode = mode
        previous = self.selector
        selector = ContextEdgeSelector(
            hidden_dim=self.config.model.hidden_size,
            type_embedding_dim=self.config.pruning.type_embedding_dim,
            dropout=self.selector_dropout,
            use_ibg_centrality=self.config.pruning.use_ibg_centrality,
        )
        selector.load_state_dict(previous.state_dict(), strict=False)
        self.selector = selector

    def _edge_context(self, nodes, graph, initial, num_sessions):
        embeddings = self.selector.encode_edges(nodes, graph.incidence)
        contexts = [torch.zeros_like(embeddings) for _ in range(4)]
        # 邻域特征全部冻结；只有评分器自身和原任务输入通过分类损失更新。
        with torch.no_grad():
            detached = embeddings.detach()
            for session in range(num_sessions):
                ids = torch.nonzero(graph.edge_to_session == session).flatten()
                if not ids.numel():
                    continue
                incidence = graph.incidence[ids]
                intersection = incidence @ incidence.t()
                sizes = incidence.sum(1)
                weights = intersection / (sizes[:, None] + sizes[None, :] - intersection).clamp_min(1)
                kept = initial.hard_mask.detach()[ids]
                weighted = weights * kept[None, :]
                local = detached[ids]
                for kind in range(3):
                    sources = kept * (graph.type_ids[ids] == kind)
                    actual = weighted * (graph.type_ids[ids] == kind)[None, :]
                    related = actual @ local / actual.sum(1, keepdim=True).clamp_min(1e-8)
                    # 控制组同样计算邻域，但评分使用类型全局均值，取消边特定关系信息。
                    global_type = (sources[:, None] * local).sum(0) / sources.sum().clamp_min(1)
                    contexts[kind][ids] = global_type if self.context_mode == "edge_control" else related
                global_kept = (kept[:, None] * local).sum(0) / kept.sum().clamp_min(1)
                contexts[3][ids] = global_kept
        return torch.cat([embeddings, *contexts], dim=1)

    def select_hyperedges(self, **kwargs):
        graph, nodes = kwargs["hypergraph"], kwargs["projected_nodes"]
        self.last_hsl_stats = {"selection_changed": 0, "candidate_edges": graph.num_edges}
        self._pilot_nodes = None
        self.selector.context_features = None
        if self.context_mode == "bbp":
            return self.selector(**kwargs)
        devices = [nodes.device.index] if nodes.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            initial = self.selector(**kwargs)
        if self.context_mode.startswith("node_"):
            # 结构支持和试探输出均冻结，消除共享 HGNN 接收额外反馈梯度的路径。
            with torch.no_grad():
                pilot = F.relu(self.backbone.hgnn.first(nodes.detach(), graph.incidence * initial.hard_mask.detach()[:, None]))
            context = nodes if self.context_mode == "node_control" else pilot
            refined = nodes + self.feedback_adapter(torch.cat([nodes, context], dim=1))
            selected = self.selector(**{**kwargs, "projected_nodes": refined})
        else:
            self.selector.context_features = self._edge_context(nodes, graph, initial, kwargs["num_sessions"])
            try:
                selected = self.selector(**kwargs)
            finally:
                # 清除本批次图，防止下一次前向使用陈旧上下文或保留计算图。
                self.selector.context_features = None
        self.last_hsl_stats["selection_changed"] = int((initial.hard_mask != selected.hard_mask).sum().detach())
        return selected
