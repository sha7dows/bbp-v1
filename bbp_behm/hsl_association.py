"""用途：以共享HGNN上下文学习STC成员贡献，保持关联支持和逐边质量，并单独保留二值IBG拓扑。"""

import torch
from torch import nn
from torch.nn import functional as F

from .hypergraph import BehaviorType
from .model import BBPBeHM


class AssociationBBPBeHM(BBPBeHM):
    """仅学习既有STC关联的连续权重；不删成员、不补关联、不修改BBP预算。"""

    def __init__(self, *args, association_mode="context", association_alpha=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        if association_mode not in {"none", "raw", "context"}:
            raise ValueError("Unknown association mode")
        if not 0 <= association_alpha <= 1:
            raise ValueError("association_alpha must be in [0, 1]")
        self.association_mode = association_mode
        self.association_alpha = float(association_alpha)
        hidden = self.config.model.hidden_size
        self.association_scorer = nn.Sequential(nn.Linear(4*hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
        # 初始分数为零，逐成员权重严格等于1，前向保持原BBP行为。
        nn.init.zeros_(self.association_scorer[-1].weight)
        nn.init.zeros_(self.association_scorer[-1].bias)
        self.last_association_stats = {}
        self.last_association_weights = None
        self._ibg_incidence = None

    def prepare_associations(self, projected_nodes, graph, incidence, hard_mask):
        # 此张量的前向是二值结构，仍保留原选择器的直通梯度路径。
        self._ibg_incidence = incidence
        self.last_association_stats = {}
        self.last_association_weights = None
        if self.association_mode == "none" or self.association_alpha == 0 or graph.num_edges == 0:
            return incidence
        stc = graph.type_ids == int(BehaviorType.STC)
        edges, nodes = torch.nonzero(graph.incidence * stc[:,None], as_tuple=True)
        if edges.numel() == 0:
            return incidence
        # 试探传播使用原候选图，避免第一次硬选择限制可见上下文；共享第一层参数。
        context = projected_nodes
        if self.association_mode == "context":
            context = F.relu(self.backbone.hgnn.first(projected_nodes,graph.incidence))
        node_features = F.layer_norm(context,(context.size(1),))
        edge_features = graph.incidence @ node_features / graph.incidence.sum(1,keepdim=True).clamp_min(1)
        edge_features = F.layer_norm(edge_features,(edge_features.size(1),))
        scores = []
        # 仅计算真实STC关联，分块避免建立E×N×隐藏维的全配对特征。
        for start in range(0,len(edges),16384):
            e,v = edges[start:start+16384],nodes[start:start+16384]
            a,b = node_features[v],edge_features[e]
            pair = torch.cat([a,b,a*b,(a-b).abs()],dim=1)
            scores.append(self.association_scorer(pair).flatten())
        logits = self.association_alpha * torch.tanh(torch.cat(scores))
        positive = logits.exp()
        denominator = positive.new_zeros(graph.num_edges).index_add(0,edges,positive)
        sizes = graph.incidence.sum(1)
        values = positive * sizes[edges] / denominator[edges].clamp_min(1e-8)
        weights = torch.ones_like(incidence).index_put((edges,nodes),values)
        self.last_association_weights = weights
        updated = incidence * weights
        selected = hard_mask.detach()[edges] > .5
        observed = values.detach()[selected]
        row_error = (updated.detach().sum(1)-incidence.detach().sum(1)).abs().max()
        zero = positive.detach().new_zeros(())
        self.last_association_stats = {
            "association_count": selected.sum().detach(),
            "association_abs_delta_sum": (observed-1).abs().sum(),
            "association_weight_min": observed.min() if observed.numel() else zero+1,
            "association_weight_max": observed.max() if observed.numel() else zero+1,
            "association_row_mass_max_error": row_error,
        }
        return updated

    def ibg_incidence(self, graph, propagation_incidence):
        """IBG按原二值支持构建，防止连续权重进入集合Jaccard公式。"""
        return self._ibg_incidence
