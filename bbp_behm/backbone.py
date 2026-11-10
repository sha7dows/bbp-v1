"""用途：迁移并稳健化 BeHM 的 HGNN、IBG、社区与层次融合主干。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import community as community_louvain
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


def pad_and_create_mask(sequences: List[Tensor], hidden_dim: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    if not sequences:
        return torch.empty((0, 0, hidden_dim), device=device), torch.empty((0, 0), dtype=torch.bool, device=device)
    max_length = max(int(sequence.size(0)) for sequence in sequences)
    padded = torch.zeros((len(sequences), max_length, hidden_dim), dtype=sequences[0].dtype, device=device)
    mask = torch.zeros((len(sequences), max_length), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        length = int(sequence.size(0))
        if length:
            padded[index, :length] = sequence
            mask[index, :length] = True
    return padded, mask


class MaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, sequences: Tensor, mask: Tensor) -> Tensor:
        if sequences.size(1) == 0:
            return sequences.new_zeros((sequences.size(0), self.hidden_dim))
        scores = self.attention(sequences).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.sum(sequences * weights.unsqueeze(-1), dim=1)


class ContextAwareHierarchicalFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.global_pooling = MaskedAttentionPooling(hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)
        )

    def forward(
        self,
        node_sequences: Tensor,
        node_mask: Tensor,
        edge_sequences: Tensor,
        edge_mask: Tensor,
        community_sequences: Tensor,
        community_mask: Tensor,
    ) -> Tensor:
        global_context = self.global_pooling(community_sequences, community_mask)
        local_sequences = torch.cat([node_sequences, edge_sequences], dim=1)
        local_mask = torch.cat([node_mask, edge_mask], dim=1)
        query = self.query_projection(global_context).unsqueeze(1)
        keys = self.key_projection(local_sequences)
        values = self.value_projection(local_sequences)
        scores = torch.matmul(query, keys.transpose(1, 2)).squeeze(1) / (self.hidden_dim ** 0.5)
        scores = scores.masked_fill(~local_mask, -1e9)
        attention = torch.softmax(scores, dim=1)
        attention = attention * local_mask.to(attention.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attended = torch.sum(values * attention.unsqueeze(-1), dim=1)
        output = self.output_projection(global_context + attended)
        return output


class HGNNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, nodes: Tensor, incidence: Tensor) -> Tensor:
        nodes = nodes.float()
        incidence = incidence.float()
        node_degree_inv_sqrt = (incidence.sum(dim=0) + 1e-8).pow(-0.5)
        edge_degree_inv = (incidence.sum(dim=1) + 1e-8).pow(-1.0)
        normalized_nodes = nodes * node_degree_inv_sqrt[:, None]
        edge_messages = incidence @ normalized_nodes
        edge_messages = edge_messages * edge_degree_inv[:, None]
        node_messages = incidence.t() @ edge_messages
        node_messages = node_messages * node_degree_inv_sqrt[:, None]
        return self.linear(node_messages)


class HGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        use_residual: bool = False,
        edge_pooling: str = "sum",
    ) -> None:
        super().__init__()
        if edge_pooling not in {"sum", "mean"}:
            raise ValueError("edge_pooling must be 'sum' or 'mean'.")
        self.first = HGNNLayer(hidden_dim, hidden_dim)
        self.second = HGNNLayer(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.use_residual = use_residual
        self.edge_pooling = edge_pooling
        if use_residual:
            self.first_norm = nn.LayerNorm(hidden_dim)
            self.second_norm = nn.LayerNorm(hidden_dim)

    def _pool_edges(self, incidence: Tensor, nodes: Tensor) -> Tensor:
        """按配置聚合节点表示；mean 使用每条超边的实际关联节点数归一化。"""
        edge_features = incidence @ nodes
        if self.edge_pooling == "mean":
            sizes = incidence.sum(dim=1, keepdim=True).clamp_min(1.0)
            edge_features = edge_features / sizes
        return edge_features

    def forward(self, nodes: Tensor, incidence: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.use_residual:
            nodes = self.first(nodes, incidence)
            nodes = F.relu(nodes)
            nodes = F.dropout(nodes, self.dropout, training=self.training)
            nodes = self.second(nodes, incidence)
            return nodes, self._pool_edges(incidence, nodes)

        first_residual = nodes
        nodes = self.first(nodes, incidence)
        nodes = F.relu(nodes)
        nodes = F.dropout(nodes, self.dropout, training=self.training)
        nodes = self.first_norm(first_residual + nodes)

        second_residual = nodes
        nodes = self.second(nodes, incidence)
        nodes = self.second_norm(second_residual + nodes)
        return nodes, self._pool_edges(incidence, nodes)


class CommunityFeatureExtractor(nn.Module):
    def __init__(self, random_seed: int) -> None:
        super().__init__()
        self.random_seed = random_seed

    def forward(self, edge_index: Tensor, edge_weight: Tensor, edge_features: Tensor) -> Tensor:
        count = int(edge_features.size(0))
        if count == 0:
            return edge_features
        if edge_index.numel() == 0:
            # 无关联时每条超边均构成一个单节点社区，避免丢失孤立行为基。
            return edge_features

        graph = nx.Graph()
        graph.add_nodes_from(range(count))
        # 一次性搬运拓扑和权重，避免每条 IBG 关系触发 GPU/CPU 同步；插入顺序保持不变。
        pairs = edge_index.t().detach().cpu().tolist()
        weights = edge_weight.detach().cpu().tolist()
        for (source, target), weight in zip(pairs, weights):
            if source <= target:
                graph.add_edge(source, target, weight=float(weight))
        partition = community_louvain.best_partition(
            graph, weight="weight", random_state=self.random_seed
        )
        grouped: Dict[int, List[int]] = {}
        for edge_id in range(count):
            community_id = int(partition.get(edge_id, edge_id + count))
            grouped.setdefault(community_id, []).append(edge_id)
        return torch.stack([edge_features[node_ids].mean(dim=0) for node_ids in grouped.values()], dim=0)


@dataclass
class BackboneOutput:
    logits: Tensor
    session_representations: Tensor
    classification_loss: Tensor
    ibg_node_counts: List[int]
    ibg_edge_counts: List[int]


class BeHMBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        class_priors: Optional[Tensor] = None,
        logit_adjustment: str = "paper",
        louvain_seed: int = 42,
        positive_class_weight: float = 2.0,
        confidence_weighting: str = "raw",
        confidence_class_means: Optional[Tensor] = None,
        confidence_global_mean: Optional[Tensor] = None,
        hgnn_residual: bool = False,
        edge_pooling: str = "sum",
        ibg_weight_mode: str = "jaccard",
        ibg_hard_tau: Optional[float] = None,
    ) -> None:
        super().__init__()
        if logit_adjustment not in {"paper", "legacy", "none"}:
            raise ValueError("Unsupported logit adjustment mode.")
        if positive_class_weight <= 0.0:
            raise ValueError("positive_class_weight must be positive.")
        if confidence_weighting not in {"raw", "class_normalized", "none"}:
            raise ValueError("Unsupported confidence weighting mode.")
        if edge_pooling not in {"sum", "mean"}:
            raise ValueError("edge_pooling must be 'sum' or 'mean'.")
        if ibg_weight_mode not in {"jaccard", "min_u", "prod_u", "softmin", "softprod", "hard_min"}:
            raise ValueError("ibg_weight_mode must be jaccard, min_u, prod_u, softmin, softprod, or hard_min.")
        if ibg_weight_mode == "hard_min" and (ibg_hard_tau is None or not 0.0 < ibg_hard_tau <= 1.0):
            raise ValueError("hard_min requires 0 < ibg_hard_tau <= 1.")
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.logit_adjustment = logit_adjustment
        self.confidence_weighting = confidence_weighting
        self.ibg_weight_mode = ibg_weight_mode
        self.ibg_hard_tau = ibg_hard_tau
        priors = torch.full((num_classes,), 1.0 / num_classes) if class_priors is None else class_priors.float()
        priors = priors / priors.sum().clamp_min(1e-8)
        self.register_buffer("class_priors", priors)
        # 仅提升索引为 1 的正类权重，其余类别保持单位权重。
        weights = torch.ones(num_classes, dtype=torch.float32)
        if num_classes > 1:
            weights[1] = positive_class_weight
        self.register_buffer("class_weights", weights)
        if confidence_weighting == "class_normalized":
            class_means = (
                torch.ones(num_classes, dtype=torch.float32)
                if confidence_class_means is None
                else confidence_class_means.detach().float().clone()
            )
            global_mean = (
                torch.tensor(1.0, dtype=torch.float32)
                if confidence_global_mean is None
                else torch.as_tensor(confidence_global_mean, dtype=torch.float32).detach().clone()
            )
            if class_means.shape != (num_classes,):
                raise ValueError("confidence_class_means must match num_classes.")
            if not bool(torch.isfinite(class_means).all()) or bool((class_means <= 0.0).any()):
                raise ValueError("confidence_class_means must be finite and positive.")
            if global_mean.numel() != 1 or not bool(torch.isfinite(global_mean)) or float(global_mean) <= 0.0:
                raise ValueError("confidence_global_mean must be finite and positive.")
            # 仅新模式持久化训练集统计，保持 raw/none 与历史 checkpoint 严格兼容。
            self.register_buffer("confidence_class_means", class_means)
            self.register_buffer("confidence_global_mean", global_mean.reshape(()))

        self.initial_projection = nn.Linear(input_dim, hidden_dim)
        self.hgnn = HGNN(hidden_dim, dropout, use_residual=hgnn_residual, edge_pooling=edge_pooling)
        self.hyperedge_gnn = GCNConv(hidden_dim, hidden_dim)
        self.community_extractor = CommunityFeatureExtractor(louvain_seed)
        self.fusion = ContextAwareHierarchicalFusion(hidden_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # softplus(0.5413) 约等于 1，使论文 logit adjustment 从常用尺度开始。
        self.raw_adjustment = nn.Parameter(torch.tensor(0.5413))

    def project_nodes(self, nodes: Tensor) -> Tensor:
        projected = F.relu(self.initial_projection(nodes))
        return F.dropout(projected, self.dropout, training=self.training)

    @staticmethod
    def build_ibg(
        incidence: Tensor,
        weight_mode: str = "jaccard",
        hard_tau: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
        """按会话内超边 Jaccard 构建 IBG；weight_mode 决定边权形态，hard_tau 控制硬裁剪。

        jaccard：基线，边权=Jaccard(i,j)（与 BeHM 原始一致）。
        min_u / prod_u：边权=Jaccard * min(u_i,u_j) / u_i*u_j（激进软降权，模拟已证过弱）。
        softmin / softprod：边权=Jaccard * (0.5+0.5*min(u_i,u_j)) / (0.5+0.5*u_i*u_j)（温和软降权，已证无效）。
        hard_min：边权=Jaccard*min(u_i,u_j)，仅保留 >= hard_tau 的边（硬裁剪，改变拓扑）。
        其中 u(e)=1-max_{e'!=e} Jaccard(e,e') 为超边唯一信息量，仅由结构决定、无学习参数。
        """
        if incidence.size(0) == 0:
            return (
                torch.empty((2, 0), dtype=torch.long, device=incidence.device),
                torch.empty((0,), dtype=incidence.dtype, device=incidence.device),
            )
        intersection = incidence @ incidence.t()
        sizes = incidence.sum(dim=1, keepdim=True)
        union = (sizes + sizes.t() - intersection).clamp_min(1e-8)
        jaccard = intersection / union
        jaccard = jaccard - torch.diag_embed(torch.diagonal(jaccard))
        edge_index = torch.nonzero(jaccard > 0.0, as_tuple=False).t().long()
        if edge_index.numel() == 0:
            return edge_index, incidence.new_empty((0,))
        edge_weight = jaccard[edge_index[0], edge_index[1]]
        if weight_mode == "jaccard":
            return edge_index, edge_weight
        max_other = jaccard.max(dim=1).values.clamp(0.0, 1.0)
        uniqueness = (1.0 - max_other).clamp(0.0, 1.0)
        u_i = uniqueness[edge_index[0]]
        u_j = uniqueness[edge_index[1]]
        if weight_mode == "min_u":
            factor = torch.minimum(u_i, u_j)
        elif weight_mode == "prod_u":
            factor = u_i * u_j
        elif weight_mode == "softmin":
            factor = 0.5 + 0.5 * torch.minimum(u_i, u_j)
        elif weight_mode == "softprod":
            factor = 0.5 + 0.5 * u_i * u_j
        elif weight_mode == "hard_min":
            if hard_tau is None or not 0.0 < hard_tau <= 1.0:
                raise ValueError("hard_min weight mode requires 0 < hard_tau <= 1.")
            factor = torch.minimum(u_i, u_j)
            scaled = edge_weight * factor
            keep = scaled >= hard_tau
            return edge_index[:, keep], scaled[keep]
        else:
            raise ValueError(f"Unsupported ibg weight_mode: {weight_mode}")
        return edge_index, edge_weight * factor

    def _adjust_logits(self, logits: Tensor) -> Tensor:
        if self.logit_adjustment == "none":
            return logits
        scale = F.softplus(self.raw_adjustment)
        if self.logit_adjustment == "paper":
            return logits - scale * torch.log(self.class_priors.clamp_min(1e-8))
        return logits + scale * self.class_priors

    def _confidence_weights(self, confidences: Optional[Tensor], labels: Tensor) -> Tensor:
        """按配置生成样本权重；类别归一化统计只允许来自训练集。"""
        raw = torch.ones_like(labels, dtype=torch.float32) if confidences is None else confidences.float()
        if self.confidence_weighting == "none":
            return torch.ones_like(raw)
        if self.confidence_weighting == "raw":
            return raw
        class_means = self.confidence_class_means[labels]
        return raw * self.confidence_global_mean / class_means.clamp_min(1e-8)

    def forward_projected(
        self,
        projected_nodes: Tensor,
        incidence: Tensor,
        graph_ids: Tensor,
        edge_to_session: Tensor,
        hard_edge_mask: Tensor,
        num_sessions: int,
        labels: Optional[Tensor] = None,
        confidences: Optional[Tensor] = None,
        topology_incidence: Optional[Tensor] = None,
    ) -> BackboneOutput:
        # 传播关联可以连续加权；IBG集合拓扑由独立输入提供，默认兼容旧调用。
        topology = incidence if topology_incidence is None else topology_incidence
        if topology.shape != incidence.shape:
            raise ValueError("Topology and propagation incidence shapes differ")
        propagated_nodes, all_edge_features = self.hgnn(projected_nodes, incidence)
        node_sequences: List[Tensor] = []
        edge_sequences: List[Tensor] = []
        community_sequences: List[Tensor] = []
        ibg_node_counts: List[int] = []
        ibg_edge_counts: List[int] = []

        for session in range(num_sessions):
            node_sequences.append(propagated_nodes[graph_ids == session])
            session_edges = torch.nonzero(
                (edge_to_session == session) & (hard_edge_mask > 0.5), as_tuple=False
            ).flatten()
            if session_edges.numel() == 0:
                empty = projected_nodes.new_empty((0, self.hidden_dim))
                edge_sequences.append(empty)
                community_sequences.append(empty)
                ibg_node_counts.append(0)
                ibg_edge_counts.append(0)
                continue

            session_incidence = topology[session_edges]
            session_features = all_edge_features[session_edges]
            edge_index, edge_weight = self.build_ibg(
                session_incidence, self.ibg_weight_mode, self.ibg_hard_tau
            )
            updated_edges = self.hyperedge_gnn(session_features, edge_index, edge_weight=edge_weight)
            communities = self.community_extractor(edge_index, edge_weight, updated_edges)
            edge_sequences.append(updated_edges)
            community_sequences.append(communities)
            ibg_node_counts.append(int(session_edges.numel()))
            ibg_edge_counts.append(int(edge_index.size(1)))

        node_sequence, node_mask = pad_and_create_mask(node_sequences, self.hidden_dim, projected_nodes.device)
        edge_sequence, edge_mask = pad_and_create_mask(edge_sequences, self.hidden_dim, projected_nodes.device)
        community_sequence, community_mask = pad_and_create_mask(
            community_sequences, self.hidden_dim, projected_nodes.device
        )
        session_representations = self.fusion(
            node_sequence,
            node_mask,
            edge_sequence,
            edge_mask,
            community_sequence,
            community_mask,
        )
        logits = self._adjust_logits(self.classifier(session_representations))
        classification_loss = logits.sum() * 0.0
        if labels is not None:
            per_session = F.cross_entropy(logits, labels, weight=self.class_weights, reduction="none")
            weights = self._confidence_weights(confidences, labels)
            classification_loss = torch.mean(per_session * weights)
        return BackboneOutput(
            logits=logits,
            session_representations=session_representations,
            classification_loss=classification_loss,
            ibg_node_counts=ibg_node_counts,
            ibg_edge_counts=ibg_edge_counts,
        )
