"""用途：实现共享—会话整边评分、精确预算选择和同类型行为基代表性目标。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .hypergraph import HyperedgeBatch


def segment_mean(values: Tensor, segment_ids: Tensor, num_segments: int) -> Tensor:
    if values.ndim != 2:
        raise ValueError("segment_mean expects a two-dimensional value tensor.")
    output = values.new_zeros((num_segments, values.size(1)))
    counts = values.new_zeros((num_segments, 1))
    if values.size(0) > 0:
        output.index_add_(0, segment_ids, values)
        counts.index_add_(0, segment_ids, values.new_ones((values.size(0), 1)))
    return output / counts.clamp_min(1.0)


@dataclass
class SelectionOutput:
    scores: Tensor
    shared_scores: Tensor
    session_scores: Tensor
    hard_mask: Tensor
    soft_mask: Tensor
    straight_through_mask: Tensor
    kept_per_session: Dict[int, int]
    retention_per_session: Dict[int, float]
    coverage_per_session: Dict[int, float]
    tail_coverage_per_session: Dict[int, float]
    tail_coverage_by_type_per_session: Dict[int, Dict[int, float]]
    stopping_reason_per_session: Dict[int, str]
    budget_mode: str
    coverage_threshold: float
    coverage_q: float
    coverage_eta: float
    proxy_scores: Tensor
    edge_embeddings: Tensor
    stable_embeddings: Tensor


@dataclass
class _TypeCoverageContext:
    """保存单个行为类型的全局边编号与固定代表矩阵。"""

    edge_ids: Tensor
    representative: Tensor


class EdgeSelector(nn.Module):
    """在会话级精确 Top-K 预算下选择完整原始超边。"""

    def __init__(
        self,
        hidden_dim: int,
        type_embedding_dim: int = 16,
        stats_dim: int = 4,
        dropout: float = 0.2,
        use_ibg_centrality: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stats_dim = stats_dim
        self.use_ibg_centrality = use_ibg_centrality
        self.type_embedding = nn.Embedding(3, type_embedding_dim)
        self.stats_norm = nn.LayerNorm(stats_dim)
        edge_input_dim = hidden_dim + type_embedding_dim + stats_dim + int(use_ibg_centrality)
        self.shared_scorer = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.session_scorer = nn.Sequential(
            nn.Linear(edge_input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # 覆盖空间必须独立于分类训练保持稳定，三种核在主模型中固定等权且不写入检查点。
        self.register_buffer("kernel_weights", torch.full((3,), 1.0 / 3.0), persistent=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # 旧模型曾保存可训练 kernel_logits；新架构固定等权，加载时主动丢弃历史值。
        state_dict.pop(prefix + "kernel_logits", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def encode_edges(projected_nodes: Tensor, incidence: Tensor) -> Tensor:
        if incidence.size(0) == 0:
            return projected_nodes.new_empty((0, projected_nodes.size(1)))
        degrees = incidence.sum(dim=1, keepdim=True).clamp_min(1.0)
        return incidence @ projected_nodes / degrees

    @staticmethod
    def encode_edges_raw(node_features: Tensor, incidence: Tensor) -> Tensor:
        """成员节点的原始特征均值，用于稳定的覆盖几何；不经过任何可训练投影。"""
        if incidence.size(0) == 0:
            return node_features.new_empty((0, node_features.size(1)))
        degrees = incidence.sum(dim=1, keepdim=True).clamp_min(1.0)
        return incidence @ node_features / degrees

    @staticmethod
    def ibg_weighted_degree_centrality(
        incidence: Tensor,
        edge_to_session: Tensor,
        num_sessions: int,
    ) -> Tensor:
        """按 MSH 定义计算会话内 Jaccard IBG 的归一化加权度。"""
        centrality = incidence.new_zeros((incidence.size(0),))
        for session in range(num_sessions):
            indices = torch.nonzero(
                edge_to_session == session, as_tuple=False
            ).flatten()
            if indices.numel() <= 1:
                continue

            local_incidence = incidence[indices].float()
            intersection = local_incidence @ local_incidence.t()
            sizes = local_incidence.sum(dim=1, keepdim=True)
            union = sizes + sizes.t() - intersection
            jaccard = intersection / union.clamp_min(1e-8)
            jaccard.fill_diagonal_(0.0)

            # MSH 的对称有向边表会在两个端点各累加一次，因此等价于两倍行和。
            local_degree = 2.0 * jaccard.sum(dim=1)
            local_degree = local_degree / local_degree.max().clamp_min(1.0)
            centrality[indices] = local_degree.to(centrality.dtype)
        return centrality

    def score_edges(
        self,
        projected_nodes: Tensor,
        graph_ids: Tensor,
        num_sessions: int,
        hypergraph: HyperedgeBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        edge_embeddings = self.encode_edges(projected_nodes, hypergraph.incidence)
        if hypergraph.num_edges == 0:
            empty = projected_nodes.new_empty((0,))
            return empty, empty, empty, edge_embeddings

        type_features = self.type_embedding(hypergraph.type_ids)
        if hypergraph.structural_stats.size(1) != self.stats_dim:
            raise ValueError(
                f"Expected {self.stats_dim} structural statistics per edge, "
                f"got {hypergraph.structural_stats.size(1)}."
            )
        stats = self.stats_norm(hypergraph.structural_stats)
        scorer_features = [edge_embeddings, type_features, stats]
        if self.use_ibg_centrality:
            centrality = self.ibg_weighted_degree_centrality(
                hypergraph.incidence,
                hypergraph.edge_to_session,
                num_sessions,
            )
            scorer_features.append(centrality.unsqueeze(1))
        edge_input = torch.cat(scorer_features, dim=1)
        session_context = segment_mean(projected_nodes, graph_ids, num_sessions)
        edge_context = session_context[hypergraph.edge_to_session]
        shared_scores = self.shared_scorer(edge_input).squeeze(-1)
        session_scores = self.session_scorer(torch.cat([edge_input, edge_context], dim=1)).squeeze(-1)
        return shared_scores + session_scores, shared_scores, session_scores, edge_embeddings

    @staticmethod
    def _sample_gumbel_like(scores: Tensor) -> Tensor:
        uniform = torch.rand_like(scores).clamp_(1e-6, 1.0 - 1e-6)
        return -torch.log(-torch.log(uniform))

    def select(
        self,
        scores: Tensor,
        edge_to_session: Tensor,
        num_sessions: int,
        retention_ratio: float,
        temperature: float,
        stochastic: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Dict[int, int]]:
        if not 0.0 < retention_ratio <= 1.0:
            raise ValueError("retention_ratio must be in (0, 1].")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if scores.numel() == 0:
            empty = scores.new_empty((0,))
            return empty, empty, empty, {session: 0 for session in range(num_sessions)}

        perturbed = scores + self._sample_gumbel_like(scores) if stochastic else scores
        hard_mask = torch.zeros_like(scores)
        soft_mask = torch.zeros_like(scores)
        kept_per_session: Dict[int, int] = {}

        for session in range(num_sessions):
            indices = torch.nonzero(edge_to_session == session, as_tuple=False).flatten()
            count = int(indices.numel())
            if count == 0:
                kept_per_session[session] = 0
                continue
            keep = max(1, min(count, int(math.ceil(retention_ratio * count))))
            kept_per_session[session] = keep
            local_scores = perturbed[indices]
            if keep == count:
                local_hard = torch.ones_like(local_scores)
                local_soft = torch.ones_like(local_scores)
            else:
                selected = torch.topk(local_scores, k=keep, sorted=False).indices
                local_hard = torch.zeros_like(local_scores)
                local_hard[selected] = 1.0
                local_soft = torch.sigmoid(local_scores / temperature)
            hard_mask[indices] = local_hard
            soft_mask[indices] = local_soft

        straight_through = soft_mask + (hard_mask - soft_mask).detach()
        return hard_mask, soft_mask, straight_through, kept_per_session

    @staticmethod
    def _budget_consistent_soft_mask(
        local_scores: Tensor,
        keep: int,
        temperature: float,
        iterations: int = 32,
    ) -> Tensor:
        """构造总质量等于K的连续mask，使反向路径表达固定预算竞争。"""
        count = int(local_scores.numel())
        if keep >= count:
            return torch.ones_like(local_scores)
        if keep <= 0:
            return torch.zeros_like(local_scores)

        # 求解sum(sigmoid((score-threshold)/T))=K；二分过程保留分段梯度。
        lower = local_scores.min() - 20.0 * temperature
        upper = local_scores.max() + 20.0 * temperature
        target = local_scores.new_tensor(float(keep))
        for _ in range(iterations):
            midpoint = (lower + upper) * 0.5
            mass = torch.sigmoid((local_scores - midpoint) / temperature).sum()
            move_lower = mass > target
            lower = torch.where(move_lower, midpoint, lower)
            upper = torch.where(move_lower, upper, midpoint)
        threshold = (lower + upper) * 0.5
        return torch.sigmoid((local_scores - threshold) / temperature)

    @staticmethod
    def _normalize_scores(values: Tensor) -> Tensor:
        """在会话内部归一化到[0,1]；常量输入不制造伪差异。"""
        if values.numel() == 0:
            return values
        lower = values.min()
        span = values.max() - lower
        if float(span.detach().cpu()) <= 1e-12:
            return torch.zeros_like(values)
        return (values - lower) / span

    @staticmethod
    def _lower_tail_mean(values: Tensor, coverage_q: float) -> Tensor:
        """计算覆盖最低q比例的均值，至少包含一条边。"""
        if not 0.0 < coverage_q <= 1.0:
            raise ValueError("coverage_q must be in (0, 1].")
        if values.numel() == 0:
            return values.new_tensor(1.0)
        tail_count = max(1, int(math.ceil(coverage_q * int(values.numel()))))
        return torch.topk(values, k=tail_count, largest=False, sorted=False).values.mean()

    def _representativeness_matrix(
        self,
        incidence: Tensor,
        stable_embeddings: Tensor,
        stats: Tensor,
    ) -> Tensor:
        intersection = incidence @ incidence.t()
        sizes = incidence.sum(dim=1, keepdim=True)
        union = (sizes + sizes.t() - intersection).clamp_min(1e-8)
        structural = (intersection / union).clamp(0.0, 1.0)

        normalized_edges = F.normalize(stable_embeddings, p=2, dim=1)
        semantic = ((normalized_edges @ normalized_edges.t()) + 1.0) * 0.5
        semantic = semantic.clamp(0.0, 1.0)

        normalized_stats = F.layer_norm(stats, (stats.size(1),))
        context_distance = (normalized_stats[:, None, :] - normalized_stats[None, :, :]).abs().mean(dim=-1)
        context = torch.exp(-context_distance)

        weights = self.kernel_weights.to(dtype=structural.dtype)
        representative = weights[0] * structural + weights[1] * semantic + weights[2] * context
        # 被保留边应覆盖自身；被删边的掩码为0，因此仍只能由其它同类型保留边代表。
        identity = torch.eye(representative.size(0), dtype=representative.dtype, device=representative.device)
        representative = representative * (1.0 - identity) + identity
        return representative.clamp(0.0, 1.0)

    def _build_type_contexts(
        self,
        hypergraph: HyperedgeBatch,
        stable_embeddings: Tensor,
        session_indices: Tensor,
    ) -> Dict[int, _TypeCoverageContext]:
        """仅为会话中实际存在的行为类型构造覆盖上下文。"""
        contexts: Dict[int, _TypeCoverageContext] = {}
        for type_id in torch.unique(hypergraph.type_ids[session_indices]).tolist():
            local = session_indices[hypergraph.type_ids[session_indices] == int(type_id)]
            contexts[int(type_id)] = _TypeCoverageContext(
                edge_ids=local,
                representative=self._representativeness_matrix(
                    hypergraph.incidence[local],
                    stable_embeddings[local],
                    hypergraph.structural_stats[local],
                ),
            )
        return contexts

    @staticmethod
    def _initialize_type_best(
        contexts: Dict[int, _TypeCoverageContext],
    ) -> Dict[int, Tensor]:
        return {
            type_id: context.representative.new_zeros((context.representative.size(0),))
            for type_id, context in contexts.items()
        }

    def _robust_coverage_state(
        self,
        best_by_type: Dict[int, Tensor],
        coverage_q: float,
        coverage_tail_gamma: float,
    ) -> Tuple[Tensor, Tensor, Dict[int, Tensor]]:
        """分别返回严格停止覆盖、平滑缺口进度和各类型尾部覆盖。"""
        if not best_by_type:
            one = self.kernel_weights.new_tensor(1.0)
            return one, one, {}
        type_tail = {
            type_id: self._lower_tail_mean(values, coverage_q)
            for type_id, values in best_by_type.items()
        }
        stacked = torch.stack(list(type_tail.values()))
        stopping_coverage = stacked.min()
        # 选择进度按类型平均，且已达标类型不再继续贡献收益；停止仍使用最差类型。
        progress = torch.clamp(stacked / coverage_tail_gamma, max=1.0).mean()
        return stopping_coverage, progress, type_tail

    def _candidate_robust_progress_gains(
        self,
        contexts: Dict[int, _TypeCoverageContext],
        best_by_type: Dict[int, Tensor],
        session_indices: Tensor,
        coverage_q: float,
        coverage_tail_gamma: float,
    ) -> Tensor:
        """计算候选边对分类型截断覆盖进度的边际改善，避免全局min零增益。"""
        if not contexts:
            return self.kernel_weights.new_zeros((session_indices.numel(),))
        sample = next(iter(contexts.values())).representative
        gains = sample.new_zeros((session_indices.numel(),))
        type_count = max(len(contexts), 1)
        for type_id, context in contexts.items():
            current_tail = self._lower_tail_mean(best_by_type[type_id], coverage_q)
            updated = torch.maximum(
                best_by_type[type_id][:, None], context.representative
            )
            tail_count = max(
                1, int(math.ceil(coverage_q * int(updated.size(0))))
            )
            candidate_tail = torch.topk(
                updated,
                k=tail_count,
                dim=0,
                largest=False,
                sorted=False,
            ).values.mean(dim=0)
            current_progress = torch.clamp(
                current_tail / coverage_tail_gamma, max=1.0
            )
            candidate_progress = torch.clamp(
                candidate_tail / coverage_tail_gamma, max=1.0
            )
            # session_indices由nonzero产生并保持升序，可用searchsorted映射回会话局部位置。
            local_positions = torch.searchsorted(session_indices, context.edge_ids)
            gains[local_positions] = (
                candidate_progress - current_progress
            ) / type_count
        return gains

    def _robust_greedy_session(
        self,
        scores: Tensor,
        session_indices: Tensor,
        contexts: Dict[int, _TypeCoverageContext],
        coverage_q: float,
        coverage_tail_gamma: float,
        coverage_eta: float,
        min_keep: int,
        max_keep: int,
    ) -> Tuple[Tensor, Tensor, float, str, Dict[int, float]]:
        """用任务评分和覆盖缺口联合选择，并构造支持一致的反向优先级。"""
        count = int(session_indices.numel())
        local_task = self._normalize_scores(scores[session_indices])
        selected = torch.zeros((count,), dtype=torch.bool, device=scores.device)
        selected_order: List[int] = []
        best_by_type = self._initialize_type_best(contexts)
        edge_locations: Dict[int, Tuple[int, int]] = {}
        for type_id, context in contexts.items():
            for position, edge_id in enumerate(context.edge_ids.tolist()):
                edge_locations[int(edge_id)] = (type_id, position)

        stopping_value = scores.new_tensor(0.0)
        last_utility = local_task.new_zeros(local_task.shape)
        type_tail: Dict[int, Tensor] = {}
        reason = "maximum"
        for step in range(1, max_keep + 1):
            raw_gains = self._candidate_robust_progress_gains(
                contexts,
                best_by_type,
                session_indices,
                coverage_q,
                coverage_tail_gamma,
            )
            normalized_gains = self._normalize_scores(raw_gains).detach()
            utility = coverage_eta * local_task + (1.0 - coverage_eta) * normalized_gains
            last_utility = utility
            chosen_local = int(
                torch.argmax(utility.masked_fill(selected, -torch.inf)).item()
            )
            selected[chosen_local] = True
            selected_order.append(chosen_local)
            chosen_edge = int(session_indices[chosen_local].item())
            type_id, position = edge_locations[chosen_edge]
            best_by_type[type_id] = torch.maximum(
                best_by_type[type_id], contexts[type_id].representative[:, position]
            )
            stopping_value, _, type_tail = self._robust_coverage_state(
                best_by_type, coverage_q, coverage_tail_gamma
            )
            if step >= min_keep and float(stopping_value.detach().cpu()) >= coverage_tail_gamma:
                reason = "coverage_full" if step == count else "coverage"
                break
            if step == max_keep:
                # max_keep=count时完整掩码覆盖必为1；all_edges仅作为不变量失效的防御信号。
                reason = "all_edges" if max_keep == count else "maximum"

        kept = len(selected_order)
        # 数值优先级严格复现贪心支持和次序；梯度只由联合效用中的任务项提供。
        target_priority = self._normalize_scores(last_utility.detach())
        for rank, local_index in enumerate(selected_order):
            target_priority[local_index] = 2.0 + float(kept - rank) / max(kept, 1)
        proxy_scores = target_priority + coverage_eta * (local_task - local_task.detach())
        selected_edges = session_indices[selected]
        type_tail_values = {
            type_id: float(value.detach().cpu()) for type_id, value in type_tail.items()
        }
        return (
            selected_edges,
            proxy_scores,
            float(stopping_value.detach().cpu()),
            reason,
            type_tail_values,
        )

    def select_robust_adaptive(
        self,
        scores: Tensor,
        hypergraph: HyperedgeBatch,
        stable_embeddings: Tensor,
        num_sessions: int,
        coverage_q: float,
        coverage_tail_gamma: float,
        coverage_eta: float,
        min_retention_ratio: float,
        max_retention_ratio: float,
        temperature: float,
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Dict[int, int],
        Dict[int, float],
        Dict[int, str],
        Dict[int, Dict[int, float]],
        Tensor,
    ]:
        """执行确定性R2联合贪心；训练与评估不额外加入Gumbel拓扑扰动。"""
        if not 0.0 < coverage_q <= 1.0:
            raise ValueError("coverage_q must be in (0, 1].")
        if not 0.0 < coverage_tail_gamma <= 1.0:
            raise ValueError("coverage_tail_gamma must be in (0, 1].")
        if not 0.0 <= coverage_eta <= 1.0:
            raise ValueError("coverage_eta must be in [0, 1].")
        if not 0.0 <= min_retention_ratio <= max_retention_ratio <= 1.0:
            raise ValueError("adaptive retention bounds must satisfy 0 <= min <= max <= 1.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if scores.numel() == 0:
            empty = scores.new_empty((0,))
            zeros = {session: 0 for session in range(num_sessions)}
            ones = {session: 1.0 for session in range(num_sessions)}
            reasons = {session: "no_edges" for session in range(num_sessions)}
            by_type = {session: {} for session in range(num_sessions)}
            return empty, empty, empty, zeros, ones, reasons, by_type, empty

        hard_mask = torch.zeros_like(scores)
        soft_mask = torch.zeros_like(scores)
        proxy_scores = torch.zeros_like(scores)
        kept_per_session: Dict[int, int] = {}
        coverage_per_session: Dict[int, float] = {}
        reasons: Dict[int, str] = {}
        coverage_by_type: Dict[int, Dict[int, float]] = {}

        for session in range(num_sessions):
            indices = torch.nonzero(
                hypergraph.edge_to_session == session, as_tuple=False
            ).flatten()
            count = int(indices.numel())
            if count == 0:
                kept_per_session[session] = 0
                coverage_per_session[session] = 1.0
                reasons[session] = "no_edges"
                coverage_by_type[session] = {}
                continue
            min_keep = max(
                1, min(count, int(math.ceil(min_retention_ratio * count)))
            )
            max_keep = max(
                min_keep, min(count, int(math.ceil(max_retention_ratio * count)))
            )
            contexts = self._build_type_contexts(
                hypergraph, stable_embeddings, indices
            )
            selected, local_proxy, coverage, reason, type_tail = (
                self._robust_greedy_session(
                    scores,
                    indices,
                    contexts,
                    coverage_q,
                    coverage_tail_gamma,
                    coverage_eta,
                    min_keep,
                    max_keep,
                )
            )
            keep = int(selected.numel())
            hard_mask[selected] = 1.0
            proxy_scores[indices] = local_proxy
            soft_mask[indices] = self._budget_consistent_soft_mask(
                local_proxy, keep, temperature
            )
            kept_per_session[session] = keep
            coverage_per_session[session] = coverage
            reasons[session] = reason
            coverage_by_type[session] = type_tail

        straight_through = soft_mask + (hard_mask - soft_mask).detach()
        return (
            hard_mask,
            soft_mask,
            straight_through,
            kept_per_session,
            coverage_per_session,
            reasons,
            coverage_by_type,
            proxy_scores,
        )

    def _type_balanced_coverage(
        self,
        hypergraph: HyperedgeBatch,
        stable_embeddings: Tensor,
        session_indices: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """计算会话内按类型等权的代表覆盖；未保留类型的覆盖为0。"""
        if session_indices.numel() == 0:
            return stable_embeddings.sum() * 0.0
        type_coverages: List[Tensor] = []
        for type_id in torch.unique(hypergraph.type_ids[session_indices]).tolist():
            local = session_indices[hypergraph.type_ids[session_indices] == int(type_id)]
            representative = self._representativeness_matrix(
                hypergraph.incidence[local],
                stable_embeddings[local],
                hypergraph.structural_stats[local],
            )
            representative_strength = representative * mask[local].clamp(0.0, 1.0)[None, :]
            edge_coverage = representative_strength.max(dim=1).values
            type_coverages.append(edge_coverage.mean())
        return torch.stack(type_coverages).mean()

    def _coverage_statistics(
        self,
        hypergraph: HyperedgeBatch,
        stable_embeddings: Tensor,
        mask: Tensor,
        num_sessions: int,
    ) -> Dict[int, float]:
        result: Dict[int, float] = {}
        for session in range(num_sessions):
            indices = torch.nonzero(hypergraph.edge_to_session == session, as_tuple=False).flatten()
            if indices.numel() == 0:
                result[session] = 1.0
                continue
            coverage = self._type_balanced_coverage(hypergraph, stable_embeddings, indices, mask)
            result[session] = float(coverage.detach().cpu().item())
        return result

    def _adaptive_keep_count(
        self,
        ranked_indices: Tensor,
        hypergraph: HyperedgeBatch,
        stable_embeddings: Tensor,
        coverage_threshold: float,
        min_keep: int,
        max_keep: int,
    ) -> Tuple[int, float, str]:
        """沿任务评分前缀增量计算覆盖，返回首次满足阈值的最小K。"""
        session_indices = ranked_indices
        type_matrices: Dict[int, Tensor] = {}
        type_best: Dict[int, Tensor] = {}
        edge_locations: Dict[int, Tuple[int, int]] = {}

        for type_id in torch.unique(hypergraph.type_ids[session_indices]).tolist():
            local = session_indices[hypergraph.type_ids[session_indices] == int(type_id)]
            matrix = self._representativeness_matrix(
                hypergraph.incidence[local],
                stable_embeddings[local],
                hypergraph.structural_stats[local],
            )
            type_key = int(type_id)
            type_matrices[type_key] = matrix
            type_best[type_key] = matrix.new_zeros((local.numel(),))
            for position, edge_id in enumerate(local.tolist()):
                edge_locations[int(edge_id)] = (type_key, position)

        coverage_value = 0.0
        keep = max_keep
        reason = "maximum"
        for prefix, edge_id in enumerate(ranked_indices.tolist(), start=1):
            type_key, position = edge_locations[int(edge_id)]
            type_best[type_key] = torch.maximum(
                type_best[type_key], type_matrices[type_key][:, position]
            )
            coverage = torch.stack([values.mean() for values in type_best.values()]).mean()
            coverage_value = float(coverage.detach().cpu().item())
            if prefix >= min_keep and coverage_value >= coverage_threshold:
                keep = prefix
                # 单独标记直到全保留才达标的会话，避免把“达到阈值”误读成发生了压缩。
                reason = (
                    "coverage_full"
                    if prefix == int(ranked_indices.numel())
                    else "coverage"
                )
                break
            if prefix >= max_keep:
                keep = max_keep
                reason = "all_edges" if max_keep == int(ranked_indices.numel()) else "maximum"
                break
        return keep, coverage_value, reason

    def select_adaptive(
        self,
        scores: Tensor,
        hypergraph: HyperedgeBatch,
        stable_embeddings: Tensor,
        num_sessions: int,
        coverage_threshold: float,
        min_retention_ratio: float,
        max_retention_ratio: float,
        temperature: float,
        stochastic: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[int, int], Dict[int, float], Dict[int, str]]:
        if not 0.0 < coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must be in (0, 1].")
        if not 0.0 <= min_retention_ratio <= max_retention_ratio <= 1.0:
            raise ValueError("adaptive retention bounds must satisfy 0 <= min <= max <= 1.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if scores.numel() == 0:
            empty = scores.new_empty((0,))
            zeros = {session: 0 for session in range(num_sessions)}
            coverages = {session: 1.0 for session in range(num_sessions)}
            reasons = {session: "no_edges" for session in range(num_sessions)}
            return empty, empty, empty, zeros, coverages, reasons

        perturbed = scores + self._sample_gumbel_like(scores) if stochastic else scores
        hard_mask = torch.zeros_like(scores)
        soft_mask = torch.zeros_like(scores)
        kept_per_session: Dict[int, int] = {}
        coverage_per_session: Dict[int, float] = {}
        reasons: Dict[int, str] = {}

        for session in range(num_sessions):
            indices = torch.nonzero(hypergraph.edge_to_session == session, as_tuple=False).flatten()
            count = int(indices.numel())
            if count == 0:
                kept_per_session[session] = 0
                coverage_per_session[session] = 1.0
                reasons[session] = "no_edges"
                continue

            min_keep = max(1, min(count, int(math.ceil(min_retention_ratio * count))))
            max_keep = max(min_keep, min(count, int(math.ceil(max_retention_ratio * count))))
            ranked_local = torch.argsort(perturbed[indices], descending=True)
            ranked_indices = indices[ranked_local]
            keep, coverage, reason = self._adaptive_keep_count(
                ranked_indices,
                hypergraph,
                stable_embeddings,
                coverage_threshold,
                min_keep,
                max_keep,
            )
            selected = ranked_indices[:keep]
            hard_mask[selected] = 1.0
            soft_mask[indices] = self._budget_consistent_soft_mask(
                perturbed[indices], keep, temperature
            )
            kept_per_session[session] = keep
            coverage_per_session[session] = coverage
            reasons[session] = reason

        straight_through = soft_mask + (hard_mask - soft_mask).detach()
        return hard_mask, soft_mask, straight_through, kept_per_session, coverage_per_session, reasons

    def forward(
        self,
        projected_nodes: Tensor,
        batch_x: Tensor,
        graph_ids: Tensor,
        num_sessions: int,
        hypergraph: HyperedgeBatch,
        retention_ratio: float,
        gumbel_temperature: float,
        stochastic: bool,
        budget_mode: str = "fixed",
        coverage_threshold: float = 0.90,
        min_retention_ratio: float = 0.0,
        max_retention_ratio: float = 1.0,
        coverage_q: float = 0.20,
        coverage_tail_gamma: float = 0.70,
        coverage_eta: float = 0.25,
    ) -> SelectionOutput:
        scores, shared, session, edge_embeddings = self.score_edges(
            projected_nodes, graph_ids, num_sessions, hypergraph
        )
        stable_embeddings = self.encode_edges_raw(batch_x, hypergraph.incidence)
        if budget_mode == "fixed":
            hard, soft, straight_through, kept = self.select(
                scores,
                hypergraph.edge_to_session,
                num_sessions,
                retention_ratio,
                gumbel_temperature,
                stochastic,
            )
            coverage = self._coverage_statistics(
                hypergraph, stable_embeddings, hard, num_sessions
            )
            reasons = {session_id: "fixed" for session_id in range(num_sessions)}
            tail_coverage: Dict[int, float] = {}
            tail_by_type: Dict[int, Dict[int, float]] = {}
            proxy_scores = scores
        elif budget_mode == "adaptive":
            hard, soft, straight_through, kept, coverage, reasons = self.select_adaptive(
                scores,
                hypergraph,
                stable_embeddings,
                num_sessions,
                coverage_threshold,
                min_retention_ratio,
                max_retention_ratio,
                gumbel_temperature,
                stochastic,
            )
            tail_coverage = {}
            tail_by_type = {}
            proxy_scores = scores
        elif budget_mode == "robust_adaptive":
            (
                hard,
                soft,
                straight_through,
                kept,
                coverage,
                reasons,
                tail_by_type,
                proxy_scores,
            ) = self.select_robust_adaptive(
                scores,
                hypergraph,
                stable_embeddings,
                num_sessions,
                coverage_q,
                coverage_tail_gamma,
                coverage_eta,
                min_retention_ratio,
                max_retention_ratio,
                gumbel_temperature,
            )
            tail_coverage = dict(coverage)
        else:
            raise ValueError("budget_mode must be fixed, adaptive, or robust_adaptive.")

        retention: Dict[int, float] = {}
        for session_id in range(num_sessions):
            count = int((hypergraph.edge_to_session == session_id).sum().item())
            retention[session_id] = float(kept[session_id] / count) if count else 1.0
        return SelectionOutput(
            scores=scores,
            shared_scores=shared,
            session_scores=session,
            hard_mask=hard,
            soft_mask=soft,
            straight_through_mask=straight_through,
            kept_per_session=kept,
            retention_per_session=retention,
            coverage_per_session=coverage,
            tail_coverage_per_session=tail_coverage,
            tail_coverage_by_type_per_session=tail_by_type,
            stopping_reason_per_session=reasons,
            budget_mode=budget_mode,
            coverage_threshold=(
                coverage_tail_gamma
                if budget_mode == "robust_adaptive"
                else coverage_threshold
            ),
            coverage_q=coverage_q,
            coverage_eta=coverage_eta,
            proxy_scores=proxy_scores,
            edge_embeddings=edge_embeddings,
            stable_embeddings=stable_embeddings,
        )
