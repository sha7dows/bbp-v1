"""用途：把 BBP 整边选择器与 BeHM 主干组合为单一端到端模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .backbone import BackboneOutput, BeHMBackbone
from .config import ExperimentConfig
from .data import SessionBatch
from .hypergraph import HyperedgeBatch
from .selector import EdgeSelector, SelectionOutput


@dataclass
class ModelOutput:
    logits: Tensor
    session_representations: Tensor
    total_loss: Tensor
    classification_loss: Tensor
    selection: Optional[SelectionOutput]
    backbone: BackboneOutput
    masked_incidence: Tensor
    # 方向 A1' 附加:kept-set 互冗余正则项(仅 redundancy_lambda>0 训练期非 None)。
    redundancy_loss: Optional[Tensor] = None


class BBPBeHM(nn.Module):
    def __init__(
        self,
        config: ExperimentConfig,
        class_priors: Optional[Tensor] = None,
        confidence_class_means: Optional[Tensor] = None,
        confidence_global_mean: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = BeHMBackbone(
            input_dim=config.data.expected_features,
            hidden_dim=config.model.hidden_size,
            num_classes=config.model.num_classes,
            dropout=config.model.dropout,
            class_priors=class_priors,
            logit_adjustment=config.model.logit_adjustment,
            louvain_seed=config.model.louvain_seed,
            positive_class_weight=config.model.positive_class_weight,
            confidence_weighting=config.model.confidence_weighting,
            confidence_class_means=confidence_class_means,
            confidence_global_mean=confidence_global_mean,
            hgnn_residual=config.model.hgnn_residual,
            edge_pooling=config.model.edge_pooling,
            ibg_weight_mode=config.model.ibg_weight_mode,
            ibg_hard_tau=config.model.ibg_hard_tau,
        )
        # Selector 可独立降低随机扰动；旧的程序化配置仍回退到全局 dropout。
        self.selector_dropout = (
            config.model.dropout
            if config.model.selector_dropout is None
            else config.model.selector_dropout
        )
        self.selector = EdgeSelector(
            hidden_dim=config.model.hidden_size,
            type_embedding_dim=config.pruning.type_embedding_dim,
            dropout=self.selector_dropout,
            use_ibg_centrality=config.pruning.use_ibg_centrality,
        )

    def forward(
        self,
        batch: SessionBatch,
        hypergraph: HyperedgeBatch,
        retention_ratio: Optional[float] = None,
        pruning_enabled: Optional[bool] = None,
        stochastic_selection: Optional[bool] = None,
        budget_mode: Optional[str] = None,
        coverage_threshold: Optional[float] = None,
        min_retention_ratio: Optional[float] = None,
        max_retention_ratio: Optional[float] = None,
        coverage_q: Optional[float] = None,
        coverage_tail_gamma: Optional[float] = None,
        coverage_eta: Optional[float] = None,
        hard_mask_override: Optional[Tensor] = None,
    ) -> ModelOutput:
        ratio = self.config.pruning.retention_ratio if retention_ratio is None else retention_ratio
        enabled = self.config.pruning.enabled if pruning_enabled is None else pruning_enabled
        stochastic = self.training if stochastic_selection is None else stochastic_selection
        selected_budget_mode = self.config.pruning.budget_mode if budget_mode is None else budget_mode
        selected_coverage_threshold = (
            self.config.pruning.coverage_threshold
            if coverage_threshold is None
            else coverage_threshold
        )
        selected_min_ratio = (
            self.config.pruning.min_retention_ratio
            if min_retention_ratio is None
            else min_retention_ratio
        )
        selected_max_ratio = (
            self.config.pruning.max_retention_ratio
            if max_retention_ratio is None
            else max_retention_ratio
        )
        selected_coverage_q = (
            self.config.pruning.coverage_q if coverage_q is None else coverage_q
        )
        selected_tail_gamma = (
            self.config.pruning.coverage_tail_gamma
            if coverage_tail_gamma is None
            else coverage_tail_gamma
        )
        selected_coverage_eta = (
            self.config.pruning.coverage_eta if coverage_eta is None else coverage_eta
        )

        projected_nodes = self.backbone.project_nodes(batch.x)
        selection: Optional[SelectionOutput]
        if hard_mask_override is not None:
            if hard_mask_override.shape != (hypergraph.num_edges,):
                raise ValueError("hard_mask_override shape does not match the number of hyperedges.")
            hard_mask = hard_mask_override.to(dtype=hypergraph.incidence.dtype, device=hypergraph.incidence.device)
            straight_through_mask = hard_mask
            selection = None
        elif not enabled:
            hard_mask = hypergraph.incidence.new_ones((hypergraph.num_edges,))
            straight_through_mask = hard_mask
            selection = None
        else:
            selection = self.select_hyperedges(
                projected_nodes=projected_nodes,
                batch_x=batch.x,
                graph_ids=batch.graph_ids,
                num_sessions=batch.num_graphs,
                hypergraph=hypergraph,
                retention_ratio=ratio,
                gumbel_temperature=self.config.pruning.gumbel_temperature,
                stochastic=stochastic,
                budget_mode=selected_budget_mode,
                coverage_threshold=selected_coverage_threshold,
                min_retention_ratio=selected_min_ratio,
                max_retention_ratio=selected_max_ratio,
                coverage_q=selected_coverage_q,
                coverage_tail_gamma=selected_tail_gamma,
                coverage_eta=selected_coverage_eta,
            )
            hard_mask = selection.hard_mask
            straight_through_mask = selection.straight_through_mask

        masked_incidence = hypergraph.incidence * straight_through_mask[:, None]
        # 默认实现保持原结构；实验子类可在整边选择后受约束地精简传播关联。
        if selection is not None:
            masked_incidence = self.refine_incidence(projected_nodes, hypergraph, masked_incidence, hard_mask)
        # 关联权重扩展点默认恒等；独立于已有成员编辑和规模分解实验。
        masked_incidence = self.prepare_associations(projected_nodes, hypergraph, masked_incidence, hard_mask)
        # 整边选择后可受约束分解传播边；默认路径保持父边及原输出不变。
        propagation_graph, masked_incidence, propagation_mask = self.prepare_propagation(
            hypergraph, masked_incidence, hard_mask
        )
        backbone_output = self.backbone.forward_projected(
            projected_nodes=projected_nodes,
            incidence=masked_incidence,
            graph_ids=batch.graph_ids,
            edge_to_session=propagation_graph.edge_to_session,
            hard_edge_mask=propagation_mask,
            num_sessions=batch.num_graphs,
            labels=batch.labels,
            confidences=batch.confidences,
            topology_incidence=self.ibg_incidence(propagation_graph, masked_incidence),
        )
        # 预算由稳定覆盖约束决定，评分器仅通过直通掩码接收分类任务梯度。
        total_loss = backbone_output.classification_loss
        redundancy_loss: Optional[Tensor] = None
        if (
            self.training
            and selection is not None
            and self.config.pruning.redundancy_lambda > 0.0
        ):
            # 架构决策(09-03):A1' 正则只塑造 Selector 打分器,不回传共享编码器。
            # 用冻结的节点特征重算分数,仅打分器参数收到该项梯度;Backbone/编码器
            # 仍只由分类损失驱动(模块解耦,便于"选择端独立增益"归因)。
            frozen = projected_nodes.detach()
            red_scores, _, _, _ = self.selector.score_edges(
                frozen, batch.graph_ids, batch.num_graphs, hypergraph
            )
            temperature = self.config.pruning.gumbel_temperature
            red_soft = torch.zeros_like(red_scores)
            for session in range(batch.num_graphs):
                indices = torch.nonzero(
                    hypergraph.edge_to_session == session, as_tuple=False
                ).flatten()
                if indices.numel() < 2:
                    continue
                keep = int(selection.kept_per_session.get(session, indices.numel()))
                red_soft[indices] = EdgeSelector._budget_consistent_soft_mask(
                    red_scores[indices], keep, temperature
                )
            redundancy_loss = self._kept_set_redundancy(hypergraph, red_soft, batch.num_graphs)
            total_loss = total_loss + self.config.pruning.redundancy_lambda * redundancy_loss
        return ModelOutput(
            logits=backbone_output.logits,
            session_representations=backbone_output.session_representations,
            total_loss=total_loss,
            classification_loss=backbone_output.classification_loss,
            selection=selection,
            backbone=backbone_output,
            masked_incidence=masked_incidence,
            redundancy_loss=redundancy_loss,
        )

    def prepare_associations(self, projected_nodes, hypergraph, incidence, hard_mask):
        """默认不改变关联权重，保持旧模型和检查点行为。"""
        return incidence

    def ibg_incidence(self, hypergraph, propagation_incidence):
        """旧分支沿用自己的传播关联；加权实验显式覆盖二值拓扑来源。"""
        return propagation_incidence

    def prepare_propagation(self, hypergraph, incidence, hard_mask):
        """默认直接传播父边；分解实验在子类显式启用。"""
        return hypergraph, incidence, hard_mask

    def select_hyperedges(self, **kwargs) -> SelectionOutput:
        """默认直接使用原选择器，保留历史模型的前向与检查点行为。"""
        return self.selector(**kwargs)

    def refine_incidence(self, projected_nodes, hypergraph, incidence, hard_mask) -> Tensor:
        """默认不编辑关联；结构修复仅由显式启用的实验子类实现。"""
        return incidence

    def _kept_set_redundancy(
        self, hypergraph: HyperedgeBatch, soft_mask: Tensor, num_sessions: int
    ) -> Tensor:
        """方向 A1'(09-03 提案):kept-set 互冗余可微正则(选择端限定)。

        R_s = Σ_{e<f} m_e m_f · J(e,f) / (Σ_e m_e)^2,其中 m 为按冻结特征打分
        得到的预算一致 soft mask(sum≈keep,经 sigmoid-阈值保持可微),J 为纯结构 Jaccard。
        训练期按会话求和平均;回传路径只经打分器(features 已 detach),
        贪心/覆盖/预算/Backbone/编码器与评估期完全一致(不改变拓扑规则,只塑造打分)。
        不做蒸馏:该项是端到端目标,不使用任何外部/参考模型输出。
        """
        device = hypergraph.incidence.device
        dtype = torch.float32
        incidence = hypergraph.incidence.to(dtype)
        total = incidence.new_zeros((), dtype=dtype)
        counted = 0
        for session in range(num_sessions):
            indices = torch.nonzero(
                hypergraph.edge_to_session == session, as_tuple=False
            ).flatten()
            if indices.numel() < 2:
                continue
            local_inc = incidence[indices]
            mask = soft_mask[indices].to(dtype)
            intersection = local_inc @ local_inc.t()
            sizes = local_inc.sum(dim=1, keepdim=True)
            union = (sizes + sizes.t() - intersection).clamp_min(1e-8)
            jaccard = intersection / union
            jaccard = jaccard - torch.diag_embed(torch.diagonal(jaccard))
            denom = mask.sum().square().clamp_min(1e-8)
            # m^T J m 对每对 (e,f),(f,e) 计两次;R_s 定义只计 e<f 一次。
            session_term = (mask @ jaccard @ mask) * 0.5 / denom
            total = total + session_term
            counted += 1
        if counted == 0:
            return incidence.new_zeros((), dtype=dtype)
        return total / counted
