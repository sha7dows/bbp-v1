"""用途：提供 Vine-first 的可复现训练、评估、分类指标与结构效率统计。"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from .data import IndexedSessionDataset, PreprocessedSessionDataset, SessionBatch, collate_sessions
from .hypergraph import BehaviorType, HyperedgeBuilder
from .model import BBPBeHM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def class_priors(dataset: PreprocessedSessionDataset, indices: Iterable[int], num_classes: int = 2) -> torch.Tensor:
    selected = torch.tensor(tuple(int(index) for index in indices), dtype=torch.long)
    counts = torch.bincount(dataset.labels[selected], minlength=num_classes).float()
    return counts / counts.sum().clamp_min(1.0)


def confidence_statistics(
    dataset: PreprocessedSessionDataset,
    indices: Iterable[int],
    num_classes: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """仅从训练索引计算类别均值与全局均值，供 confidence 类别内归一化使用。"""
    selected = torch.tensor(tuple(int(index) for index in indices), dtype=torch.long)
    if selected.numel() == 0:
        raise ValueError("Confidence statistics require non-empty training indices.")
    labels = dataset.labels[selected]
    confidences = dataset.session_confidences[selected]
    class_means = []
    for class_id in range(num_classes):
        values = confidences[labels == class_id]
        if values.numel() == 0:
            raise ValueError(f"Training split has no samples for class {class_id}.")
        class_means.append(values.mean())
    return torch.stack(class_means), confidences.mean()


def make_loader(
    dataset: PreprocessedSessionDataset,
    indices: Iterable[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        IndexedSessionDataset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate_sessions,
    )


@dataclass
class EpochSummary:
    loss: float
    classification_loss: float
    session_retention_mean: float
    session_coverage_mean: float
    sessions: int


def with_training_feature_noise(batch: SessionBatch, feature_noise_std: float) -> SessionBatch:
    """仅为当前训练批次创建特征扰动副本，不修改数据集快照或原批次张量。"""
    if feature_noise_std < 0.0:
        raise ValueError("feature_noise_std must be non-negative.")
    if feature_noise_std == 0.0:
        return batch
    noisy_x = batch.x + torch.randn_like(batch.x) * feature_noise_std
    return replace(batch, x=noisy_x)


def train_epoch(
    model: BBPBeHM,
    loader: DataLoader,
    builder: HyperedgeBuilder,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    retention_ratio: float,
    pruning_enabled: bool,
    budget_mode: Optional[str] = None,
    coverage_threshold: Optional[float] = None,
    min_retention_ratio: Optional[float] = None,
    max_retention_ratio: Optional[float] = None,
    coverage_q: Optional[float] = None,
    coverage_tail_gamma: Optional[float] = None,
    coverage_eta: Optional[float] = None,
    feature_noise_std: float = 0.0,
) -> EpochSummary:
    model.train()
    total_loss = 0.0
    total_classification = 0.0
    total_retention = 0.0
    total_coverage = 0.0
    total_selection_sessions = 0
    total_sessions = 0
    for cpu_batch in loader:
        batch: SessionBatch = cpu_batch.to(device)
        # 超图拓扑始终由干净特征构建；扰动仅作用于本次训练前向的临时节点特征。
        hypergraph = builder.build_batch(batch)
        model_batch = with_training_feature_noise(batch, feature_noise_std)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            model_batch,
            hypergraph,
            retention_ratio=retention_ratio,
            pruning_enabled=pruning_enabled,
            stochastic_selection=pruning_enabled,
            budget_mode=budget_mode,
            coverage_threshold=coverage_threshold,
            min_retention_ratio=min_retention_ratio,
            max_retention_ratio=max_retention_ratio,
            coverage_q=coverage_q,
            coverage_tail_gamma=coverage_tail_gamma,
            coverage_eta=coverage_eta,
        )
        output.total_loss.backward()
        optimizer.step()
        count = batch.num_graphs
        total_sessions += count
        total_loss += float(output.total_loss.detach().cpu()) * count
        total_classification += float(output.classification_loss.detach().cpu()) * count
        if output.selection is None:
            total_retention += float(count)
            total_coverage += float(count)
            total_selection_sessions += count
        else:
            total_retention += sum(output.selection.retention_per_session.values())
            total_coverage += sum(output.selection.coverage_per_session.values())
            total_selection_sessions += len(output.selection.retention_per_session)
    denominator = max(total_sessions, 1)
    return EpochSummary(
        loss=total_loss / denominator,
        classification_loss=total_classification / denominator,
        session_retention_mean=total_retention / max(total_selection_sessions, 1),
        session_coverage_mean=total_coverage / max(total_selection_sessions, 1),
        sessions=total_sessions,
    )


@torch.no_grad()
def _kept_structure_session_stats(
    hypergraph: "HyperedgeBatch", hard_mask: torch.Tensor, num_sessions: int
) -> tuple[list[float], int, list[float], int]:
    """P1'(09-03 提案,纯仪器化,零机制改动):按会话统计 kept 边集的结构冗余。

    返回 (dup_shares, dup_sessions, densities, density_sessions):
    - dup_shares:每会话 kept STC 边中,存在另一条 kept STC 边且 Jaccard>=0.7 的边占比;
    - densities:每会话 kept 边(全类型)两两 Jaccard>0 对儿的平均边权(IBG 密度代理)。
    仅在 kept 边数 >=2 且满足各自口径的会话计入;计算不影响任何选择/训练决策。
    """
    incidence = hypergraph.incidence.float()
    hard = hard_mask > 0.5
    dup_shares: list[float] = []
    dup_sessions = 0
    densities: list[float] = []
    density_sessions = 0
    for session in range(num_sessions):
        indices = torch.nonzero(
            hypergraph.edge_to_session == session, as_tuple=False
        ).flatten()
        if indices.numel() < 2:
            continue
        kept = indices[hard[indices]]
        if kept.numel() < 2:
            continue
        kept_inc = incidence[kept]
        intersection = kept_inc @ kept_inc.t()
        sizes = kept_inc.sum(dim=1, keepdim=True)
        union = (sizes + sizes.t() - intersection).clamp_min(1e-8)
        jaccard = intersection / union
        jaccard = jaccard - torch.diag_embed(torch.diagonal(jaccard))
        triu = torch.triu(jaccard, diagonal=1)
        positive = triu[triu > 0.0]
        if positive.numel():
            densities.append(float(positive.mean().item()))
            density_sessions += 1
        # STC 内部近重复占比(同类型口径,与 E0 盘存一致)。
        stc_flags = hypergraph.type_ids[kept] == int(BehaviorType.STC)
        if int(stc_flags.sum().item()) >= 2:
            stc_rows = torch.nonzero(stc_flags, as_tuple=False).flatten()
            stc_jac = jaccard[stc_rows][:, stc_rows]
            max_jac = stc_jac.max(dim=1).values
            dup_shares.append(float((max_jac >= 0.7).float().mean().item()))
            dup_sessions += 1
    return dup_shares, dup_sessions, densities, density_sessions


@torch.no_grad()
def evaluate(
    model: BBPBeHM,
    loader: DataLoader,
    builder: HyperedgeBuilder,
    device: torch.device,
    retention_ratio: float,
    pruning_enabled: bool,
    budget_mode: Optional[str] = None,
    coverage_threshold: Optional[float] = None,
    min_retention_ratio: Optional[float] = None,
    max_retention_ratio: Optional[float] = None,
    coverage_q: Optional[float] = None,
    coverage_tail_gamma: Optional[float] = None,
    coverage_eta: Optional[float] = None,
) -> Dict[str, float]:
    model.eval()
    effective_budget_mode = (
        model.config.pruning.budget_mode if budget_mode is None else budget_mode
    )
    labels = []
    predictions = []
    positive_probabilities = []
    candidate_edges = 0
    kept_edges = 0
    candidate_incidence = 0.0
    kept_incidence = 0.0
    ibg_nodes = 0
    ibg_edges = 0
    session_retentions = []
    session_coverages = []
    coverage_satisfied = 0
    stopping_reasons: Counter = Counter()
    candidate_edges_by_type: Counter = Counter()
    kept_edges_by_type: Counter = Counter()
    tail_coverages = []
    tail_coverages_by_type: Dict[int, list] = {
        int(behavior_type): [] for behavior_type in BehaviorType
    }
    kept_stc_dup_shares: list[float] = []
    kept_stc_dup_sessions = 0
    kept_ibg_densities: list[float] = []
    kept_ibg_density_sessions = 0

    for cpu_batch in loader:
        batch: SessionBatch = cpu_batch.to(device)
        hypergraph = builder.build_batch(batch)
        output = model(
            batch,
            hypergraph,
            retention_ratio=retention_ratio,
            pruning_enabled=pruning_enabled,
            stochastic_selection=False,
            budget_mode=budget_mode,
            coverage_threshold=coverage_threshold,
            min_retention_ratio=min_retention_ratio,
            max_retention_ratio=max_retention_ratio,
            coverage_q=coverage_q,
            coverage_tail_gamma=coverage_tail_gamma,
            coverage_eta=coverage_eta,
        )
        probabilities = torch.softmax(output.logits, dim=1)
        labels.extend(batch.labels.detach().cpu().tolist())
        predictions.extend(torch.argmax(probabilities, dim=1).detach().cpu().tolist())
        positive_probabilities.extend(probabilities[:, 1].detach().cpu().tolist())
        candidate_edges += hypergraph.num_edges
        candidate_incidence += float(hypergraph.incidence.sum().detach().cpu())
        for behavior_type in BehaviorType:
            type_mask = hypergraph.type_ids == int(behavior_type)
            candidate_edges_by_type[behavior_type.label] += int(type_mask.sum().item())
        if output.selection is None:
            hard_mask = hypergraph.incidence.new_ones((hypergraph.num_edges,))
            batch_retentions = [1.0] * batch.num_graphs
            batch_coverages = [1.0] * batch.num_graphs
            stopping_reasons.update(["disabled"] * batch.num_graphs)
        else:
            hard_mask = output.selection.hard_mask
            batch_retentions = list(output.selection.retention_per_session.values())
            batch_coverages = list(output.selection.coverage_per_session.values())
            stopping_reasons.update(output.selection.stopping_reason_per_session.values())
            tail_coverages.extend(output.selection.tail_coverage_per_session.values())
            for session_values in output.selection.tail_coverage_by_type_per_session.values():
                for type_id, value in session_values.items():
                    tail_coverages_by_type[int(type_id)].append(float(value))
        for behavior_type in BehaviorType:
            type_mask = hypergraph.type_ids == int(behavior_type)
            kept_edges_by_type[behavior_type.label] += int(
                (hard_mask[type_mask] > 0.5).sum().item()
            )
        session_retentions.extend(batch_retentions)
        session_coverages.extend(batch_coverages)
        threshold = (
            (
                model.config.pruning.coverage_tail_gamma
                if coverage_tail_gamma is None
                else coverage_tail_gamma
            )
            if effective_budget_mode == "robust_adaptive"
            else (
                model.config.pruning.coverage_threshold
                if coverage_threshold is None
                else coverage_threshold
            )
        )
        coverage_satisfied += sum(value + 1e-8 >= threshold for value in batch_coverages)
        kept_edges += int(hard_mask.sum().detach().cpu().item())
        kept_incidence += float((hypergraph.incidence * hard_mask[:, None]).sum().detach().cpu())
        ibg_nodes += sum(output.backbone.ibg_node_counts)
        ibg_edges += sum(output.backbone.ibg_edge_counts)
        dup_shares, dup_sessions, densities, density_sessions = (
            _kept_structure_session_stats(hypergraph, hard_mask, batch.num_graphs)
        )
        kept_stc_dup_shares.extend(dup_shares)
        kept_stc_dup_sessions += dup_sessions
        kept_ibg_densities.extend(densities)
        kept_ibg_density_sessions += density_sessions

    if not labels:
        raise ValueError("Evaluation loader is empty.")
    metrics = {
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "positive_f1": f1_score(labels, predictions, pos_label=1, zero_division=0),
        "precision": precision_score(labels, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(labels, predictions, pos_label=1, zero_division=0),
        "auroc": roc_auc_score(labels, positive_probabilities) if len(set(labels)) > 1 else float("nan"),
        "candidate_edges": float(candidate_edges),
        "kept_edges": float(kept_edges),
        "edge_retention": float(kept_edges / max(candidate_edges, 1)),
        "candidate_incidence": float(candidate_incidence),
        "kept_incidence": float(kept_incidence),
        "incidence_retention": float(kept_incidence / max(candidate_incidence, 1.0)),
        "ibg_nodes": float(ibg_nodes),
        "ibg_directed_edges": float(ibg_edges),
        "session_retention_mean": float(np.mean(session_retentions)),
        "session_retention_std": float(np.std(session_retentions)),
        "session_retention_min": float(np.min(session_retentions)),
        "session_retention_max": float(np.max(session_retentions)),
        "session_coverage_mean": float(np.mean(session_coverages)),
        "session_coverage_min": float(np.min(session_coverages)),
        "coverage_satisfied_fraction": float(
            coverage_satisfied / max(len(session_coverages), 1)
        ),
    }
    for reason in (
        "coverage",
        "coverage_full",
        "maximum",
        "all_edges",
        "no_edges",
        "fixed",
        "disabled",
    ):
        metrics[f"stopping_{reason}_fraction"] = float(
            stopping_reasons[reason] / max(len(session_coverages), 1)
        )
    for behavior_type in BehaviorType:
        label = behavior_type.label.lower()
        candidates = candidate_edges_by_type[behavior_type.label]
        retained = kept_edges_by_type[behavior_type.label]
        metrics[f"{label}_candidate_edges"] = float(candidates)
        metrics[f"{label}_kept_edges"] = float(retained)
        metrics[f"{label}_edge_retention"] = float(retained / max(candidates, 1))
        values = tail_coverages_by_type[int(behavior_type)]
        if values:
            metrics[f"{label}_tail_coverage_mean"] = float(np.mean(values))
            metrics[f"{label}_tail_coverage_min"] = float(np.min(values))
    if tail_coverages:
        metrics["tail_coverage_mean"] = float(np.mean(tail_coverages))
        metrics["tail_coverage_min"] = float(np.min(tail_coverages))
    # P1'(09-03 提案)新增结构观测(纯仪器,不改任何机制)。
    metrics["kept_stc_dup_frac_0_7"] = (
        float(np.mean(kept_stc_dup_shares)) if kept_stc_dup_shares else 0.0
    )
    metrics["kept_stc_dup_frac_0_7_sessions"] = float(kept_stc_dup_sessions)
    metrics["kept_ibg_density"] = (
        float(np.mean(kept_ibg_densities)) if kept_ibg_densities else 0.0
    )
    metrics["kept_ibg_density_sessions"] = float(kept_ibg_density_sessions)
    return {key: float(value) for key, value in metrics.items()}
