"""用途：导出行为基、同类型代表映射，并用模型内删除干预计算预测贡献。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .data import SessionBatch
from .hypergraph import HyperedgeBatch
from .model import BBPBeHM, ModelOutput


def representative_assignments(
    model: BBPBeHM,
    hypergraph: HyperedgeBatch,
    output: ModelOutput,
) -> Dict[int, Tuple[int, float]]:
    if output.selection is None or hypergraph.num_edges == 0:
        return {}
    assignments: Dict[int, Tuple[int, float]] = {}
    hard_mask = output.selection.hard_mask
    for session in torch.unique(hypergraph.edge_to_session).tolist():
        session_indices = torch.nonzero(hypergraph.edge_to_session == int(session), as_tuple=False).flatten()
        for type_id in torch.unique(hypergraph.type_ids[session_indices]).tolist():
            local = session_indices[hypergraph.type_ids[session_indices] == int(type_id)]
            kept = local[hard_mask[local] > 0.5]
            deleted = local[hard_mask[local] <= 0.5]
            if kept.numel() == 0 or deleted.numel() == 0:
                continue
            combined = torch.cat([deleted, kept])
            representative = model.selector._representativeness_matrix(
                hypergraph.incidence[combined],
                output.selection.stable_embeddings[combined],
                hypergraph.structural_stats[combined],
            )
            deleted_count = int(deleted.numel())
            kept_similarity = representative[:deleted_count, deleted_count:]
            best_similarity, best_position = torch.max(kept_similarity, dim=1)
            for offset, deleted_id in enumerate(deleted.tolist()):
                representative_id = int(kept[int(best_position[offset].item())].item())
                assignments[int(deleted_id)] = (
                    representative_id,
                    float(best_similarity[offset].detach().cpu().item()),
                )
    return assignments


@torch.no_grad()
def deletion_contributions(
    model: BBPBeHM,
    batch: SessionBatch,
    hypergraph: HyperedgeBatch,
    base_output: ModelOutput,
    edge_ids: Optional[Iterable[int]] = None,
    target_classes: Optional[torch.Tensor] = None,
) -> Dict[int, float]:
    if base_output.selection is None:
        return {}
    model.eval()
    base_mask = base_output.selection.hard_mask.detach()
    if target_classes is None:
        target_classes = torch.argmax(base_output.logits, dim=1)
    base_probabilities = F.softmax(base_output.logits, dim=1)
    candidates = (
        [int(edge_id) for edge_id in edge_ids]
        if edge_ids is not None
        else torch.nonzero(base_mask > 0.5, as_tuple=False).flatten().tolist()
    )
    contributions: Dict[int, float] = {}
    for edge_id in candidates:
        if edge_id < 0 or edge_id >= hypergraph.num_edges or base_mask[edge_id] <= 0.5:
            continue
        record = hypergraph.records[edge_id]
        intervention_mask = base_mask.clone()
        intervention_mask[edge_id] = 0.0
        intervened = model(
            batch,
            hypergraph,
            pruning_enabled=False,
            stochastic_selection=False,
            hard_mask_override=intervention_mask,
        )
        session = record.session_position
        target = int(target_classes[session].item())
        intervened_probability = F.softmax(intervened.logits, dim=1)[session, target]
        contributions[edge_id] = float(
            (base_probabilities[session, target] - intervened_probability).detach().cpu().item()
        )
    return contributions


def build_explanation_rows(
    batch: SessionBatch,
    hypergraph: HyperedgeBatch,
    output: ModelOutput,
    model: BBPBeHM,
    contributions: Optional[Dict[int, float]] = None,
) -> List[dict]:
    contributions = contributions or {}
    assignments = representative_assignments(model, hypergraph, output)
    represented_counts = Counter(representative_id for representative_id, _ in assignments.values())
    probabilities = F.softmax(output.logits.detach(), dim=1).cpu()
    rows: List[dict] = []
    for record in hypergraph.records:
        edge_id = record.hyperedge_id
        session_position = record.session_position
        if output.selection is None:
            kept = True
            score = shared = session_score = None
            budget_mode = "disabled"
            session_candidate_edges = int(
                (hypergraph.edge_to_session == session_position).sum().item()
            )
            session_kept_edges = session_candidate_edges
            session_retention = 1.0
            session_coverage = 1.0
            coverage_threshold = None
            stopping_reason = "disabled"
        else:
            kept = bool(output.selection.hard_mask[edge_id].item() > 0.5)
            score = float(output.selection.scores[edge_id].detach().cpu().item())
            shared = float(output.selection.shared_scores[edge_id].detach().cpu().item())
            session_score = float(output.selection.session_scores[edge_id].detach().cpu().item())
            budget_mode = output.selection.budget_mode
            session_candidate_edges = int(
                (hypergraph.edge_to_session == session_position).sum().item()
            )
            session_kept_edges = output.selection.kept_per_session[session_position]
            session_retention = output.selection.retention_per_session[session_position]
            session_coverage = output.selection.coverage_per_session[session_position]
            coverage_threshold = output.selection.coverage_threshold
            stopping_reason = output.selection.stopping_reason_per_session[session_position]
        representative_id, representative_similarity = assignments.get(edge_id, (None, None))
        row = asdict(record)
        row["behavior_type"] = record.behavior_type.label
        row["members"] = list(record.members)
        row["source_ids"] = list(record.source_ids)
        row.update(
            {
                "kept": kept,
                "selection_score": score,
                "shared_score": shared,
                "session_score": session_score,
                "budget_mode": budget_mode,
                "session_candidate_edges": session_candidate_edges,
                "session_kept_edges": session_kept_edges,
                "session_retention_ratio": session_retention,
                "session_representative_coverage": session_coverage,
                "coverage_threshold": coverage_threshold,
                "stopping_reason": stopping_reason,
                "representative_hyperedge_id": representative_id,
                "representative_similarity": representative_similarity,
                "represented_deleted_edges": int(represented_counts.get(edge_id, 0)),
                "deletion_contribution": contributions.get(edge_id),
                "true_label": int(batch.labels[record.session_position].detach().cpu().item()),
                "predicted_label": int(torch.argmax(probabilities[record.session_position]).item()),
                "positive_probability": float(probabilities[record.session_position, 1].item()),
            }
        )
        rows.append(row)
    return rows


def export_jsonl(path: Path, rows: Iterable[dict], metadata: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "_purpose": "BBP-BeHM behavior-basis explanation export",
                    "_metadata": metadata or {},
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
