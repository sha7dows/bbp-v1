"""用途：加载 Vine checkpoint，预检稳定覆盖语义、自适应保留率和分类型剪枝行为。"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset
from bbp_behm.hypergraph import BehaviorType, HyperedgeBuilder
from bbp_behm.model import BBPBeHM
from bbp_behm.training import make_loader, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan adaptive Vine coverage without training.")
    parser.add_argument("--config", type=Path, default=Path("configs/vine_dedup_cv5_8020_seed42_pw1.5.ini"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_experiment_config(args.config)
    if config.pruning.budget_mode != "adaptive":
        raise ValueError("Coverage scan requires budget_mode=adaptive.")
    set_seed(args.seed)
    device = resolve_device(args.device or config.training.device)
    dataset = PreprocessedSessionDataset(
        config.data.processed_path,
        config.data.processed_sha256,
        config.data.expected_sessions,
        config.data.expected_nodes,
        config.data.expected_features,
    )
    loader = make_loader(dataset, range(len(dataset)), args.batch_size, False, args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model = BBPBeHM(config, torch.tensor([0.5, 0.5])).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    builder = HyperedgeBuilder(config.hypergraph)

    session_retentions = []
    session_coverages = []
    full_coverages = []
    stopping_reasons: Counter = Counter()
    candidate_by_type: Counter = Counter()
    kept_by_type: Counter = Counter()
    candidate_edges = 0
    kept_edges = 0

    with torch.no_grad():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            hypergraph = builder.build_batch(batch)
            projected_nodes = model.backbone.project_nodes(batch.x)
            selection = model.selector(
                projected_nodes=projected_nodes,
                batch_x=batch.x,
                graph_ids=batch.graph_ids,
                num_sessions=batch.num_graphs,
                hypergraph=hypergraph,
                retention_ratio=config.pruning.retention_ratio,
                gumbel_temperature=config.pruning.gumbel_temperature,
                stochastic=False,
                budget_mode="adaptive",
                coverage_threshold=config.pruning.coverage_threshold,
                min_retention_ratio=config.pruning.min_retention_ratio,
                max_retention_ratio=config.pruning.max_retention_ratio,
            )
            # 完整掩码必须在每个非空真实会话上达到精确覆盖1。
            full = model.selector._coverage_statistics(
                hypergraph,
                selection.stable_embeddings,
                hypergraph.incidence.new_ones((hypergraph.num_edges,)),
                batch.num_graphs,
            )
            full_coverages.extend(full.values())
            session_retentions.extend(selection.retention_per_session.values())
            session_coverages.extend(selection.coverage_per_session.values())
            stopping_reasons.update(selection.stopping_reason_per_session.values())
            candidate_edges += hypergraph.num_edges
            kept_edges += int(selection.hard_mask.sum().item())
            for behavior_type in BehaviorType:
                type_mask = hypergraph.type_ids == int(behavior_type)
                candidate_by_type[behavior_type.label] += int(type_mask.sum().item())
                kept_by_type[behavior_type.label] += int(
                    (selection.hard_mask[type_mask] > 0.5).sum().item()
                )

    full_failures = sum(abs(value - 1.0) > 1e-6 for value in full_coverages)
    report = {
        "_purpose": "Real Vine preflight for fixed-kernel adaptive coverage",
        "checkpoint": str(args.checkpoint),
        "sessions": len(session_retentions),
        "coverage_threshold": config.pruning.coverage_threshold,
        "coverage_kernel_weights": model.selector.kernel_weights.detach().cpu().tolist(),
        "candidate_edges": candidate_edges,
        "kept_edges": kept_edges,
        "edge_retention": kept_edges / max(candidate_edges, 1),
        "session_retention_mean": float(np.mean(session_retentions)),
        "session_retention_std": float(np.std(session_retentions)),
        "session_retention_min": float(np.min(session_retentions)),
        "session_retention_max": float(np.max(session_retentions)),
        "session_coverage_mean": float(np.mean(session_coverages)),
        "session_coverage_min": float(np.min(session_coverages)),
        "full_mask_coverage_min": float(np.min(full_coverages)),
        "full_mask_coverage_failures": full_failures,
        "stopping_reasons": dict(stopping_reasons),
        "retention_by_type": {
            behavior_type.label: {
                "candidate": candidate_by_type[behavior_type.label],
                "kept": kept_by_type[behavior_type.label],
                "retention": kept_by_type[behavior_type.label]
                / max(candidate_by_type[behavior_type.label], 1),
            }
            for behavior_type in BehaviorType
        },
    }
    if full_failures:
        raise RuntimeError(f"Full-mask coverage invariant failed for {full_failures} sessions.")
    logging.info("Coverage scan completed: %s", json.dumps(report, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info("Coverage scan written: %s", args.output)


if __name__ == "__main__":
    main()
