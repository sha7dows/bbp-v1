"""用途：在少量真实 Vine 会话上验证数据、构图、剪枝、反向传播与解释导出。"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import torch

from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset, collate_sessions
from bbp_behm.explain import build_explanation_rows, deletion_contributions, export_jsonl
from bbp_behm.hypergraph import HyperedgeBuilder
from bbp_behm.model import BBPBeHM
from bbp_behm.training import class_priors, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real Vine smoke test for BBP-BeHM.")
    parser.add_argument("--config", type=Path, default=Path("configs/vine_dedup_cv5_8020_seed42_pw1.5.ini"))
    parser.add_argument("--retention-ratio", type=float)
    parser.add_argument(
        "--budget-mode", choices=("fixed", "adaptive", "robust_adaptive")
    )
    parser.add_argument("--coverage-threshold", type=float)
    parser.add_argument("--coverage-q", type=float)
    parser.add_argument("--coverage-tail-gamma", type=float)
    parser.add_argument("--coverage-eta", type=float)
    parser.add_argument("--min-retention-ratio", type=float)
    parser.add_argument("--max-retention-ratio", type=float)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--contribution-edges", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke_vine_explanations.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_experiment_config(args.config)
    retention_ratio = (
        config.pruning.retention_ratio
        if args.retention_ratio is None
        else args.retention_ratio
    )
    budget_mode = config.pruning.budget_mode if args.budget_mode is None else args.budget_mode
    coverage_threshold = (
        config.pruning.coverage_threshold
        if args.coverage_threshold is None
        else args.coverage_threshold
    )
    min_retention_ratio = (
        config.pruning.min_retention_ratio
        if args.min_retention_ratio is None
        else args.min_retention_ratio
    )
    max_retention_ratio = (
        config.pruning.max_retention_ratio
        if args.max_retention_ratio is None
        else args.max_retention_ratio
    )
    coverage_q = (
        config.pruning.coverage_q if args.coverage_q is None else args.coverage_q
    )
    coverage_tail_gamma = (
        config.pruning.coverage_tail_gamma
        if args.coverage_tail_gamma is None
        else args.coverage_tail_gamma
    )
    coverage_eta = (
        config.pruning.coverage_eta if args.coverage_eta is None else args.coverage_eta
    )
    seed = config.split.seeds[0]
    set_seed(seed)
    dataset = PreprocessedSessionDataset(
        config.data.processed_path,
        config.data.processed_sha256,
        config.data.expected_sessions,
        config.data.expected_nodes,
        config.data.expected_features,
    )
    lengths = [int(dataset.slices["x"][index + 1] - dataset.slices["x"][index]) for index in range(len(dataset))]
    selected_ids = sorted(range(len(dataset)), key=lambda index: lengths[index])[: args.sessions]
    batch = collate_sessions([dataset[index] for index in selected_ids])
    hypergraph = HyperedgeBuilder(config.hypergraph).build_batch(batch)
    model = BBPBeHM(config, class_priors(dataset, range(len(dataset)))).train()
    output = model(
        batch,
        hypergraph,
        retention_ratio=retention_ratio,
        pruning_enabled=True,
        stochastic_selection=False,
        budget_mode=budget_mode,
        coverage_threshold=coverage_threshold,
        min_retention_ratio=min_retention_ratio,
        max_retention_ratio=max_retention_ratio,
        coverage_q=coverage_q,
        coverage_tail_gamma=coverage_tail_gamma,
        coverage_eta=coverage_eta,
    )
    output.total_loss.backward()
    selector_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.selector.parameters()
        if parameter.grad is not None
    )
    logging.info(
        "Smoke forward/backward complete: sessions=%d nodes=%d candidate_edges=%d kept_edges=%d retention=%.6f coverage=%.6f selector_grad_l1=%.6f",
        batch.num_graphs,
        batch.x.size(0),
        hypergraph.num_edges,
        int(output.selection.hard_mask.sum().item()),
        sum(output.selection.retention_per_session.values()) / batch.num_graphs,
        sum(output.selection.coverage_per_session.values()) / batch.num_graphs,
        selector_gradient,
    )

    model.eval()
    with torch.no_grad():
        deterministic = model(
            batch,
            hypergraph,
            retention_ratio=retention_ratio,
            pruning_enabled=True,
            stochastic_selection=False,
            budget_mode=budget_mode,
            coverage_threshold=coverage_threshold,
            min_retention_ratio=min_retention_ratio,
            max_retention_ratio=max_retention_ratio,
            coverage_q=coverage_q,
            coverage_tail_gamma=coverage_tail_gamma,
            coverage_eta=coverage_eta,
        )
    kept_ids = torch.nonzero(deterministic.selection.hard_mask > 0.5, as_tuple=False).flatten().tolist()
    contributions = deletion_contributions(
        model, batch, hypergraph, deterministic, edge_ids=kept_ids[: args.contribution_edges]
    )
    rows = build_explanation_rows(batch, hypergraph, deterministic, model, contributions)
    export_jsonl(
        args.output,
        rows,
        metadata={
            "dataset": "vine",
            "session_indices": selected_ids,
            "retention_ratio": retention_ratio,
            "budget_mode": budget_mode,
            "coverage_threshold": coverage_threshold,
            "min_retention_ratio": min_retention_ratio,
            "max_retention_ratio": max_retention_ratio,
            "coverage_q": coverage_q,
            "coverage_tail_gamma": coverage_tail_gamma,
            "coverage_eta": coverage_eta,
            "stopping_reasons": dict(Counter(deterministic.selection.stopping_reason_per_session.values())),
            "confidence_source": "dataset" if dataset.has_confidence else "fallback_1.0",
        },
    )
    logging.info("Explanation export complete: path=%s rows=%d", args.output, len(rows))


if __name__ == "__main__":
    main()
