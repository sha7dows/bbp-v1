"""只读探索：在少量验证集会话上跑通 explain 全流程，实测 deletion_contributions 耗时与输出完整性。

范围：只取验证集前 N 个会话（默认 12），不读取测试集，不训练。
输出：outputs/explore_explain/explore_val12.jsonl + 控制台统计。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset, collate_sessions
from bbp_behm.explain import (
    build_explanation_rows,
    deletion_contributions,
    export_jsonl,
    representative_assignments,
)
from bbp_behm.hypergraph import HyperedgeBuilder
from bbp_behm.model import BBPBeHM
from bbp_behm.training import make_loader, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="探索 explain 导出流程与耗时。")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--processed-path", type=Path, required=True)
    parser.add_argument("--n-sessions", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/explore_explain/explore_val12.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    from dataclasses import replace

    config = replace(config, data=replace(config.data, processed_path=args.processed_path))
    set_seed(42)
    device = resolve_device(args.device)

    split = json.loads(args.split.read_text(encoding="utf-8"))
    indices = list(split["validation"])[: args.n_sessions]
    print(f"validation sessions selected: {len(indices)} (of {len(split['validation'])})")

    dataset = PreprocessedSessionDataset(
        config.data.processed_path,
        config.data.processed_sha256,
        config.data.expected_sessions,
        config.data.expected_nodes,
        config.data.expected_features,
    )
    loader = make_loader(dataset, indices, len(indices), False, 42)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model = BBPBeHM(config, torch.tensor([0.5, 0.5])).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    builder = HyperedgeBuilder(config.hypergraph)

    total_candidates = 0
    total_kept = 0
    contributions_all: list[float] = []
    contribution_count = 0
    assign_count = 0
    rows_total = 0

    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        hypergraph = builder.build_batch(batch)
        total_candidates += hypergraph.num_edges
        with torch.no_grad():
            base_output = model(
                batch,
                hypergraph,
                pruning_enabled=True,
                stochastic_selection=False,
                budget_mode="robust_adaptive",
                coverage_q=config.pruning.coverage_q,
                coverage_tail_gamma=config.pruning.coverage_tail_gamma,
                coverage_eta=config.pruning.coverage_eta,
            )
        total_kept += int(base_output.selection.hard_mask.sum().item())
        print(
            f"batch: sessions={batch.num_graphs} nodes={batch.x.size(0)} "
            f"candidate_edges={hypergraph.num_edges} kept_edges={int(base_output.selection.hard_mask.sum().item())} "
            f"ibg_edges={sum(base_output.backbone.ibg_edge_counts)}"
        )

        # 代表映射（只读，一次计算）。
        t0 = time.perf_counter()
        assignments = representative_assignments(model, hypergraph, base_output)
        assign_count += len(assignments)
        t_assign = time.perf_counter() - t0

        # 删除贡献：先测前 8 条保留边的耗时，再全量。
        kept_ids = torch.nonzero(base_output.selection.hard_mask > 0.5, as_tuple=False).flatten().tolist()
        probe_ids = kept_ids[:8]
        t0 = time.perf_counter()
        probe_contributions = deletion_contributions(
            model, batch, hypergraph, base_output, edge_ids=probe_ids
        )
        t_probe = time.perf_counter() - t0
        per_edge_seconds = t_probe / max(len(probe_contributions), 1)
        print(
            f"  deletion probe: {len(probe_contributions)} edges in {t_probe:.2f}s "
            f"({per_edge_seconds:.3f}s/edge) -> full batch kept {len(kept_ids)} edges would take "
            f"{per_edge_seconds * len(kept_ids):.1f}s"
        )

        t0 = time.perf_counter()
        contributions = deletion_contributions(
            model, batch, hypergraph, base_output, edge_ids=kept_ids
        )
        t_full = time.perf_counter() - t0
        contributions_all.extend(contributions.values())
        contribution_count += len(contributions)
        print(f"  deletion full: {len(contributions)} edges in {t_full:.2f}s")

        rows = build_explanation_rows(batch, hypergraph, base_output, model, contributions)
        rows_total += len(rows)
        export_jsonl(
            args.output,
            rows,
            metadata={
                "scope": "explore",
                "validation_indices": indices,
                "dedup_mode": config.hypergraph.dedup_mode,
                "device": str(device),
                "timing": {
                    "representative_assignments_seconds": t_assign,
                    "deletion_probe_edges": len(probe_contributions),
                    "deletion_probe_seconds": t_probe,
                    "deletion_full_edges": len(contributions),
                    "deletion_full_seconds": t_full,
                    "seconds_per_edge": per_edge_seconds,
                },
            },
        )

    import numpy as np

    print("\n=== SUMMARY ===")
    print(f"candidate_edges_total={total_candidates} kept_edges_total={total_kept}")
    print(f"rows_exported={rows_total} assignments={assign_count} contributions={contribution_count}")
    if contributions_all:
        arr = np.asarray(contributions_all)
        print(
            f"deletion_contribution: min={arr.min():.5f} p25={np.quantile(arr,0.25):.5f} "
            f"median={np.median(arr):.5f} mean={arr.mean():.5f} p75={np.quantile(arr,0.75):.5f} "
            f"max={arr.max():.5f} zero_frac={(np.abs(arr)<1e-9).mean():.4f}"
        )
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
