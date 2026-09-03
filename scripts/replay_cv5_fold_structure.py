"""用途：P1'(09-03 提案)—— 用归档 CV5-80/20 的 fold checkpoint 在各自折验证上重放 evaluate,
产出带结构观测指标(kept_stc_dup_frac_0_7 / kept_ibg_density 等)的"结构基线"。

- 零机制改动、不训练、不触碰固定测试集(只用 folds_detail 的 validation 索引)。
- 模型/先验/loader 口径与 scripts/train_vine_cv5.py 完全一致:priors 与置信度统计只用该折训练会话。
- 用法:
  python -m scripts.replay_cv5_fold_structure \
    --config configs/vine_dedup_cv5_8020_seed42_pw1.5.ini \
    --run-dir outputs/cv5/vine_dedup_cv5_8020_seed42_pw1.5 \
    --split-json outputs/cv5/vine_dedup_cv5_8020_seed42_pw1.5/split.json \
    --output outputs/diagnostics/p1_baseline_structure.json [--device cuda:0] [--fold 1]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset
from bbp_behm.hypergraph import HyperedgeBuilder
from bbp_behm.model import BBPBeHM
from bbp_behm.training import (
    class_priors,
    confidence_statistics,
    evaluate,
    make_loader,
    resolve_device,
    set_seed,
)

LOCAL_PROCESSED = Path(
    r"E:\Paper\Code\BeHM\data\vine_with_5\KbertConfidence\processed\KbertConfidence_data_bert.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1' 结构基线重放(CV5 fold checkpoints, 只推理)。")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="含 fold_checkpoints/ 的 CV5 run 目录。")
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--processed-path", type=Path, default=LOCAL_PROCESSED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fold", type=int, default=0, help="只重放单个折(调试);0=全部。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    config = replace(config, data=replace(config.data, processed_path=args.processed_path))
    device = resolve_device(args.device)
    dataset = PreprocessedSessionDataset(
        config.data.processed_path,
        config.data.processed_sha256,
        config.data.expected_sessions,
        config.data.expected_nodes,
        config.data.expected_features,
    )
    split_data = json.loads(args.split_json.read_text(encoding="utf-8"))
    ckpt_dir = args.run_dir / "fold_checkpoints"
    # global_epoch 从任一 fold checkpoint 读取,避免硬编码。
    first_ckpt = sorted(ckpt_dir.glob("fold_*_epoch_*.pt"))[0]
    global_epoch = int(torch.load(first_ckpt, map_location="cpu", weights_only=False)["global_epoch"])

    builder = HyperedgeBuilder(config.hypergraph)
    per_fold: list[dict[str, Any]] = []
    for fold in split_data["folds_detail"]:
        fold_no = int(fold["fold"])
        if args.fold and fold_no != args.fold:
            continue
        train_indices = tuple(int(i) for i in fold["train"])
        validation_indices = tuple(int(i) for i in fold["validation"])
        ckpt_path = ckpt_dir / f"fold_{fold_no:02d}_epoch_{global_epoch:03d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        set_seed(42)
        priors = class_priors(dataset, train_indices, config.model.num_classes)
        conf_means, conf_global = confidence_statistics(
            dataset, train_indices, config.model.num_classes
        )
        model = BBPBeHM(
            config,
            priors,
            confidence_class_means=conf_means,
            confidence_global_mean=conf_global,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        loader = make_loader(
            dataset, validation_indices, config.training.batch_size, False, 42
        )
        metrics = evaluate(
            model,
            loader,
            builder,
            device,
            retention_ratio=config.pruning.retention_ratio,
            pruning_enabled=config.pruning.enabled,
            budget_mode=config.pruning.budget_mode,
            coverage_threshold=config.pruning.coverage_threshold,
            min_retention_ratio=config.pruning.min_retention_ratio,
            max_retention_ratio=config.pruning.max_retention_ratio,
            coverage_q=config.pruning.coverage_q,
            coverage_tail_gamma=config.pruning.coverage_tail_gamma,
            coverage_eta=config.pruning.coverage_eta,
        )
        per_fold.append({"fold": fold_no, "metrics": metrics})
        print(
            json.dumps(
                {
                    "fold": fold_no,
                    "positive_f1": metrics["positive_f1"],
                    "auroc": metrics["auroc"],
                    "candidate_edges": metrics["candidate_edges"],
                    "kept_edges": metrics["kept_edges"],
                    "kept_ibg_density": metrics["kept_ibg_density"],
                    "kept_stc_dup_frac_0_7": metrics["kept_stc_dup_frac_0_7"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    result = {
        "_purpose": "P1' structure baseline replay (fold-validation only; test never touched).",
        "config": str(args.config),
        "run_dir": str(args.run_dir),
        "global_epoch": global_epoch,
        "per_fold": per_fold,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "folds": len(per_fold)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
