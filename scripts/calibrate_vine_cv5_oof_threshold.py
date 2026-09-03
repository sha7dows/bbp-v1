"""作用：从已完成的 VINE CV5 折检查点生成开发集 OOF 概率，并在不访问最终测试集的前提下校准统一分类阈值。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset
from bbp_behm.hypergraph import HyperedgeBuilder
from bbp_behm.training import make_loader, resolve_device, set_seed
from scripts.train_vine_cv5 import build_fold_model, classification_metrics, collect_positive_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate one VINE decision threshold from CV5 out-of-fold development predictions."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--cv-run-dir",
        type=Path,
        required=True,
        help="Completed CV5 run containing split.json and fold_checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <cv-run-dir>/oof_threshold_calibration.",
    )
    parser.add_argument("--device", type=str, help="Optional device override, for example cuda:2 or cpu.")
    parser.add_argument(
        "--processed-path",
        type=Path,
        help="Optional override for data.processed_path (e.g. local snapshot when config points to server).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 清单，并在缺失时给出明确错误。"""

    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metrics_at_threshold(
    labels: list[int], positive_probabilities: list[float], threshold: float
) -> dict[str, float]:
    """使用指定阈值计算分类指标，保留 AUROC 便于与 0.5 基线对照。"""

    predictions = [int(value >= threshold) for value in positive_probabilities]
    # 复用已有指标实现时先把概率转为 0/1，使其在固定阈值下得到相同 F1、精确率和召回率。
    metrics = classification_metrics(labels, [float(value) for value in predictions])
    # AUROC 只反映排序，不应因为阈值改变；用真实概率单独重算。
    metrics["auroc"] = classification_metrics(labels, positive_probabilities)["auroc"]
    return metrics


def threshold_curve(labels: list[int], positive_probabilities: list[float]) -> list[dict[str, float]]:
    """枚举 OOF 概率形成的全部有效决策点，并包含固定 0.5 基线。"""

    candidates = sorted({0.0, 0.5, 1.0, *(float(value) for value in positive_probabilities)})
    return [
        {"threshold": threshold, **metrics_at_threshold(labels, positive_probabilities, threshold)}
        for threshold in candidates
    ]


def choose_threshold(curve: list[dict[str, float]]) -> dict[str, float]:
    """优先最大化正类 F1；完全并列时依次选择更高宏 F1 和更接近 0.5 的阈值。"""

    if not curve:
        raise ValueError("Threshold curve is empty.")
    return max(
        curve,
        key=lambda item: (
            item["positive_f1"],
            item["macro_f1"],
            -abs(item["threshold"] - 0.5),
        ),
    )


def expected_labels(dataset: PreprocessedSessionDataset, indices: list[int]) -> list[int]:
    """读取验证会话标签，用于验证 DataLoader 输出顺序未发生变化。"""

    tensor_indices = torch.tensor(indices, dtype=torch.long)
    return [int(value) for value in dataset.labels[tensor_indices].detach().cpu().tolist()]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写入可直接检查的 CSV 结果文件。"""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    config.validate()
    if args.device is not None:
        config = replace(config, training=replace(config.training, device=args.device))
    if args.processed_path is not None:
        config = replace(config, data=replace(config.data, processed_path=args.processed_path))

    cv_run_dir = args.cv_run_dir.resolve()
    output_dir = (args.output_dir or cv_run_dir / "oof_threshold_calibration").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "calibration.log", encoding="utf-8"),
        ],
        force=True,
    )

    split_manifest = load_json(cv_run_dir / "split.json")
    result = load_json(cv_run_dir / "result.json")
    folds = split_manifest.get("folds_detail")
    if not isinstance(folds, list) or not folds:
        raise ValueError("split.json does not contain folds_detail.")
    selected_epoch = int(result["cv"]["global_epoch"])
    configured_folds = int(result["cv"]["folds"])
    if len(folds) != configured_folds:
        raise ValueError("split.json fold count differs from result.json.")

    device = resolve_device(config.training.device)
    set_seed(int(split_manifest["seed"]))
    dataset = PreprocessedSessionDataset(
        config.data.processed_path,
        config.data.processed_sha256,
        config.data.expected_sessions,
        config.data.expected_nodes,
        config.data.expected_features,
    )

    # 仅从清单读取每折验证集；最终测试索引在此脚本中不参与 loader 构建或预测。
    oof_records: list[dict[str, Any]] = []
    for fold_detail in folds:
        fold_id = int(fold_detail["fold"])
        train_indices = [int(index) for index in fold_detail["train"]]
        validation_indices = [int(index) for index in fold_detail["validation"]]
        checkpoint_path = cv_run_dir / "fold_checkpoints" / f"fold_{fold_id:02d}_epoch_{selected_epoch:03d}.pt"
        # 检查点仅包含张量和基础元数据；显式限制为权重加载，避免未来 PyTorch 默认值变更告警。
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if int(checkpoint["fold"]) != fold_id or int(checkpoint["global_epoch"]) != selected_epoch:
            raise ValueError(f"Checkpoint metadata does not match fold {fold_id}: {checkpoint_path}")

        fold_seed = int(checkpoint["fold_seed"])
        fold_config = replace(config, model=replace(config.model, louvain_seed=fold_seed))
        model = build_fold_model(fold_config, dataset, tuple(train_indices), device)
        model.load_state_dict(checkpoint["model"])
        validation_loader = make_loader(
            dataset,
            validation_indices,
            fold_config.training.batch_size,
            False,
            fold_seed,
        )
        labels, probabilities = collect_positive_probabilities(
            model,
            validation_loader,
            HyperedgeBuilder(fold_config.hypergraph),
            device,
            retention_ratio=fold_config.pruning.retention_ratio,
            pruning_enabled=result["mode"] == "bbp",
            budget_mode=fold_config.pruning.budget_mode,
            coverage_threshold=fold_config.pruning.coverage_threshold,
            min_retention_ratio=fold_config.pruning.min_retention_ratio,
            max_retention_ratio=fold_config.pruning.max_retention_ratio,
            coverage_q=fold_config.pruning.coverage_q,
            coverage_tail_gamma=fold_config.pruning.coverage_tail_gamma,
            coverage_eta=fold_config.pruning.coverage_eta,
        )
        if labels != expected_labels(dataset, validation_indices):
            raise RuntimeError(f"Validation label order mismatch in fold {fold_id}.")
        if len(probabilities) != len(validation_indices):
            raise RuntimeError(f"Prediction count mismatch in fold {fold_id}.")
        oof_records.extend(
            {
                "fold": fold_id,
                "session_index": session_index,
                "label": label,
                "positive_probability": probability,
            }
            for session_index, label, probability in zip(validation_indices, labels, probabilities)
        )
        logging.info("OOF predictions collected: fold=%d sessions=%d", fold_id, len(validation_indices))

    expected_development = sorted(int(index) for index in split_manifest["development_indices"])
    observed_development = sorted(record["session_index"] for record in oof_records)
    if observed_development != expected_development:
        raise RuntimeError("OOF session indices are not an exact partition of the development set.")

    labels = [int(record["label"]) for record in oof_records]
    positive_probabilities = [float(record["positive_probability"]) for record in oof_records]
    curve = threshold_curve(labels, positive_probabilities)
    baseline = next(item for item in curve if item["threshold"] == 0.5)
    selected = choose_threshold(curve)
    summary = {
        "_purpose": "OOF threshold selection from fixed CV development folds; no final-test predictions are loaded",
        "source_cv_run": str(cv_run_dir),
        "source_global_epoch": selected_epoch,
        "folds": configured_folds,
        "development_sessions": len(oof_records),
        "baseline_threshold": baseline,
        "selected_threshold": selected,
        "selection_rule": "maximize OOF positive_f1, then macro_f1, then nearest threshold to 0.5",
        "oof_auroc": classification_metrics(labels, positive_probabilities)["auroc"],
    }
    (output_dir / "calibration_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(
        output_dir / "oof_predictions.csv",
        oof_records,
        ["fold", "session_index", "label", "positive_probability"],
    )
    write_csv(
        output_dir / "threshold_curve.csv",
        curve,
        ["threshold", "macro_f1", "positive_f1", "precision", "recall", "auroc"],
    )
    logging.info("OOF baseline threshold=0.500000 metrics=%s", json.dumps(baseline, sort_keys=True))
    logging.info("OOF selected threshold=%.6f metrics=%s", selected["threshold"], json.dumps(selected, sort_keys=True))
    logging.info("OOF threshold calibration completed: %s", output_dir)


if __name__ == "__main__":
    main()
