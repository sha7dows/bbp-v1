"""作用：仅在 VINE 固定开发集五折上使用 Optuna 搜索已启用的非 BeHM 超参数，不访问最终测试集。"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset
from bbp_behm.hypergraph import HyperedgeBuilder
from bbp_behm.training import evaluate, make_loader, resolve_device, set_seed, train_epoch
from scripts.train_vine_cv5 import aggregate_epoch_metrics, build_fold_model, json_ready, select_global_epoch


@dataclass(frozen=True)
class OptunaSettings:
    """保存受限搜索空间和逐折剪枝参数。"""

    n_trials: int
    sampler_seed: int
    n_startup_trials: int
    prune_after_fold: int
    learning_rate_low: float
    learning_rate_high: float
    weight_decay_low: float
    weight_decay_high: float
    selector_dropout_low: float
    selector_dropout_high: float
    positive_class_weight_low: float
    positive_class_weight_high: float
    coverage_threshold_low: float
    coverage_threshold_high: float

    def validate(self) -> None:
        if self.n_trials <= 0:
            raise ValueError("optuna.n_trials must be positive.")
        if self.prune_after_fold < 2:
            raise ValueError("optuna.prune_after_fold must be at least 2.")
        ranges = (
            ("learning_rate", self.learning_rate_low, self.learning_rate_high),
            ("weight_decay", self.weight_decay_low, self.weight_decay_high),
            ("selector_dropout", self.selector_dropout_low, self.selector_dropout_high),
            ("positive_class_weight", self.positive_class_weight_low, self.positive_class_weight_high),
            ("coverage_threshold", self.coverage_threshold_low, self.coverage_threshold_high),
        )
        for name, low, high in ranges:
            if not 0.0 < low <= high:
                raise ValueError(f"Invalid Optuna range for {name}: {low}..{high}")
        if self.positive_class_weight_high > 2.0:
            raise ValueError("positive_class_weight_high must not exceed the current value 2.0.")
        if self.coverage_threshold_high > 1.0:
            raise ValueError("coverage_threshold_high must not exceed 1.0.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search VINE non-BeHM active hyperparameters with fixed development-set CV5."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--cv-run-dir",
        type=Path,
        required=True,
        help="Existing CV5 run used only for its fixed split.json manifest.",
    )
    parser.add_argument("--mode", choices=("baseline", "bbp"), default="bbp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, help="Optional override for optuna.n_trials.")
    parser.add_argument("--run-name", type=str, default="vine_ibg_optuna_cv5_seed42")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/optuna"))
    parser.add_argument("--no-progress", action="store_true", help="Reserved for command compatibility.")
    return parser.parse_args()


def load_optuna_settings(path: Path) -> OptunaSettings:
    """读取独立 optuna 配置段，避免改变现有实验配置加载逻辑。"""

    parser = configparser.ConfigParser()
    with path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    if not parser.has_section("optuna"):
        raise ValueError("Optuna search requires an [optuna] section.")
    settings = OptunaSettings(
        n_trials=parser.getint("optuna", "n_trials", fallback=15),
        sampler_seed=parser.getint("optuna", "sampler_seed", fallback=42),
        n_startup_trials=parser.getint("optuna", "n_startup_trials", fallback=3),
        prune_after_fold=parser.getint("optuna", "prune_after_fold", fallback=2),
        learning_rate_low=parser.getfloat("optuna", "learning_rate_low"),
        learning_rate_high=parser.getfloat("optuna", "learning_rate_high"),
        weight_decay_low=parser.getfloat("optuna", "weight_decay_low"),
        weight_decay_high=parser.getfloat("optuna", "weight_decay_high"),
        selector_dropout_low=parser.getfloat("optuna", "selector_dropout_low"),
        selector_dropout_high=parser.getfloat("optuna", "selector_dropout_high"),
        positive_class_weight_low=parser.getfloat("optuna", "positive_class_weight_low"),
        positive_class_weight_high=parser.getfloat("optuna", "positive_class_weight_high"),
        coverage_threshold_low=parser.getfloat("optuna", "coverage_threshold_low"),
        coverage_threshold_high=parser.getfloat("optuna", "coverage_threshold_high"),
    )
    settings.validate()
    return settings


def load_split_manifest(path: Path) -> tuple[list[tuple[int, tuple[int, ...], tuple[int, ...]]], dict[str, Any]]:
    """只读取既有开发集折索引，不重新划分且不读取最终测试索引。"""

    split_path = path / "split.json"
    with split_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    folds = []
    for detail in manifest.get("folds_detail", []):
        folds.append(
            (
                int(detail["fold"]),
                tuple(int(index) for index in detail["train"]),
                tuple(int(index) for index in detail["validation"]),
            )
        )
    if not folds:
        raise ValueError(f"No development folds found in {split_path}")
    return folds, manifest


def suggest_config(base_config, trial, settings: OptunaSettings):
    """仅替换五个已启用的非 BeHM 超参数，其余结构和关闭机制保持原值。"""

    learning_rate = trial.suggest_float(
        "learning_rate", settings.learning_rate_low, settings.learning_rate_high, log=True
    )
    weight_decay = trial.suggest_float(
        "weight_decay", settings.weight_decay_low, settings.weight_decay_high, log=True
    )
    selector_dropout = trial.suggest_float(
        "selector_dropout", settings.selector_dropout_low, settings.selector_dropout_high
    )
    positive_class_weight = trial.suggest_float(
        "positive_class_weight",
        settings.positive_class_weight_low,
        settings.positive_class_weight_high,
    )
    coverage_threshold = trial.suggest_float(
        "coverage_threshold", settings.coverage_threshold_low, settings.coverage_threshold_high
    )
    return replace(
        base_config,
        training=replace(base_config.training, learning_rate=learning_rate, weight_decay=weight_decay),
        model=replace(
            base_config.model,
            selector_dropout=selector_dropout,
            positive_class_weight=positive_class_weight,
        ),
        pruning=replace(base_config.pruning, coverage_threshold=coverage_threshold),
    )


def write_best_config(source_path: Path, output_path: Path, values: dict[str, float]) -> None:
    """基于搜索 INI 生成可直接用于确认训练的最佳独立配置。"""

    parser = configparser.ConfigParser()
    with source_path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    parser.set("training", "learning_rate", f"{values['learning_rate']:.10g}")
    parser.set("training", "weight_decay", f"{values['weight_decay']:.10g}")
    parser.set("model", "selector_dropout", f"{values['selector_dropout']:.10g}")
    parser.set("model", "positive_class_weight", f"{values['positive_class_weight']:.10g}")
    parser.set("pruning", "coverage_threshold", f"{values['coverage_threshold']:.10g}")
    # 最佳配置供后续确认使用，不应将搜索控制项误当作训练配置。
    parser.remove_section("optuna")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# 作用：由开发集 CV5 Optuna 搜索选出的 VINE 候选配置；尚未据此选择最终测试集结果。\n")
        parser.write(handle)


def write_trial_table(path: Path, trials) -> None:
    """导出 trial 状态、目标值和参数，便于筛除结果审计。"""

    rows = []
    for trial in trials:
        row = {
            "number": trial.number,
            "state": trial.state.name,
            "value": "" if trial.value is None else trial.value,
            "selected_epoch": trial.user_attrs.get("selected_epoch", ""),
            "selected_epoch_std": trial.user_attrs.get("selected_epoch_std", ""),
            **trial.params,
        }
        rows.append(row)
    fieldnames = [
        "number",
        "state",
        "value",
        "selected_epoch",
        "selected_epoch_std",
        "learning_rate",
        "weight_decay",
        "selector_dropout",
        "positive_class_weight",
        "coverage_threshold",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError("Optuna is required. Install it in the training environment before running this script.") from error

    base_config = load_experiment_config(args.config)
    base_config.validate()
    settings = load_optuna_settings(args.config)
    if args.n_trials is not None:
        settings = replace(settings, n_trials=args.n_trials)
        settings.validate()
    if base_config.hypergraph.tau != 0.10:
        raise ValueError("hypergraph.tau must remain fixed at 0.10 for this search.")
    if base_config.training.gamma_curriculum_enabled or base_config.training.checkpoint_soup_enabled:
        raise ValueError("Disabled curriculum and checkpoint-soup mechanisms must remain disabled.")
    if base_config.training.feature_noise_std != 0.0 or base_config.model.hgnn_residual:
        raise ValueError("Disabled feature noise and HGNN residual mechanisms must remain disabled.")

    folds, manifest = load_split_manifest(args.cv_run_dir.resolve())
    if len(folds) != int(manifest["folds"]):
        raise ValueError("Fold count mismatch in split.json.")
    run_dir = (args.output_dir / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "search.log", encoding="utf-8"),
        ],
        force=True,
    )
    device = resolve_device(base_config.training.device)
    set_seed(args.seed)
    dataset = PreprocessedSessionDataset(
        base_config.data.processed_path,
        base_config.data.processed_sha256,
        base_config.data.expected_sessions,
        base_config.data.expected_nodes,
        base_config.data.expected_features,
    )
    (run_dir / "search_protocol.json").write_text(
        json.dumps(
            {
                "_purpose": "Optuna development-set CV5 search; final-test indices are intentionally not loaded",
                "source_cv_run": str(args.cv_run_dir.resolve()),
                "split_seed": manifest["seed"],
                "fold_seed": manifest["fold_seed"],
                "folds": len(folds),
                "fixed": {
                    "hypergraph_tau": base_config.hypergraph.tau,
                    "original_behm_hidden_size": base_config.model.hidden_size,
                    "original_behm_dropout": base_config.model.dropout,
                    "gamma_curriculum_enabled": base_config.training.gamma_curriculum_enabled,
                    "checkpoint_soup_enabled": base_config.training.checkpoint_soup_enabled,
                    "feature_noise_std": base_config.training.feature_noise_std,
                    "hgnn_residual": base_config.model.hgnn_residual,
                },
                "search_space": json_ready(asdict(settings)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pruning_enabled = args.mode == "bbp"
    started = time.perf_counter()

    def objective(trial) -> float:
        trial_config = suggest_config(base_config, trial, settings)
        all_histories: list[list[dict[str, Any]]] = []
        for fold_id, train_indices, validation_indices in folds:
            # 每折固定同一 seed；差异仅来自交叉验证中轮换的训练/验证会话。
            set_seed(args.seed)
            train_loader = make_loader(
                dataset, train_indices, trial_config.training.batch_size, True, args.seed
            )
            validation_loader = make_loader(
                dataset, validation_indices, trial_config.training.batch_size, False, args.seed
            )
            fold_config = replace(trial_config, model=replace(trial_config.model, louvain_seed=args.seed))
            model = build_fold_model(fold_config, dataset, train_indices, device)
            builder = HyperedgeBuilder(fold_config.hypergraph)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=fold_config.training.learning_rate,
                weight_decay=fold_config.training.weight_decay,
            )
            history = []
            for epoch in range(1, fold_config.training.epochs + 1):
                train_epoch(
                    model,
                    train_loader,
                    builder,
                    optimizer,
                    device,
                    retention_ratio=fold_config.pruning.retention_ratio,
                    pruning_enabled=pruning_enabled,
                    budget_mode=fold_config.pruning.budget_mode,
                    coverage_threshold=fold_config.pruning.coverage_threshold,
                    min_retention_ratio=fold_config.pruning.min_retention_ratio,
                    max_retention_ratio=fold_config.pruning.max_retention_ratio,
                    coverage_q=fold_config.pruning.coverage_q,
                    coverage_tail_gamma=fold_config.pruning.coverage_tail_gamma,
                    coverage_eta=fold_config.pruning.coverage_eta,
                    feature_noise_std=fold_config.training.feature_noise_std,
                )
                validation = evaluate(
                    model,
                    validation_loader,
                    builder,
                    device,
                    retention_ratio=fold_config.pruning.retention_ratio,
                    pruning_enabled=pruning_enabled,
                    budget_mode=fold_config.pruning.budget_mode,
                    coverage_threshold=fold_config.pruning.coverage_threshold,
                    min_retention_ratio=fold_config.pruning.min_retention_ratio,
                    max_retention_ratio=fold_config.pruning.max_retention_ratio,
                    coverage_q=fold_config.pruning.coverage_q,
                    coverage_tail_gamma=fold_config.pruning.coverage_tail_gamma,
                    coverage_eta=fold_config.pruning.coverage_eta,
                )
                history.append({"epoch": epoch, "validation": validation})
            all_histories.append(history)
            partial = aggregate_epoch_metrics(all_histories, "positive_f1")
            partial_best = select_global_epoch(partial)
            trial.report(float(partial_best["selection_mean"]), step=fold_id)
            logging.info(
                "Trial %d fold=%d partial_best_epoch=%d partial_positive_f1=%.6f",
                trial.number,
                fold_id,
                partial_best["epoch"],
                partial_best["selection_mean"],
            )
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if fold_id >= settings.prune_after_fold and trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned after fold {fold_id}: partial_positive_f1={partial_best['selection_mean']:.6f}"
                )

        aggregate = aggregate_epoch_metrics(all_histories, "positive_f1")
        selected = select_global_epoch(aggregate)
        trial.set_user_attr("selected_epoch", int(selected["epoch"]))
        trial.set_user_attr("selected_epoch_std", float(selected["selection_std"]))
        trial.set_user_attr("mean_auroc", float(selected["auroc_mean"]))
        trial.set_user_attr("mean_precision", float(selected["precision_mean"]))
        trial.set_user_attr("mean_recall", float(selected["recall_mean"]))
        logging.info(
            "Trial %d completed: epoch=%d positive_f1=%.6f std=%.6f auroc=%.6f params=%s",
            trial.number,
            selected["epoch"],
            selected["selection_mean"],
            selected["selection_std"],
            selected["auroc_mean"],
            json.dumps(trial.params, sort_keys=True),
        )
        return float(selected["selection_mean"])

    storage = f"sqlite:///{(run_dir / 'study.db').as_posix()}"
    study = optuna.create_study(
        study_name=args.run_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.sampler_seed, n_startup_trials=settings.n_startup_trials),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=settings.n_startup_trials,
            n_warmup_steps=settings.prune_after_fold - 1,
        ),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=settings.n_trials)
    write_trial_table(run_dir / "trials.csv", study.trials)
    best = study.best_trial
    best_summary = {
        "_purpose": "Best Optuna trial selected only by fixed development-set CV5 positive F1",
        "best_trial": best.number,
        "positive_f1": best.value,
        "params": best.params,
        "user_attrs": best.user_attrs,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (run_dir / "best_trial.json").write_text(
        json.dumps(best_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_best_config(args.config, run_dir / "best_config.ini", best.params)
    logging.info("Best trial=%d positive_f1=%.6f params=%s", best.number, best.value, json.dumps(best.params))
    logging.info("Optuna CV5 search completed in %.2f seconds: %s", best_summary["elapsed_seconds"], run_dir)


if __name__ == "__main__":
    main()
