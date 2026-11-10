"""用途：以固定最终测试集和开发集五折交叉验证训练 VINE，按跨折验证指标选择全局 epoch。"""

from __future__ import annotations

import argparse
import configparser
import json
import logging
import statistics
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset
from bbp_behm.hypergraph import HyperedgeBuilder
from bbp_behm.model import BBPBeHM
from bbp_behm.hsl_association import AssociationBBPBeHM
from bbp_behm.cached_hypergraph import CachedHyperedgeBuilder
from bbp_behm.training import (
    class_priors,
    confidence_statistics,
    evaluate,
    make_loader,
    resolve_device,
    set_seed,
    train_epoch,
)


@dataclass(frozen=True)
class CVSettings:
    """保存独立五折协议的可复现参数。"""

    folds: int
    fold_seed: int

    def validate(self) -> None:
        if self.folds < 2:
            raise ValueError("cv.folds must be at least 2.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VINE with a fixed final holdout and stratified development-set cross-validation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "bbp"), default="bbp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture",choices=("original","association"),default="original")
    parser.add_argument("--association-mode",choices=("none","raw","context"),default="context")
    parser.add_argument("--association-alpha",type=float,default=0.2)
    parser.add_argument("--cache-graphs",action="store_true")
    parser.add_argument("--torch-threads",type=int)
    workers = parser.add_mutually_exclusive_group()
    workers.add_argument("--worker-fold",type=int,choices=range(1,6))
    workers.add_argument("--aggregate-workers",action="store_true")
    parser.add_argument("--prepared-cache",type=Path)
    parser.add_argument("--total-started-at",type=float)


    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--selection-metric",
        choices=("positive_f1", "macro_f1"),
        default="positive_f1",
        help="Cross-fold validation metric used to select one global epoch.",
    )
    parser.add_argument(
        "--evaluate-final-test",
        action="store_true",
        help="Evaluate the fixed final holdout after selection; leave disabled while comparing configurations.",
    )
    parser.add_argument("--run-name", type=str, default="vine_ibg_cv5_seed42")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cv5"))
    parser.add_argument("--no-progress", action="store_true", help="Reserved for command compatibility.")
    return parser.parse_args()


def load_cv_settings(path: Path) -> CVSettings:
    """读取独立 cv 段，避免改变现有实验配置加载逻辑。"""

    parser = configparser.ConfigParser()
    with path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    if not parser.has_section("cv"):
        raise ValueError("CV training requires a [cv] section in the INI file.")
    settings = CVSettings(
        folds=parser.getint("cv", "folds", fallback=5),
        fold_seed=parser.getint("cv", "fold_seed", fallback=42),
    )
    settings.validate()
    return settings


def clone_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """复制当前 checkpoint 到 CPU，供跨折全局 epoch 选择后复用。"""

    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def label_counts(dataset: PreprocessedSessionDataset, indices: Iterable[int]) -> dict[str, int]:
    """统计会话级标签数量，作为划分清单的一部分。"""

    selected = torch.tensor(tuple(int(index) for index in indices), dtype=torch.long)
    counts = torch.bincount(dataset.labels[selected], minlength=2)
    return {"negative": int(counts[0]), "positive": int(counts[1])}


def json_ready(value: Any) -> Any:
    """递归转换路径和 NumPy/Torch 标量，保证结果 JSON 可序列化。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def build_fold_model(config, dataset, train_indices: tuple[int, ...], device: torch.device, *, architecture="original", association_mode="context", association_alpha=0.2) -> BBPBeHM:
    """只利用当前折训练会话构建类别先验和置信度统计，避免验证或测试泄漏。"""

    priors = class_priors(dataset, train_indices, config.model.num_classes)
    confidence_class_means, confidence_global_mean = confidence_statistics(
        dataset, train_indices, config.model.num_classes
    )
    model_class = AssociationBBPBeHM if architecture == "association" else BBPBeHM
    options = dict(association_mode=association_mode,association_alpha=association_alpha) if architecture == "association" else {}
    return model_class(
        config,
        priors,
        confidence_class_means=confidence_class_means,
        confidence_global_mean=confidence_global_mean,
        **options,
    ).to(device)


@torch.no_grad()
def collect_positive_probabilities(
    model: BBPBeHM,
    loader,
    builder: HyperedgeBuilder,
    device: torch.device,
    *,
    retention_ratio: float,
    pruning_enabled: bool,
    budget_mode: str,
    coverage_threshold: float,
    min_retention_ratio: float,
    max_retention_ratio: float,
    coverage_q: float,
    coverage_tail_gamma: float,
    coverage_eta: float,
) -> tuple[list[int], list[float]]:
    """收集固定最终测试集的概率，供五模型概率平均后进行一次最终评估。"""

    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    for cpu_batch in loader:
        batch = cpu_batch.to(device)
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
        labels.extend(batch.labels.detach().cpu().tolist())
        probabilities.extend(torch.softmax(output.logits, dim=1)[:, 1].detach().cpu().tolist())
    return labels, probabilities


def classification_metrics(labels: list[int], positive_probabilities: list[float]) -> dict[str, float]:
    """以固定 0.5 决策点计算最终集成分类指标。"""

    predictions = [int(value >= 0.5) for value in positive_probabilities]
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "auroc": float(roc_auc_score(labels, positive_probabilities)),
    }


def aggregate_epoch_metrics(fold_histories: list[list[dict[str, Any]],], metric: str) -> list[dict[str, Any]]:
    """按 epoch 聚合五折验证指标，并保留均值和样本标准差。"""

    epochs = len(fold_histories[0])
    if any(len(history) != epochs for history in fold_histories):
        raise ValueError("All folds must complete the same number of epochs.")
    aggregate = []
    for epoch_offset in range(epochs):
        validations = [history[epoch_offset]["validation"] for history in fold_histories]
        metric_values = [float(values[metric]) for values in validations]
        aggregate.append(
            {
                "epoch": epoch_offset + 1,
                "selection_mean": float(statistics.fmean(metric_values)),
                "selection_std": float(statistics.stdev(metric_values)) if len(metric_values) > 1 else 0.0,
                "macro_f1_mean": float(statistics.fmean(float(values["macro_f1"]) for values in validations)),
                "positive_f1_mean": float(statistics.fmean(float(values["positive_f1"]) for values in validations)),
                "precision_mean": float(statistics.fmean(float(values["precision"]) for values in validations)),
                "recall_mean": float(statistics.fmean(float(values["recall"]) for values in validations)),
                "auroc_mean": float(statistics.fmean(float(values["auroc"]) for values in validations)),
            }
        )
    return aggregate


def select_global_epoch(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    """先最大化跨折验证均值，再最小化标准差，最后选择更早的 epoch。"""

    return max(
        aggregate,
        key=lambda item: (item["selection_mean"], -item["selection_std"], -item["epoch"]),
    )


def main() -> None:
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    model_options = dict(architecture=args.architecture,association_mode=args.association_mode,
                         association_alpha=args.association_alpha)
    config = load_experiment_config(args.config)
    config.validate()
    settings = load_cv_settings(args.config)
    if config.training.checkpoint_soup_enabled:
        raise ValueError("CV global-epoch selection does not support checkpoint soup.")
    if config.training.gamma_curriculum_enabled:
        raise ValueError("CV global-epoch selection currently requires gamma_curriculum_enabled=false.")
    if config.split.test_fraction <= 0.0 or config.split.test_fraction >= 1.0:
        raise ValueError("test_fraction must be in (0, 1) to define a development-set protocol.")
    if config.split.validation_fraction < 0.0:
        raise ValueError("validation_fraction must be non-negative.")

    epochs = config.training.epochs if args.epochs is None else args.epochs
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    device = resolve_device(config.training.device)
    pruning_enabled = args.mode == "bbp"
    run_dir = args.output_dir / args.run_name
    if (run_dir / "result.json").exists() or (run_dir / "cv_progress.json").exists():
        raise FileExistsError("Refusing to overwrite an existing CV run")
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "training.log", encoding="utf-8")],
        force=True,
    )

    set_seed(args.seed)
    dataset = PreprocessedSessionDataset(
        config.data.processed_path,
        config.data.processed_sha256,
        config.data.expected_sessions,
        config.data.expected_nodes,
        config.data.expected_features,
    )
    # 复用原有外层分层逻辑，保证固定最终测试集与现有 seed 协议一致。
    base_split = dataset.make_split(
        seed=args.seed,
        test_fraction=config.split.test_fraction,
        validation_fraction=config.split.validation_fraction,
    )
    final_test_indices = tuple(int(index) for index in base_split.test)
    development_indices = tuple(sorted((*base_split.train, *base_split.validation)))
    development_labels = dataset.labels[list(development_indices)].cpu().numpy()
    splitter = StratifiedKFold(n_splits=settings.folds, shuffle=True, random_state=settings.fold_seed)
    fold_indices = []
    for fold_id, (train_positions, validation_positions) in enumerate(
        splitter.split(np.zeros(len(development_indices)), development_labels), start=1
    ):
        train_indices = tuple(development_indices[position] for position in train_positions)
        validation_indices = tuple(development_indices[position] for position in validation_positions)
        fold_indices.append((fold_id, train_indices, validation_indices))

    split_manifest = {
        "_purpose": "Fixed final holdout plus stratified development-set cross-validation split",
        "seed": args.seed,
        "fold_seed": settings.fold_seed,
        "folds": settings.folds,
        "final_test_fraction": config.split.test_fraction,
        "development_indices": list(development_indices),
        "final_test_indices": list(final_test_indices),
        "class_counts": {
            "development": label_counts(dataset, development_indices),
            "final_test": label_counts(dataset, final_test_indices),
        },
        "folds_detail": [
            {
                "fold": fold_id,
                "train": list(train_indices),
                "validation": list(validation_indices),
                "class_counts": {
                    "train": label_counts(dataset, train_indices),
                    "validation": label_counts(dataset, validation_indices),
                },
            }
            for fold_id, train_indices, validation_indices in fold_indices
        ],
        "processed_sha256": config.data.processed_sha256,
    }
    (run_dir / "split.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    logging.info(
        "CV protocol: final_test=%d development=%d folds=%d final_test_labels=%s",
        len(final_test_indices),
        len(development_indices),
        settings.folds,
        split_manifest["class_counts"]["final_test"],
    )

    ratio = config.pruning.retention_ratio
    budget_mode = config.pruning.budget_mode
    coverage_threshold = config.pruning.coverage_threshold
    min_retention_ratio = config.pruning.min_retention_ratio
    max_retention_ratio = config.pruning.max_retention_ratio
    coverage_q = config.pruning.coverage_q
    coverage_tail_gamma = config.pruning.coverage_tail_gamma
    coverage_eta = config.pruning.coverage_eta
    feature_noise_std = config.training.feature_noise_std
    all_fold_histories: list[list[dict[str, Any]]] = []
    fold_states: list[dict[int, dict[str, torch.Tensor]]] = []
    started = time.perf_counter()

    # 仅缓存原始无标签确定性候选图；所有折训练统计仍独立计算。
    cached_builder = None
    if args.cache_graphs and not args.aggregate_workers:
        if args.prepared_cache is not None:
            saved = torch.load(args.prepared_cache,map_location="cpu",weights_only=False)
            if (saved["indices"] != list(development_indices) or
                saved["processed_sha256"] != config.data.processed_sha256 or
                saved["hypergraph_config"] != asdict(config.hypergraph)):
                raise ValueError("Prepared graph cache metadata mismatch")
            cached_builder = CachedHyperedgeBuilder(saved["cache"])
        else:
            cached_builder = CachedHyperedgeBuilder.prepare(dataset,development_indices,config.hypergraph,device)

    prediction_dir = run_dir / "validation_predictions"
    prediction_dir.mkdir(exist_ok=True)
    epoch_checkpoint_dir = run_dir / "epoch_checkpoints"
    epoch_checkpoint_dir.mkdir(exist_ok=True)
    training_folds = [] if args.aggregate_workers else [
        fold for fold in fold_indices if args.worker_fold is None or fold[0] == args.worker_fold]
    for fold_id, train_indices, validation_indices in training_folds:
        # 五折仅轮换开发集的训练/验证会话，所有折保持同一训练随机种子。
        fold_seed = args.seed
        fold_config = replace(config, model=replace(config.model, louvain_seed=fold_seed))
        set_seed(fold_seed)
        train_loader = make_loader(
            dataset, train_indices, fold_config.training.batch_size, True, fold_seed
        )
        validation_loader = make_loader(
            dataset, validation_indices, fold_config.training.batch_size, False, fold_seed
        )
        model = build_fold_model(fold_config, dataset, train_indices, device, **model_options)
        builder = cached_builder or HyperedgeBuilder(fold_config.hypergraph)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=fold_config.training.learning_rate,
            weight_decay=fold_config.training.weight_decay,
        )
        history: list[dict[str, Any]] = []
        states: dict[int, dict[str, torch.Tensor]] = {}
        logging.info(
            "Fold %d/%d started: seed=%d train=%d validation=%d train_labels=%s validation_labels=%s",
            fold_id,
            settings.folds,
            fold_seed,
            len(train_indices),
            len(validation_indices),
            label_counts(dataset, train_indices),
            label_counts(dataset, validation_indices),
        )
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
            summary = train_epoch(
                model,
                train_loader,
                builder,
                optimizer,
                device,
                retention_ratio=ratio,
                pruning_enabled=pruning_enabled,
                budget_mode=budget_mode,
                coverage_threshold=coverage_threshold,
                min_retention_ratio=min_retention_ratio,
                max_retention_ratio=max_retention_ratio,
                coverage_q=coverage_q,
                coverage_tail_gamma=coverage_tail_gamma,
                coverage_eta=coverage_eta,
                feature_noise_std=feature_noise_std,
            )
            predictions = []
            validation_metrics = evaluate(
                model,
                validation_loader,
                builder,
                device,
                retention_ratio=ratio,
                pruning_enabled=pruning_enabled,
                budget_mode=budget_mode,
                coverage_threshold=coverage_threshold,
                min_retention_ratio=min_retention_ratio,
                max_retention_ratio=max_retention_ratio,
                coverage_q=coverage_q,
                coverage_tail_gamma=coverage_tail_gamma,
                coverage_eta=coverage_eta,
                predictions_out=predictions,
            )
            (prediction_dir / f"fold_{fold_id:02d}_epoch_{epoch:03d}.json").write_text(
                json.dumps(predictions),encoding="utf-8")
            states[epoch] = clone_state_dict_to_cpu(model)
            history.append({"epoch":epoch,"train":asdict(summary),"validation":validation_metrics,
                            "epoch_seconds":time.perf_counter()-epoch_started,
                            "peak_memory_mb":torch.cuda.max_memory_allocated(device)/1024**2 if device.type=="cuda" else None})
            torch.save({"model":states[epoch],"fold":fold_id,"epoch":epoch,"model_options":model_options},
                       epoch_checkpoint_dir/f"fold_{fold_id:02d}_epoch_{epoch:03d}.pt")
            (run_dir/"cv_progress.json").write_text(json.dumps({"phase":"training","fold":fold_id,
                "epoch":epoch,"folds":settings.folds,"epochs":epochs,"last_validation":validation_metrics}),encoding="utf-8")
            (run_dir/f"fold_{fold_id:02d}_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
            logging.info("ASSOCIATION fold=%d epoch=%d mean_abs_delta=%.8f row_mass_error=%.8g epoch_seconds=%.2f peak_memory_mb=%.1f",
                fold_id,epoch,validation_metrics.get("association_mean_abs_weight_delta",0),
                validation_metrics.get("association_row_mass_max_error",0),history[-1]["epoch_seconds"],history[-1]["peak_memory_mb"] or 0)

            logging.info(
                "Fold %d epoch %03d train_loss=%.6f val_%s=%.6f val_auroc=%.6f",
                fold_id,
                epoch,
                summary.loss,
                args.selection_metric,
                validation_metrics[args.selection_metric],
                validation_metrics["auroc"],
            )
        all_fold_histories.append(history)
        fold_states.append(states)

    if args.worker_fold is not None:
        if args.evaluate_final_test:
            raise ValueError("Individual workers must not evaluate the final holdout")
        (run_dir/"worker_complete.json").write_text(json.dumps({"fold":args.worker_fold,"epochs":epochs,
            "model_options":model_options,"final_test_evaluated":False}),encoding="utf-8")
        logging.info("CV_FOLD_WORKER_COMPLETE fold=%d epochs=%d",args.worker_fold,epochs)
        return

    if args.aggregate_workers:
        # 五折独立训练完毕后才聚合；所有checkpoint仍以共同epoch统一选择。
        for fold_id,_,_ in fold_indices:
            worker = run_dir/"workers"/f"fold_{fold_id:02d}"
            completed = json.loads((worker/"worker_complete.json").read_text())
            if completed["epochs"] != epochs or completed["model_options"] != model_options:
                raise ValueError("Worker protocol mismatch")
            worker_split = json.loads((worker/"split.json").read_text())
            if worker_split != split_manifest:
                raise ValueError("Worker split manifest mismatch")
            history = json.loads((worker/f"fold_{fold_id:02d}_history.json").read_text())
            if len(history) != epochs:
                raise ValueError("Incomplete worker history")
            all_fold_histories.append(history)
            states = {}
            for epoch in range(1,epochs+1):
                name = f"fold_{fold_id:02d}_epoch_{epoch:03d}"
                checkpoint = torch.load(worker/"epoch_checkpoints"/(name+".pt"),map_location="cpu",weights_only=False)
                states[epoch] = checkpoint["model"]
                shutil.copyfile(worker/"epoch_checkpoints"/(name+".pt"),epoch_checkpoint_dir/(name+".pt"))
                shutil.copyfile(worker/"validation_predictions"/(name+".json"),prediction_dir/(name+".json"))
            shutil.copyfile(worker/f"fold_{fold_id:02d}_history.json",run_dir/f"fold_{fold_id:02d}_history.json")
            fold_states.append(states)

    aggregate_history = aggregate_epoch_metrics(all_fold_histories, args.selection_metric)
    selected_epoch = select_global_epoch(aggregate_history)
    selected_epoch_number = int(selected_epoch["epoch"])
    logging.info(
        "Global epoch selected from development folds only: epoch=%d mean_%s=%.6f std=%.6f mean_auroc=%.6f",
        selected_epoch_number,
        args.selection_metric,
        selected_epoch["selection_mean"],
        selected_epoch["selection_std"],
        selected_epoch["auroc_mean"],
    )

    checkpoint_dir = run_dir / "fold_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for (fold_id, train_indices, _), states in zip(fold_indices, fold_states):
        fold_seed = args.seed
        fold_config = replace(config, model=replace(config.model, louvain_seed=fold_seed))
        torch.save(
            {
                "fold": fold_id,
                "fold_seed": fold_seed,
                "global_epoch": selected_epoch_number,
                "model_options": model_options,
                "model": states[selected_epoch_number],
                "config": json_ready(asdict(fold_config)),
            },
            checkpoint_dir / f"fold_{fold_id:02d}_epoch_{selected_epoch_number:03d}.pt",
        )

    if args.evaluate_final_test:
        test_loader = make_loader(
            dataset, final_test_indices, config.training.batch_size, False, args.seed
        )
        all_test_probabilities: list[list[float]] = []
        final_test_labels: list[int] | None = None
        for (fold_id, train_indices, _), states in zip(fold_indices, fold_states):
            # 最终集成重建时保持与对应折训练阶段完全相同的随机种子。
            fold_seed = args.seed
            fold_config = replace(config, model=replace(config.model, louvain_seed=fold_seed))
            model = build_fold_model(fold_config, dataset, train_indices, device, **model_options)
            model.load_state_dict(states[selected_epoch_number])
            builder = HyperedgeBuilder(fold_config.hypergraph)
            labels, probabilities = collect_positive_probabilities(
                model,
                test_loader,
                builder,
                device,
                retention_ratio=ratio,
                pruning_enabled=pruning_enabled,
                budget_mode=budget_mode,
                coverage_threshold=coverage_threshold,
                min_retention_ratio=min_retention_ratio,
                max_retention_ratio=max_retention_ratio,
                coverage_q=coverage_q,
                coverage_tail_gamma=coverage_tail_gamma,
                coverage_eta=coverage_eta,
            )
            if final_test_labels is None:
                final_test_labels = labels
            elif final_test_labels != labels:
                raise RuntimeError("Final test label order differs across fold models.")
            all_test_probabilities.append(probabilities)

        if final_test_labels is None:
            raise RuntimeError("No final test predictions were collected.")
        ensemble_probabilities = np.mean(np.asarray(all_test_probabilities, dtype=np.float64), axis=0).tolist()
        final_test_metrics = classification_metrics(final_test_labels, ensemble_probabilities)
    else:
        final_test_metrics = {
            "evaluated": False,
            "message": "Final test was intentionally skipped. Use --evaluate-final-test only after locking the configuration.",
        }
    elapsed_seconds = time.perf_counter() - started
    if args.aggregate_workers and args.total_started_at is not None:
        elapsed_seconds = time.time() - args.total_started_at
    fold_selected_metrics = [
        history[selected_epoch_number - 1]["validation"] for history in all_fold_histories
    ]
    oof = []
    for fold_id, _, _ in fold_indices:
        oof.extend(json.loads((prediction_dir/f"fold_{fold_id:02d}_epoch_{selected_epoch_number:03d}.json").read_text()))
    assert len(oof)==len(development_indices) and {p["index"] for p in oof}==set(development_indices)
    assert len({p["index"] for p in oof})==len(oof) and not {p["index"] for p in oof}.intersection(final_test_indices)
    oof.sort(key=lambda p:p["index"])
    (run_dir/"oof_predictions.json").write_text(json.dumps(oof,indent=2),encoding="utf-8")
    oof_metrics = classification_metrics([p["label"] for p in oof],[p["positive_probability"] for p in oof])
    result = {
        "_purpose": "VINE fixed final holdout plus stratified five-fold development-set cross-validation",
        "mode": args.mode,
        "model_options": model_options,
        "graph_cache_enabled": args.cache_graphs,
        "oof_metrics_at_global_epoch": oof_metrics,
        "seed": args.seed,
        "selection_metric": args.selection_metric,
        "epochs": epochs,
        "cv": {
            "folds": settings.folds,
            "fold_seed": settings.fold_seed,
            "global_epoch": selected_epoch_number,
            "global_validation": selected_epoch,
            "per_fold_validation_at_global_epoch": fold_selected_metrics,
            "history": aggregate_history,
        },
        "final_test": {
            "protocol": "Fixed holdout is evaluated only after development-fold configuration selection",
            "evaluated": args.evaluate_final_test,
            "ensemble": "mean_positive_probability",
            "models": settings.folds,
            "threshold": 0.5,
            "metrics": final_test_metrics,
        },
        "split_sizes": {
            "development": len(development_indices),
            "final_test": len(final_test_indices),
            "fold_train": len(fold_indices[0][1]),
            "fold_validation": len(fold_indices[0][2]),
        },
        "resolved_config": json_ready(asdict(config)),
        "elapsed_seconds": elapsed_seconds,
    }
    (run_dir / "cv_history.json").write_text(
        json.dumps({"fold_histories": all_fold_histories, "aggregate": aggregate_history}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(
        {
            "global_epoch": selected_epoch_number,
            "model_options": model_options,
            "fold_checkpoints": [states[selected_epoch_number] for states in fold_states],
            "config": json_ready(asdict(config)),
            "result": result,
        },
        run_dir / "ensemble_model.pt",
    )
    if args.evaluate_final_test:
        logging.info(
            "Final fixed-holdout ensemble metrics: %s",
            json.dumps(final_test_metrics, sort_keys=True),
        )
    else:
        logging.info("Final test was intentionally skipped to preserve the fixed holdout.")
    (run_dir/"cv_progress.json").write_text(json.dumps({"phase":"complete","folds":settings.folds,
        "epochs":epochs,"global_epoch":selected_epoch_number,"final_test_evaluated":args.evaluate_final_test}),encoding="utf-8")
    logging.info("CV run completed in %.2f seconds: %s", elapsed_seconds, run_dir)


if __name__ == "__main__":
    main()
