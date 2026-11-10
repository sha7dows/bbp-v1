"""用途：在固定最终留出集之外运行类 HSL 单折消融，缓存相同候选图并记录关联精简与真实验证指标。"""

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset, collate_sessions
from bbp_behm.hsl import HSLBBPBeHM
from bbp_behm.hsl_context import ContextHSLBBPBeHM
from bbp_behm.hypergraph import BehaviorType, HyperedgeBatch, HyperedgeBuilder
from bbp_behm.model import BBPBeHM
from bbp_behm.training import class_priors, confidence_statistics, make_loader, set_seed


class CachedBuilder:
    """候选图逐会话缓存，避免不同方法重复构图或改变原始候选集。"""

    def __init__(self, cache):
        self.cache = cache

    def build_batch(self, batch):
        graphs = [self.cache[int(i)] for i in batch.session_indices.cpu().tolist()]
        device = batch.x.device
        return HyperedgeBatch(
            torch.block_diag(*[g.incidence.to(device) for g in graphs]),
            torch.cat([torch.full((g.num_edges,), i, dtype=torch.long, device=device) for i, g in enumerate(graphs)]),
            torch.cat([g.type_ids.to(device) for g in graphs]),
            torch.cat([g.structural_stats.to(device) for g in graphs]),
            [r for g in graphs for r in g.records],
        )


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--family", choices=["original", "context"], default="original")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.7)
    parser.add_argument("--modes", nargs="+", default=["bbp", "control", "feedback", "feedback_edit"])
    parser.add_argument("--budget-mode", choices=["fixed", "robust_adaptive"], default="fixed")
    return parser.parse_args()


def subset(indices, labels, limit, seed):
    if not limit or limit >= len(indices):
        return list(indices)
    result, _ = train_test_split(indices, train_size=limit, stratify=labels[indices].numpy(), random_state=seed)
    return sorted(result)


def check_invariants(model, batch, graph, output):
    """验证成员修复没有新增关联、编辑其他类型、隔离节点或超过逐边预算。"""
    original = graph.incidence * output.selection.hard_mask.detach()[:, None]
    actual = output.masked_incidence.detach()
    assert torch.isfinite(output.total_loss), "Non-finite loss"
    assert torch.all((actual == 0) | (actual == 1)), "Non-binary topology"
    assert torch.all(actual <= original), "Unexpected added incidence"
    non_stc = graph.type_ids != int(BehaviorType.STC)
    assert torch.equal(actual[non_stc], original[non_stc]), "Edited non-STC edge"
    assert torch.all((actual.sum(0) > 0) | (original.sum(0) == 0)), "New isolated node"
    sizes = original.sum(1)
    assert torch.all(sizes - actual.sum(1) <= torch.floor(sizes * model.edit_fraction)), "Exceeded edit budget"
    if model.hsl_mode == "feedback_edit":
        assert torch.all((sizes == 0) | (actual.sum(1) >= 2)), "Invalid small edge"
    if model.config.pruning.budget_mode == "fixed":
        for session in range(batch.num_graphs):
            count = int((graph.edge_to_session == session).sum())
            expected = max(1, int(np.ceil(model._experiment_ratio * count))) if count else 0
            assert output.selection.kept_per_session[session] == expected, "Changed edge budget"


def validate(model, loader, builder, config, ratio, device, check=False):
    model.eval()
    labels, probabilities, predictions = [], [], []
    totals = {k: 0.0 for k in ["candidates", "kept", "selected_incidence", "actual_incidence", "edits", "changed", "coverage", "structural_coverage"]}
    count = 0
    start = time.perf_counter()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            graph = builder.build_batch(batch)
            output = model(batch, graph, retention_ratio=ratio, budget_mode=config.pruning.budget_mode, stochastic_selection=False)
            if check:
                check_invariants(model, batch, graph, output)
            prob = output.logits.softmax(1)[:, 1].cpu().tolist()
            ys = batch.labels.cpu().tolist()
            labels.extend(ys)
            probabilities.extend(prob)
            predictions.extend({"index": int(i), "label": y, "positive_probability": p} for i, y, p in zip(batch.session_indices.cpu().tolist(), ys, prob))
            selected = graph.incidence * output.selection.hard_mask[:, None]
            totals["candidates"] += graph.num_edges
            totals["kept"] += float(output.selection.hard_mask.sum())
            totals["selected_incidence"] += float(selected.sum())
            totals["actual_incidence"] += float(output.masked_incidence.sum())
            totals["edits"] += model.last_hsl_stats["edits"]
            totals["changed"] += model.last_hsl_stats["selection_changed"]
            totals["coverage"] += sum(output.selection.coverage_per_session.values())
            # 用原候选对实际保留成员求同类型 Jaccard，单独揭示原覆盖口径看不到的编辑失真。
            for session in range(batch.num_graphs):
                local_values = []
                for kind in torch.unique(graph.type_ids[graph.edge_to_session == session]):
                    rows = (graph.edge_to_session == session) & (graph.type_ids == kind)
                    a, b = graph.incidence[rows], output.masked_incidence[rows]
                    intersection = a @ b.t()
                    union = a.sum(1)[:, None] + b.sum(1)[None, :] - intersection
                    local_values.append((intersection / union.clamp_min(1)).max(1).values.mean())
                totals["structural_coverage"] += float(torch.stack(local_values).mean()) if local_values else 1.0
            count += batch.num_graphs
    predicted = np.asarray(probabilities) >= 0.5
    metrics = {"positive_f1": float(f1_score(labels, predicted, zero_division=0)),
               "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
               "precision": float(precision_score(labels, predicted, zero_division=0)),
               "recall": float(recall_score(labels, predicted, zero_division=0)),
               "auroc": float(roc_auc_score(labels, probabilities)),
               "edge_retention": totals["kept"] / max(1, totals["candidates"]),
               "member_retention_within_selected": totals["actual_incidence"] / max(1, totals["selected_incidence"]),
               "anchor_coverage": totals["coverage"] / count,
               "actual_structural_coverage": totals["structural_coverage"] / count,
               "cached_graph_validation_seconds": time.perf_counter() - start, **totals}
    return metrics, predictions


def main():
    args = arguments()
    if args.epochs < 1 or not 1 <= args.fold <= 5 or not 0 < args.ratio <= 1:
        raise ValueError("Invalid experiment arguments")
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "training.log", encoding="utf-8")])
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("Server experiment requires CUDA")
    config = load_experiment_config(args.config)
    config = replace(config, data=replace(config.data, processed_path=args.data),
                     training=replace(config.training, device=str(device), batch_size=args.batch_size),
                     pruning=replace(config.pruning, budget_mode=args.budget_mode))
    dataset = PreprocessedSessionDataset(args.data, config.data.processed_sha256, config.data.expected_sessions, config.data.expected_nodes, config.data.expected_features)
    split = dataset.make_split(args.split_seed, config.split.test_fraction, 0)
    development = np.asarray(sorted(split.train))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(development, dataset.labels[development].numpy()))
    train_pos, validation_pos = folds[args.fold - 1]
    train_ids = subset(development[train_pos].tolist(), dataset.labels, args.train_limit, args.seed)
    validation_ids = subset(development[validation_pos].tolist(), dataset.labels, args.validation_limit, args.seed)
    assert not (set(train_ids) & set(validation_ids) or (set(train_ids) | set(validation_ids)) & set(split.test))
    manifest = {"purpose": "Exploratory single development-fold HSL ablation; final test not evaluated", "train": train_ids, "validation": validation_ids, "final_test_excluded": list(split.test), "fold": args.fold, "seed": args.seed, "split_seed": args.split_seed}
    (args.output / "split.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("DATA_VERIFIED train=%d validation=%d final_test_excluded=%d", len(train_ids), len(validation_ids), len(split.test))
    builder = HyperedgeBuilder(config.hypergraph)
    cache, stc_sizes = {}, []
    start = time.perf_counter()
    for offset, index in enumerate(sorted(set(train_ids + validation_ids))):
        graph = builder.build_batch(collate_sessions([dataset[index]]).to(device))
        cache[index] = replace(graph, incidence=graph.incidence.cpu(), edge_to_session=graph.edge_to_session.cpu(), type_ids=graph.type_ids.cpu(), structural_stats=graph.structural_stats.cpu())
        if index in train_ids:
            stc_sizes.extend(r.size for r in graph.records if r.behavior_type == BehaviorType.STC)
        if offset % 100 == 0:
            logging.info("CACHE_PROGRESS sessions=%d", offset + 1)
    build_seconds = time.perf_counter() - start
    threshold = max(16, int(np.quantile(stc_sizes, 0.75))) if stc_sizes else 16
    builder = CachedBuilder(cache)
    priors = class_priors(dataset, train_ids)
    means, global_mean = confidence_statistics(dataset, train_ids)
    set_seed(args.seed)
    model_class = ContextHSLBBPBeHM if args.family == "context" else HSLBBPBeHM
    initial_model = model_class(config, priors, means, global_mean, mode="bbp", large_edge_threshold=threshold)
    initial_state = {k: v.detach().clone() for k, v in initial_model.state_dict().items()}
    # 先验证插入扩展点后的关闭路径与原模型逐值一致。
    reference = BBPBeHM(config, priors, means, global_mean).to(device).eval()
    reference.load_state_dict({k: v for k, v in initial_state.items() if not k.startswith(("feedback_adapter.", "member_scorer.", "selector.context_scorer."))})
    initial_model = initial_model.to(device).eval()
    sample = collate_sessions([dataset[train_ids[0]]]).to(device)
    sample_graph = builder.build_batch(sample)
    with torch.no_grad():
        left = reference(sample, sample_graph, retention_ratio=args.ratio, stochastic_selection=False)
        right = initial_model(sample, sample_graph, retention_ratio=args.ratio, stochastic_selection=False)
        # CUDA 聚合可能有末位舍入差异；离散支持必须完全相同，logits 使用明确浮点容差。
        torch.testing.assert_close(left.selection.hard_mask, right.selection.hard_mask, atol=0, rtol=0)
        maximum_difference = float((left.logits-right.logits).abs().max())
        torch.testing.assert_close(left.logits, right.logits, atol=1e-6, rtol=1e-6)
        logging.info("BASELINE_EQUIVALENCE_MAX_DIFF %.10g", maximum_difference)
    del reference, initial_model
    logging.info("BASELINE_EQUIVALENCE_PASS large_edge_threshold=%d cache_seconds=%.2f", threshold, build_seconds)
    results = {"protocol": manifest["purpose"], "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "config": json.loads(json.dumps(asdict(config), default=str)), "candidate_build_seconds": build_seconds, "large_edge_threshold_from_train": threshold, "modes": {}}
    for mode in args.modes:
        set_seed(args.seed)
        model = model_class(config, priors, means, global_mean, mode=mode, large_edge_threshold=threshold).to(device)
        model.load_state_dict(initial_state)
        model._experiment_ratio = args.ratio
        train_loader = make_loader(dataset, train_ids, args.batch_size, True, args.seed)
        validation_loader = make_loader(dataset, validation_ids, args.batch_size, False, args.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        history, best = [], -1.0
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_start, total, sessions = time.perf_counter(), 0.0, 0
            for cpu_batch in train_loader:
                batch = cpu_batch.to(device)
                graph = builder.build_batch(batch)
                optimizer.zero_grad(set_to_none=True)
                # 所有组关闭额外 Gumbel 噪声，隔离显式反馈机制；与归档训练协议区分。
                output = model(batch, graph, retention_ratio=args.ratio, stochastic_selection=False)
                check_invariants(model, batch, graph, output)
                output.total_loss.backward()
                # 批量检查梯度有限性，仅进行一次设备同步，不改变梯度数值。
                finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
                if not finite:
                    raise RuntimeError("Non-finite gradient")
                optimizer.step()
                total += float(output.total_loss.detach()) * batch.num_graphs
                sessions += batch.num_graphs
            train_seconds = time.perf_counter() - epoch_start
            metrics, predictions = validate(model, validation_loader, builder, config, args.ratio, device, check=True)
            record = {"epoch": epoch, "train_loss": total / sessions, "train_seconds": train_seconds, "validation": metrics}
            history.append(record)
            # 保存每轮预测，以便固定轮次配对比较，避免只剩验证集选优结果。
            (args.output / f"{mode}_epoch_{epoch:02d}_predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
            logging.info("MODE=%s EPOCH=%d loss=%.6f f1=%.4f recall=%.4f auroc=%.4f edits=%d changes=%d train_seconds=%.2f", mode, epoch, record["train_loss"], metrics["positive_f1"], metrics["recall"], metrics["auroc"], metrics["edits"], metrics["changed"], train_seconds)
            if metrics["positive_f1"] > best:
                best = metrics["positive_f1"]
                torch.save({"model": model.state_dict(), "mode": mode, "epoch": epoch, "large_edge_threshold": threshold, "config": results["config"]}, args.output / (mode + "_best.pt"))
                (args.output / (mode + "_validation_predictions.json")).write_text(json.dumps(predictions, indent=2), encoding="utf-8")
            results["modes"][mode] = {"history": history, "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2, "best_epoch": max(history, key=lambda item: item["validation"]["positive_f1"])["epoch"]}
            (args.output / "result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "mode": mode, "epoch": args.epochs, "large_edge_threshold": threshold, "config": results["config"]}, args.output / (mode + "_last.pt"))
        del model, optimizer
        torch.cuda.empty_cache()
    logging.info("EXPERIMENT_COMPLETE final_test_evaluated=false")


if __name__ == "__main__":
    main()
