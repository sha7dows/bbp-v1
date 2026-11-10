"""用途：仅在远端VINE开发折拟合分解阈值、核对结构约束并运行父边预算匹配的三组消融。"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset, collate_sessions
from bbp_behm.hypergraph import BehaviorType, HyperedgeBuilder
from bbp_behm.hsl_split import SplitBBPBeHM, SplitCachedBuilder, dispersion, eligible, fit_thresholds, propose_split
from bbp_behm.model import BBPBeHM
from bbp_behm.training import class_priors, confidence_statistics, make_loader, set_seed


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--modes", nargs="+", default=["none", "random", "semantic"], choices=["none", "random", "semantic"])
    return p.parse_args()


def prepare(dataset, config, train, validation, output, device):
    """缓存原候选，并用训练折的无标签统计拟合分解门槛。"""
    file = output / "prepared.pt"
    if file.exists():
        bundle = torch.load(file, weights_only=False)
        assert bundle["train"] == train and bundle["validation"] == validation
        return bundle
    builder = HyperedgeBuilder(config.hypergraph)
    cache, proposals, diagnostic, training_records = {}, {}, [], []
    start = time.perf_counter()
    for offset, index in enumerate(sorted(train + validation)):
        sample = dataset[index]
        graph = builder.build_batch(collate_sessions([sample]).to(device))
        graph = replace(graph, incidence=graph.incidence.cpu(), edge_to_session=graph.edge_to_session.cpu(),
                        type_ids=graph.type_ids.cpu(), structural_stats=graph.structural_stats.cpu())
        cache[index] = graph
        degree = {}
        for r in graph.records:
            if r.behavior_type != BehaviorType.STC:
                for g in r.members:
                    degree[g] = degree.get(g, 0) + 1
        proposals[index] = {}
        by_id = {int(g): i for i, g in enumerate(sample.global_node_indices.tolist())}
        for r in graph.records:
            if r.behavior_type != BehaviorType.STC:
                continue
            plan = propose_split(r, sample, degree) if r.size >= 16 else {
                "size": r.size, "dispersion": dispersion(sample.x[[by_id[g] for g in r.members]]), "gain": 0.0}
            if r.size >= 16:
                proposals[index][r.hyperedge_id] = plan
            entry = {k: plan[k] for k in ["size", "dispersion", "gain"]}
            entry.update(index=index, parent_id=r.hyperedge_id, partition="train" if index in train else "validation",
                         user_count=r.unique_user_count, time_span_days=r.time_span_days,
                         shared_member_count=sum(degree.get(g, 0) > 0 for g in r.members))
            diagnostic.append(entry)
            if index in train:
                training_records.append(entry)
        if offset % 100 == 0:
            logging.info("DIAGNOSTIC_PROGRESS sessions=%d", offset + 1)
    thresholds = fit_thresholds(training_records)
    plans = {i: {k: p for k, p in ps.items() if eligible(p, thresholds)} for i, ps in proposals.items()}
    # 每个父边的两种分解必须严格匹配规模、锚点、成员并集和额外关联。
    checked = 0
    for index, ps in plans.items():
        for parent, p in ps.items():
            a, b = p["semantic"], p["random"]
            original = set(p["members"])
            for groups in (a, b):
                assert len(groups) == 2 and min(map(len, groups)) >= 2
                assert set(groups[0]) | set(groups[1]) == original
                assert set(groups[0]) & set(groups[1]) == set(p["anchors"])
                assert sum(map(len, groups)) - len(original) <= int(len(original) * 0.1)
            assert list(map(len, a)) == list(map(len, b))
            checked += 1
    summaries = {}
    for partition, indices in [("train", train), ("validation", validation)]:
        ds = [d for d in diagnostic if d["partition"] == partition]
        accepted = [p for i in indices for p in plans[i].values()]
        large = [d for d in ds if d["size"] >= thresholds["size"]]
        summaries[partition] = {
            "sessions": len(indices), "stc_edges": len(ds), "large_edges": len(large),
            "split_parents": len(accepted), "affected_sessions": sum(bool(plans[i]) for i in indices),
            "size_quantiles": np.quantile([d["size"] for d in ds], [0,.25,.5,.75,.9,1]).tolist(),
            "dispersion_quantiles": np.quantile([d["dispersion"] for d in ds], [0,.25,.5,.75,.9,1]).tolist(),
            "large_dispersion_quantiles": np.quantile([d["dispersion"] for d in large], [0,.25,.5,.75,.9,1]).tolist(),
            "accepted_gain_mean": float(np.mean([p["gain"] for p in accepted])) if accepted else 0,
            "extra_candidate_edges": len(accepted),
            "extra_candidate_incidence": sum(len(p["anchors"]) for p in accepted),
            "candidate_incidence": sum(float(cache[i].incidence.sum()) for i in indices),
            "candidate_edges": sum(cache[i].num_edges for i in indices),
        }
    bundle = dict(train=train, validation=validation, cache=cache, plans=plans, thresholds=thresholds)
    torch.save(bundle, file)
    dump(output / "diagnostic.json", dict(thresholds=thresholds, summaries=summaries,
        structure_checks=checked, preprocessing_seconds=time.perf_counter()-start, records=diagnostic))
    dump(output / "provenance.json", {i: plans[i] for i in sorted(plans) if plans[i]})
    logging.info("DIAGNOSTIC_COMPLETE thresholds=%s summary=%s", thresholds, summaries)
    return bundle


def check_output(model, graph, output, ratio=.7):
    selection = output.selection
    for session in range(int(graph.edge_to_session.max()) + 1 if graph.num_edges else 0):
        count = int((graph.edge_to_session == session).sum())
        assert selection.kept_per_session[session] == max(1, int(np.ceil(count * ratio))) if count else selection.kept_per_session[session] == 0
    actual, prop = output.masked_incidence.detach(), model.last_propagation_graph
    rows = model.last_parent_rows
    selected_parent = graph.incidence * selection.hard_mask[:, None]
    assert actual.shape == prop.incidence.shape and torch.isfinite(output.total_loss)
    assert torch.all((actual == 0) | (actual == 1))
    assert torch.all(actual <= selected_parent[rows])
    non_stc = prop.type_ids != int(BehaviorType.STC)
    assert torch.equal(actual[non_stc], selected_parent[rows[non_stc]])
    assert torch.equal(actual.sum(0) > 0, selected_parent.sum(0) > 0)
    # 将子边按父边合并后，必须逐父边恢复完整的原成员与选择掩码。
    merged = actual.new_zeros(graph.incidence.shape)
    merged.index_add_(0, rows, actual)
    assert torch.equal(merged > 0, selected_parent > 0)
    assert torch.all(actual.sum(1) <= graph.incidence[rows].sum(1))
    original_count = float(selected_parent.sum())
    assert float(actual.sum()) <= original_count * 1.1 + 1e-4


def evaluate(model, loader, builder, device):
    model.eval()
    predictions = []
    totals = {k: 0.0 for k in ["parent_candidates", "kept_parents", "propagation_edges", "parent_incidence",
        "propagation_incidence", "split_kept_parents", "anchor_coverage", "actual_jaccard_coverage", "propagation_max_size"]}
    sessions = 0
    sizes, parent_sizes = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            graph = builder.build_batch(batch)
            out = model(batch, graph, retention_ratio=.7, stochastic_selection=False)
            check_output(model, graph, out)
            for index, label, prob in zip(batch.session_indices.tolist(), batch.labels.tolist(), out.logits.softmax(1)[:,1].tolist()):
                predictions.append(dict(index=index, label=label, positive_probability=prob))
            hard = out.selection.hard_mask.bool()
            parent_h = graph.incidence[hard]
            propagation_h = out.masked_incidence.detach()
            nonempty = propagation_h.sum(1) > 0
            totals["parent_candidates"] += graph.num_edges
            totals["kept_parents"] += int(hard.sum())
            totals["propagation_edges"] += int(nonempty.sum())
            totals["parent_incidence"] += float(parent_h.sum())
            totals["propagation_incidence"] += float(propagation_h.sum())
            totals["split_kept_parents"] += int(nonempty.sum()) - int(hard.sum())
            totals["anchor_coverage"] += sum(out.selection.coverage_per_session.values())
            # 父边覆盖与真实子边覆盖分开统计，避免把父边重构守恒等同于传播等价。
            for session in range(batch.num_graphs):
                by_type = []
                for kind in torch.unique(graph.type_ids[graph.edge_to_session == session]):
                    parent_rows = (graph.edge_to_session == session) & (graph.type_ids == kind)
                    prop = model.last_propagation_graph
                    child_rows = (prop.edge_to_session == session) & (prop.type_ids == kind)
                    a, b = graph.incidence[parent_rows], propagation_h[child_rows]
                    intersection = a @ b.t()
                    union = a.sum(1)[:,None] + b.sum(1)[None,:] - intersection
                    by_type.append((intersection / union.clamp_min(1)).max(1).values.mean())
                totals["actual_jaccard_coverage"] += float(torch.stack(by_type).mean()) if by_type else 1.
            sizes.extend(propagation_h[nonempty].sum(1).tolist())
            parent_sizes.extend(parent_h.sum(1).tolist())
            sessions += batch.num_graphs
    y = np.array([p["label"] for p in predictions])
    probs = np.array([p["positive_probability"] for p in predictions])
    pred = probs >= .5
    totals["propagation_max_size"] = max(sizes, default=0)
    metrics = dict(positive_f1=float(f1_score(y,pred,zero_division=0)), macro_f1=float(f1_score(y,pred,average="macro")),
        precision=float(precision_score(y,pred,zero_division=0)), recall=float(recall_score(y,pred,zero_division=0)),
        auroc=float(roc_auc_score(y,probs)), parent_size_mean=float(np.mean(parent_sizes)),
        propagation_size_mean=float(np.mean(sizes)), extra_incidence_fraction=(totals["propagation_incidence"] /
        max(1,totals["parent_incidence"])-1), parent_anchor_coverage=totals["anchor_coverage"]/sessions,
        actual_structural_coverage=totals["actual_jaccard_coverage"]/sessions,
        validation_seconds=time.perf_counter()-start, **totals)
    return metrics, predictions


def main():
    args = arguments()
    assert 1 <= args.fold <= 5 and args.epochs > 0
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output/"training.log",encoding="utf-8")])
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "Server GPU required"
    device = torch.device("cuda:0")
    config = load_experiment_config(Path("configs/vine_hsl_server.ini"))
    config = replace(config, training=replace(config.training, batch_size=8,device=str(device)),
                     pruning=replace(config.pruning,budget_mode="fixed"))
    dataset = PreprocessedSessionDataset(config.data.processed_path,config.data.processed_sha256,
        config.data.expected_sessions,config.data.expected_nodes,config.data.expected_features)
    split = dataset.make_split(42, config.split.test_fraction, 0)
    development = np.array(sorted(split.train))
    folds = list(StratifiedKFold(5,shuffle=True,random_state=42).split(development,dataset.labels[development].numpy()))
    tp, vp = folds[args.fold-1]
    train, validation = development[tp].tolist(), development[vp].tolist()
    assert not ((set(train)|set(validation)) & set(split.test))
    dump(args.output/"split.json",dict(train=train,validation=validation,final_test_excluded=list(split.test),
        fold=args.fold,seed=args.seed,split_seed=42))
    bundle = prepare(dataset,config,train,validation,args.output,device)
    if args.prepare_only:
        return
    builder = SplitCachedBuilder(bundle["cache"],bundle["plans"],dataset)
    priors = class_priors(dataset,train)
    means, global_mean = confidence_statistics(dataset,train)
    set_seed(args.seed)
    initial = SplitBBPBeHM(config,priors,means,global_mean,mode="none")
    initial_state = {k:v.clone() for k,v in initial.state_dict().items()}
    reference = BBPBeHM(config,priors,means,global_mean).to(device).eval()
    reference.load_state_dict(initial_state)
    initial = initial.to(device).eval()
    batch = collate_sessions([dataset[train[0]]]).to(device)
    graph = builder.build_batch(batch)
    with torch.no_grad():
        a = reference(batch,graph,retention_ratio=.7,stochastic_selection=False)
        b = initial(batch,graph,retention_ratio=.7,stochastic_selection=False)
        torch.testing.assert_close(a.selection.hard_mask,b.selection.hard_mask,atol=0,rtol=0)
        torch.testing.assert_close(a.logits,b.logits,atol=1e-6,rtol=1e-6)
    logging.info("DEFAULT_EQUIVALENCE_PASS max_diff=%.9g",float((a.logits-b.logits).abs().max()))
    del reference, initial
    result = dict(arguments=vars(args),thresholds=bundle["thresholds"],config=asdict(config),
        final_test_evaluated=False,modes={})
    for mode in args.modes:
        builder.mode = mode
        set_seed(args.seed)
        model = SplitBBPBeHM(config,priors,means,global_mean,mode=mode).to(device)
        model.load_state_dict(initial_state)
        train_loader = make_loader(dataset,train,8,True,args.seed)
        validation_loader = make_loader(dataset,validation,8,False,args.seed)
        optimizer = torch.optim.Adam(model.parameters(),lr=config.training.learning_rate,weight_decay=config.training.weight_decay)
        history = []
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,args.epochs+1):
            model.train()
            start, loss, count = time.perf_counter(), 0.0, 0
            for cpu_batch in train_loader:
                batch = cpu_batch.to(device)
                graph = builder.build_batch(batch)
                optimizer.zero_grad(set_to_none=True)
                out = model(batch,graph,retention_ratio=.7,stochastic_selection=False)
                check_output(model,graph,out)
                out.total_loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                assert bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()), "Non-finite gradient"
                optimizer.step()
                loss += float(out.total_loss.detach())*batch.num_graphs
                count += batch.num_graphs
            train_seconds = time.perf_counter()-start
            metrics,predictions = evaluate(model,validation_loader,builder,device)
            history.append(dict(epoch=epoch,train_loss=loss/count,train_seconds=train_seconds,validation=metrics))
            dump(args.output/f"{mode}_epoch_{epoch:02d}_predictions.json",predictions)
            result["modes"][mode] = dict(history=history,peak_memory_mb=torch.cuda.max_memory_allocated()/1024**2)
            dump(args.output/"result.json",result)
            logging.info("MODE=%s EPOCH=%d loss=%.6f f1=%.6f auroc=%.6f parent_kept=%d split_kept=%d extra_incidence=%.5f train_seconds=%.2f",
                mode,epoch,loss/count,metrics["positive_f1"],metrics["auroc"],metrics["kept_parents"],
                metrics["split_kept_parents"],metrics["extra_incidence_fraction"],train_seconds)
        torch.save(dict(model=model.state_dict(),mode=mode,epoch=args.epochs,thresholds=bundle["thresholds"],config=result["config"]),
                   args.output/f"{mode}_last.pt")
        del model,optimizer
        torch.cuda.empty_cache()
    logging.info("EXPERIMENT_COMPLETE final_test_evaluated=false")


if __name__ == "__main__":
    main()
