"""用途：在真实开发会话上验证类 HSL 与原鲁棒覆盖预算的前向、反向及结构约束兼容性。"""
import json
from pathlib import Path
from dataclasses import replace
import torch
from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset,collate_sessions
from bbp_behm.hsl import HSLBBPBeHM
from bbp_behm.hypergraph import HyperedgeBuilder
from scripts.train_hsl_pilot import check_invariants

def main():
    root=Path("outputs/hsl_pilot/fold1_seed42_fixed70_6ep_v2")
    config=load_experiment_config(Path("configs/vine_hsl_server.ini"))
    config=replace(config,pruning=replace(config.pruning,budget_mode="robust_adaptive"))
    # 哈希已由主试验核验；这里只取清单中的训练会话，不访问最终测试样本。
    dataset=PreprocessedSessionDataset(config.data.processed_path,expected_sessions=970,expected_nodes=68662,expected_features=768,verify_hash=False)
    ids=json.loads((root/"split.json").read_text())["train"][:4]
    batch=collate_sessions([dataset[i] for i in ids]).to(torch.device("cuda:0"))
    graph=HyperedgeBuilder(config.hypergraph).build_batch(batch)
    rows=[]
    for mode in ["bbp","feedback","feedback_edit"]:
        checkpoint=torch.load(root/(mode+"_best.pt"),map_location="cpu",weights_only=False)
        model=HSLBBPBeHM(config,mode=mode,large_edge_threshold=63).cuda().train()
        model.load_state_dict(checkpoint["model"])
        output=model(batch,graph,stochastic_selection=False)
        check_invariants(model,batch,graph,output)
        output.total_loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert min(output.selection.coverage_per_session.values())+1e-6>=config.pruning.coverage_tail_gamma
        rows.append({"mode":mode,"training_sessions":ids,"loss":float(output.total_loss.detach()),"minimum_anchor_tail_coverage":min(output.selection.coverage_per_session.values()),"kept":output.selection.kept_per_session,"edits":model.last_hsl_stats["edits"]})
    (root/"robust_compatibility.json").write_text(json.dumps({"purpose":"Functional compatibility only; not robust-budget performance evaluation","checks":rows},indent=2))
    print("ROBUST_COMPATIBILITY_PASS",json.dumps(rows))
if __name__=="__main__":
    main()
