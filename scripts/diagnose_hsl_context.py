"""用途：仅用训练会话诊断同预算同类型的选边替换敏感度，并记录评分梯度和上下文残差。"""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset,collate_sessions
from bbp_behm.hypergraph import HyperedgeBuilder
from bbp_behm.hsl_context import ContextHSLBBPBeHM

def main():
    torch.set_num_threads(4)
    root=Path("outputs/hsl_context_20261109")
    promoted=json.loads((root/"promotion.json").read_text())["promoted"]
    folder=root/"screen_fold1_seed42"
    split=json.loads((folder/"split.json").read_text())
    config=load_experiment_config(Path("configs/vine_hsl_server.ini"))
    config=replace(config,pruning=replace(config.pruning,budget_mode="fixed"))
    data=PreprocessedSessionDataset(config.data.processed_path,expected_sessions=970,expected_nodes=68662,expected_features=768,verify_hash=False)
    ids=[]
    for label in [0,1]:
        ids.extend([i for i in split["train"] if int(data.labels[i])==label][:12])
    assert not set(ids)&set(split["final_test_excluded"])
    builder=HyperedgeBuilder(config.hypergraph)
    results={}
    for mode in dict.fromkeys(["bbp",promoted,"edge_context"]):
        checkpoint=torch.load(folder/(mode+"_last.pt"),map_location="cpu",weights_only=False)
        model=ContextHSLBBPBeHM(config,mode=mode).cuda().eval()
        model.load_state_dict(checkpoint["model"])
        random=torch.Generator(device="cuda").manual_seed(20261109)
        records=[]; score_grad=[]; delta_magnitude=[]
        for start in range(0,len(ids),4):
            current=ids[start:start+4]
            batch=collate_sessions([data[i] for i in current]).to(torch.device("cuda"))
            graph=builder.build_batch(batch)
            model.zero_grad(set_to_none=True)
            output=model(batch,graph,retention_ratio=0.7,stochastic_selection=False)
            if output.selection.scores.requires_grad:
                output.selection.scores.retain_grad()
            probabilities=output.logits.softmax(1)[:,1].detach()
            output.total_loss.backward()
            gradient=output.selection.scores.grad
            score_grad.append(float(gradient.abs().mean()) if gradient is not None else 0.0)
            if model.selector.last_delta is not None:
                delta_magnitude.append(float(model.selector.last_delta.detach().abs().mean()))
            hard=output.selection.hard_mask.detach()
            variants=[]
            with torch.no_grad():
                identity=model(batch,graph,hard_mask_override=hard)
                torch.testing.assert_close(identity.logits,output.logits.detach(),rtol=1e-6,atol=1e-6)
                for repeat in range(8):
                    mask=torch.zeros_like(hard)
                    for session in range(batch.num_graphs):
                        for kind in range(3):
                            positions=torch.nonzero((graph.edge_to_session==session)&(graph.type_ids==kind)).flatten()
                            keep=int(hard[positions].sum())
                            if keep:
                                scores=torch.rand(len(positions),device="cuda",generator=random)
                                mask[positions[torch.topk(scores,keep).indices]]=1
                    assert float(mask.sum())==float(hard.sum())
                    alternate=model(batch,graph,hard_mask_override=mask)
                    variants.append(alternate.logits.softmax(1)[:,1].cpu().numpy())
            values=np.stack(variants)
            original=probabilities.cpu().numpy()
            for offset,index in enumerate(current):
                records.append({"index":index,"label":int(batch.labels[offset]),"selected_probability":float(original[offset]),"random_probability_std":float(values[:,offset].std()),"mean_absolute_probability_change":float(np.abs(values[:,offset]-original[offset]).mean()),"decision_flip_fraction":float(((values[:,offset]>=0.5)!=(original[offset]>=0.5)).mean())})
        results[mode]={"purpose":"Training-only balanced-case sensitivity diagnostic, not validation performance","cases":records,"mean_absolute_probability_change":float(np.mean([r["mean_absolute_probability_change"] for r in records])),"mean_random_probability_std":float(np.mean([r["random_probability_std"] for r in records])),"mean_decision_flip_fraction":float(np.mean([r["decision_flip_fraction"] for r in records])),"mean_score_gradient_abs":float(np.mean(score_grad)),"mean_context_delta_abs":float(np.mean(delta_magnitude)) if delta_magnitude else None}
    (root/"sensitivity.json").write_text(json.dumps(results,indent=2))
    print(json.dumps({m:{k:v for k,v in r.items() if k!="cases"} for m,r in results.items()}))
if __name__=="__main__":
    main()
