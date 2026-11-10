"""用途：按预先记录的规则顺序执行上下文筛查与独立开发折复核，固定最终留出并保留所有尝试。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path("/root/TRY")
OUT=ROOT/"outputs/hsl_context_20261109"
PLAN={
    "purpose":"Exploratory context screening and second-development-fold confirmation; final test never evaluated",
    "split_seed":42,"fold_seed":42,"ratio":0.7,"batch_size":8,
    "screen":{"fold":1,"seed":42,"epochs":8,"modes":["bbp","node_control","node_detach","edge_control","edge_context"]},
    "promotion":"Choose node_detach or edge_context by mean positive F1 over last three screening epochs; tie-break by mean AUROC. Preserve all screening results.",
    "confirmation":{"fold":2,"seeds":[42,43],"epochs":12,"modes":"bbp + matched control + promoted candidate"},
    "limitations":["Final test excluded","Training seeds independent from fixed data split","Fold 2 reused for two training seeds, not two independent datasets","Fixed K, not the archived robust_adaptive performance protocol","No member editing in this round"]
}

def write_status(**kwargs):
    (OUT/"status.json").write_text(json.dumps(kwargs,indent=2),encoding="utf-8")

def run(name,fold,seed,epochs,modes):
    folder=OUT/name
    folder.mkdir(parents=True,exist_ok=True)
    files=sorted(str(p.relative_to(ROOT)) for p in (ROOT/"bbp_behm").glob("*.py"))
    files+=["scripts/train_hsl_pilot.py","scripts/check_hsl_context.py","scripts/run_hsl_context_study.py","configs/vine_hsl_server.ini"]
    files += ["scripts/resume_hsl_context_study.py"]
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    (folder/"source_manifest.json").write_text(json.dumps(hashes,indent=2))
    subprocess.run(["tar","-czf",str(folder/"source.tgz"),*files],cwd=ROOT,check=True)
    args=[sys.executable,"-u","-m","scripts.train_hsl_pilot","--family","context","--config","configs/vine_hsl_server.ini","--data","/root/TRY/data/vine/KbertConfidence/processed/KbertConfidence_data_bert.pt","--output",str(folder),"--epochs",str(epochs),"--batch-size","8","--ratio","0.7","--fold",str(fold),"--seed",str(seed),"--split-seed","42","--modes",*modes]
    with (folder/"console.log").open("w") as log:
        child=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        write_status(phase=name,pid=child.pid,modes=modes,epochs=epochs)
        code=child.wait()
    if code:
        raise RuntimeError(f"{name} exited {code}")
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in hashes.items()),"Sources changed during study"
    result=json.loads((folder/"result.json").read_text())
    assert all(len(result["modes"][m]["history"])==epochs for m in modes)
    manifest=json.loads((folder/"split.json").read_text())
    for epoch in range(epochs):
        assert len({result["modes"][m]["history"][epoch]["validation"]["kept"] for m in modes})==1
    return result,manifest

def main():
    os.environ["TZ"]="Asia/Shanghai"
    time.tzset()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"plan.json").write_text(json.dumps(PLAN,indent=2))
    try:
        screen,split=run("screen_fold1_seed42",1,42,8,PLAN["screen"]["modes"])
        def key(mode):
            history=screen["modes"][mode]["history"][-3:]
            return (sum(h["validation"]["positive_f1"] for h in history)/3,sum(h["validation"]["auroc"] for h in history)/3)
        promoted=max(["node_detach","edge_context"],key=key)
        control="node_control" if promoted=="node_detach" else "edge_control"
        (OUT/"promotion.json").write_text(json.dumps({"promoted":promoted,"control":control,"screen_scores":{m:key(m) for m in ["node_detach","edge_context"]},"rule":PLAN["promotion"]},indent=2))
        print("PROMOTED",promoted,flush=True)
        confirmations={}
        for seed in [42,43]:
            result,manifest=run(f"confirm_fold2_seed{seed}",2,seed,12,["bbp",control,promoted])
            assert manifest["final_test_excluded"]==split["final_test_excluded"]
            if confirmations:
                prior=next(iter(confirmations.values()))["split"]
                assert manifest["train"]==prior["train"] and manifest["validation"]==prior["validation"]
            confirmations[str(seed)]={"split":manifest,"modes":result["modes"]}
        (OUT/"study.json").write_text(json.dumps({"plan":PLAN,"promotion":promoted,"control":control,"screen":screen,"confirmations":confirmations},indent=2))
        write_status(phase="complete",promoted=promoted,control=control,final_test_evaluated=False)
        print("CONTEXT_STUDY_COMPLETE",flush=True)
    except Exception as exc:
        write_status(phase="failed",error=str(exc))
        raise

if __name__=="__main__":
    main()
