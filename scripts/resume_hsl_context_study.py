"""用途：保留已完成筛查，仅在等价检查浮点容差修正后恢复预定的第二折双种子复核。"""
import json
from pathlib import Path
from scripts.run_hsl_context_study import OUT,PLAN,run,write_status

def main():
    screen=json.loads((OUT/"screen_fold1_seed42/result.json").read_text())
    split=json.loads((OUT/"screen_fold1_seed42/split.json").read_text())
    assert all(len(v["history"])==8 for v in screen["modes"].values())
    promotion=json.loads((OUT/"promotion.json").read_text())
    def key(mode):
        history=screen["modes"][mode]["history"][-3:]
        return (sum(h["validation"]["positive_f1"] for h in history)/3,sum(h["validation"]["auroc"] for h in history)/3)
    promoted=max(["node_detach","edge_context"],key=key)
    control="node_control" if promoted=="node_detach" else "edge_control"
    assert promoted==promotion["promoted"] and control==promotion["control"]
    confirmations={}
    try:
        for seed in [42,43]:
            result,manifest=run(f"confirm_fold2_seed{seed}",2,seed,12,["bbp",control,promoted])
            assert manifest["final_test_excluded"]==split["final_test_excluded"]
            if confirmations:
                previous=next(iter(confirmations.values()))["split"]
                assert manifest["train"]==previous["train"] and manifest["validation"]==previous["validation"]
            confirmations[str(seed)]={"split":manifest,"modes":result["modes"]}
        (OUT/"study.json").write_text(json.dumps({"plan":PLAN,"promotion":promoted,"control":control,"screen":screen,"confirmations":confirmations,"numeric_guard_fix":True},indent=2))
        write_status(phase="complete",promoted=promoted,control=control,final_test_evaluated=False)
        print("CONTEXT_STUDY_COMPLETE",flush=True)
    except Exception as exc:
        write_status(phase="failed",error=str(exc))
        raise
if __name__=="__main__":
    main()
