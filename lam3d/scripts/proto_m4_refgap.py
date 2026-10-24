
import sys, math, torch, numpy as np
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch.nn.functional as F
exec(open("/workspace/lam3d/scripts/proto_m4_final.py").read().split("VAR={")[0])
# Markley 聚合 R_agg 与一环边 Kabsch R_ref 的夹角（ones preset 单独看）
def ang(A,B):
    R=A.transpose(-1,-2)@B
    tr=R[...,0,0]+R[...,1,1]+R[...,2,2]
    return torch.acos(((tr-1)*.5).clamp(-1,1))*180/math.pi
Rk=quat_to_matrix(qR).double()
a=ang(rref.double(),Rk)
print("per-preset  angle(Kabsch_ref, Markley_agg) p50/p90:")
for k,nm in enumerate(names):
    v=a[k]; print("  %-12s %6.3f %6.3f"%(nm,v.median(),torch.quantile(v,.9)))
print("mean over presets: p50=%.3f p90=%.3f"%(a.median(1).values.mean(),torch.quantile(a,.9,1).mean()))
