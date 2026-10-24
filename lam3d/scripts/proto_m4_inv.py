
import sys, torch, torch.nn.functional as F
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import proto_m4_lib as L
from lam3d.core import tga_covariance as T, rot_sync as RS
x0,xB,faces,q0,s0,names=L.load()
a,b,c=faces.unbind(-1)
e1=x0[b]-x0[a]; e2=x0[c]-x0[a]
n0=F.normalize(torch.cross(e1,e2,dim=-1),dim=-1,eps=1e-12)
E=torch.stack((e1,e2,n0),-1)                     # [F,3,3] f32
E64=E.double()
inv_ref=torch.linalg.inv(E64).float()
for nm,invf in [("repo inv3 (adj/det, floor 1e-12)",T.inv3),("torch.linalg.inv(f32)",torch.linalg.inv)]:
    try:
        iv=invf(E)
        rel=((iv-inv_ref).abs().amax(-1).amax(-1)/inv_ref.abs().amax(-1).amax(-1).clamp(min=1e-30))
        print("%-34s rel err p50=%.3e p99=%.3e max=%.3e"%(nm,rel.median(),torch.quantile(rel,.99),rel.max()))
    except Exception as ex:
        print(nm,"FAILED",ex)
print()
print("det(E) = 2*area:  p50=%.3e  min=%.3e"%(float((2*L.face_areas(x0,faces,1).reshape(-1)).median()),
      float((2*L.face_areas(x0,faces,1).reshape(-1)).min())))
