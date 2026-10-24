
# -*- coding: utf-8 -*-
"""方法四（RSB）候选的原型评测：在真实 rot_capture 张量上复算
   rot-only / TGA / RSB(alpha sweep) 的 shape / rigid / false-move / 越界 / 薄轴 / 一环抖动。
"""
import sys, os, json, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/LAM"); sys.path.insert(0,"/workspace/lam3d/scripts")
os.environ.setdefault("LAM_ROOT","/workspace/LAM")
from lam3d.core.rot_sync import polar_rotation
from lam3d.core.tga_covariance import quat_to_matrix, matrix_to_quat
import proto_m4_lib as L
torch.set_grad_enabled(False)
dev="cuda"; F32=torch.float32

x0,xB,faces,q0,s0,names=L.load()
M,N=xB.shape[0],x0.shape[0]; Fn=faces.shape[0]
R0=quat_to_matrix(q0).double(); D0=(s0.double()**2); lam0=torch.sort(D0,-1).values
area=L.face_areas(x0,faces,M)
Jf,n1f=L.face_jacobians(x0,xB,faces)
Rf=polar_rotation(Jf)
Jv9,_=L.scatter_vertex(Jf,area,faces,N); Jv=Jv9.view(M,N,3,3)
q_face=matrix_to_quat(Rf.reshape(-1,3,3)).reshape(M,Fn,4)
qR,conf=L.markley(q_face,area,faces,N)       # [M,N,4], r_v

def vsum(per_face,w):
    s=torch.zeros(M,N,3,device=dev)
    for k in range(3): s.index_add_(1,faces[:,k],per_face*w[:,:,None])
    return s
nv=F.normalize(vsum(n1f,area),dim=-1,eps=1e-12)
n0f=F.normalize(torch.cross(x0[faces[:,1]]-x0[faces[:,0]],x0[faces[:,2]]-x0[faces[:,0]],dim=-1),dim=-1,eps=1e-12).unsqueeze(0).expand(M,-1,-1).contiguous()
n0v=F.normalize(vsum(n0f,area),dim=-1,eps=1e-12)

Af2=0.5*torch.cross(xB[...,faces[:,1],:]-xB[...,faces[:,0],:],xB[...,faces[:,2],:]-xB[...,faces[:,0],:],dim=-1).norm(dim=-1)
logr=torch.log((Af2/area.clamp(min=1e-20)).clamp(min=1e-3,max=1e3))   # = log|detJ| for our data
lk=torch.zeros(M,N,device=dev); ws=torch.zeros(M,N,device=dev)
for k in range(3):
    lk.index_add_(1,faces[:,k],logr*area); ws.index_add_(1,faces[:,k],area)
lk=lk/ws.clamp(min=1e-12)

# ---------- coherence r_v 直方图 ----------
def q_(t,ps): 
    t=t.flatten().float(); return [round(float(v),4) for v in torch.quantile(t, torch.tensor(ps,device=dev))]
print("[coh] r_v quantiles (.1,1,5,50 pct):", q_(conf,[.001,.01,.05,.5]))
for i,nm in enumerate(names):
    c=conf[i]; print("  %-11s r_v p0.1=%.3f p1=%.3f p5=%.3f  frac<0.3=%.4f frac<0.5=%.4f frac<0.9=%.4f"%(
        nm,q_(c,[.001])[0],q_(c,[.01])[0],q_(c,[.05])[0],
        (c<0.3).float().mean().item(),(c<0.5).float().mean().item(),(c<0.9).float().mean().item()))
print("[thick] per-vertex log(A'/A) q50/90/99/max:", q_(lk,[.5,.9,.99]), round(float(lk.max()),3))
print("[thick] frac |lk|>log1.35: %.4f  >log2: %.4f  >log3: %.4f"%( (lk.abs()>np.log(1.35)).float().mean().item(),
      (lk.abs()>np.log(2)).float().mean().item(), (lk.abs()>np.log(3)).float().mean().item()))
print("[aniso] s0 ratio p50/90/99:", q_((s0.max(-1).values/s0.min(-1).values.clamp(min=1e-9)),[.5,.9,.99]))
print("[det] detJ<=0 frac: %.5f  detJq .5/99: %s"%( (torch.linalg.det(Jf.float())<=0).float().mean().item(), q_(torch.linalg.det(Jf.float()),[.5,.99])))
i3=s0.argmin(-1); aniso=(s0.max(-1).values/s0.min(-1).values.clamp(min=1e-9))
np.save("/tmp/_coh.npy", conf.cpu().numpy()); np.save("/tmp/_lk.npy", lk.cpu().numpy())
json.dump({"coh_all":q_(conf,[.001,.01,.05,.5,.99]),
           "coh_per":{nm:{"p0.1":q_(conf[i],[.001])[0],"p1":q_(conf[i],[.01])[0],"p5":q_(conf[i],[.05])[0],
                          "f03":float((conf[i]<0.3).float().mean()),"f05":float((conf[i]<0.5).float().mean()),
                          "f09":float((conf[i]<0.9).float().mean())} for i,nm in enumerate(names)},
           "lk":q_(lk,[.01,.5,.9,.99]),"aniso":q_(aniso,[.5,.9,.99])}, open("/tmp/m4_stats.json","w"))
torch.save({"qR":qR.cpu(),"conf":conf.cpu(),"lk":lk.cpu(),"nv":nv.cpu(),"n0v":n0v.cpu(),"Jv":Jv.cpu()}, "/tmp/m4_pre.pt")
print("saved /tmp/m4_pre.pt")
