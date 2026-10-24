
# -*- coding: utf-8 -*-
"""方法四候选离线评测（真实 rot_capture 张量）。输出 /tmp/m4_eval.json。"""
import sys, os, json, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/LAM"); sys.path.insert(0,"/workspace/lam3d/scripts")
os.environ.setdefault("LAM_ROOT","/workspace/LAM")
from lam3d.core.rot_sync import polar_rotation
from lam3d.core.tga_covariance import quat_to_matrix, matrix_to_quat
import proto_m4_lib as L
torch.set_grad_enabled(False)
dev="cuda"

x0,xB,faces,q0,s0,names=L.load()
M,N=xB.shape[0],x0.shape[0]; Fn=faces.shape[0]
R0=quat_to_matrix(q0); D0=s0**2; lam0=torch.sort(D0,-1).values
area=L.face_areas(x0,faces,M)
Jf,n1f=L.face_jacobians(x0,xB,faces)          # n1f = normalized deformed face normal
Rf=polar_rotation(Jf)
Jv9,den=L.scatter_vertex(Jf,area,faces,N); Jv=Jv9.view(M,N,3,3)
q_face=matrix_to_quat(Rf.reshape(-1,3,3)).reshape(M,Fn,4)
qR,conf=L.markley(q_face,area,faces,N)

def vnorm(per_face, w):
    s=torch.zeros(M,N,3,device=dev)
    for k in range(3): s.index_add_(1,faces[:,k],per_face*w[:,:,None])
    return s
nv=F.normalize(vnorm(n1f,area),dim=-1,eps=1e-12)          # 形变后协变法向
n0f=F.normalize(torch.cross(x0[faces[:,1]]-x0[faces[:,0]],x0[faces[:,2]]-x0[faces[:,0]],dim=-1),dim=-1,eps=1e-12).unsqueeze(0).expand(M,-1,-1).contiguous()
n0v=F.normalize(vnorm(n0f,area),dim=-1,eps=1e-12)          # 静止法向

Af2=0.5*torch.cross(xB[...,faces[:,1],:]-xB[...,faces[:,0],:],xB[...,faces[:,2],:]-xB[...,faces[:,0],:],dim=-1).norm(dim=-1)
detJ=torch.linalg.det(Jf.float())
# --- 关键恒等式核查 ---
Jinv=torch.linalg.pinv(Jf.double(), rtol=1e-6, atol=0.0)
nt_Jinv=(Jinv.transpose(-1,-2)@n0f.double().unsqueeze(-1)).squeeze(-1).norm(dim=-1)   # |J^{-T} n0|
k_nanson=(detJ.double().abs()*area.double())/Af2.double().clamp(min=1e-30)            # 文档 S2 的 k_f
print("[id] |detJ| vs A'/A  max rel dev = %.3e" % (((detJ.double().abs()*area.double()/Af2.double().clamp(min=1e-30))-1).abs().max().item()))
print("[id] |J^-T n0|  q01/50/99 = %s" % [round(v,4) for v in torch.quantile(nt_Jinv.flatten(),torch.tensor([.01,.5,.99],device=dev)).tolist()])
print("[id] k_nanson   q01/50/99 = %s" % [round(v,4) for v in torch.quantile(k_nanson.flatten(),torch.tensor([.01,.5,.99],device=dev)).tolist()])
print("[id] area ratio A'/A q50/90/99 = %s" % [round(v,3) for v in torch.quantile((Af2/area.clamp(min=1e-20)).flatten(),torch.tensor([.5,.9,.99],device=dev)).tolist()])

logr=torch.log((Af2/area.clamp(min=1e-20)).clamp(min=1e-3,max=1e3))
lk=torch.zeros(M,N,device=dev); ws=torch.zeros(M,N,device=dev)
for k in range(3):
    lk.index_add_(1,faces[:,k],logr*area); ws.index_add_(1,faces[:,k],area)
lk=lk/ws.clamp(min=1e-12)
print("[id] per-vertex log(A'/A) q01/50/99 = %s" % [round(v,4) for v in torch.quantile(lk.flatten(),torch.tensor([.01,.5,.99],device=dev)).tolist()])
json.dump({"Jinv_n0_quant":[float(v) for v in torch.quantile(nt_Jinv.flatten(),torch.tensor([.01,.5,.99,.999],device=dev))],
           "k_nanson_quant":[float(v) for v in torch.quantile(k_nanson.flatten(),torch.tensor([.01,.5,.99,.999],device=dev))]},
          open("/tmp/m4_id.json","w"))
torch.save({"lk":lk.cpu(),"nv":nv.cpu(),"n0v":n0v.cpu(),"qR":qR.cpu(),"conf":conf.cpu(),"Jv":Jv.cpu()}, "/tmp/m4_pre.pt")
print("done")
