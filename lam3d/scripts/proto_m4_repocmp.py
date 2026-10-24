
# -*- coding: utf-8 -*-
"""决定性对照：仓库 RotSyncModulator 的 delta_q vs 原型 proto 的 qR（同一 capture、同一度量）。"""
import sys, math, json
sys.path.insert(0, "/workspace/lam3d"); sys.path.insert(0, "/workspace/LAM"); sys.path.insert(0,"/workspace/lam3d/scripts")
import numpy as np, torch
import torch.nn.functional as F
import proto_m4_lib as L
from lam3d.core import cov_switch
from lam3d.core.pipeline import SingleImage3DPipeline
from lam3d.core.tga_covariance import quat_to_matrix

DEV="cuda"
x0,xB,faces,q0,s0,names=L.load()
Mu=xB.shape[0]; N=x0.shape[0]

torch.set_grad_enabled(True)
pipe=SingleImage3DPipeline(device=DEV, lazy_load=True)
pipe.run(image_path="/workspace/LAM/assets/sample_input/messi.png", job_dir="/tmp/offline_job2", job_id="cmp")
torch.set_grad_enabled(False)
rend=pipe.arkit_session["renderer"]
mod=cov_switch._make("rot_sync", rend, {})
print("mod", type(mod).__name__, "average", mod.average, "max_angle", mod.max_angle_deg, "cohgate", mod.coherence_gate)
fup=mod.faces.long().to(DEV)
print("faces_up", tuple(fup.shape), "capture faces", tuple(faces.shape), "equal:", bool(torch.equal(fup.cpu(), faces.cpu())))
print("faces_up max", int(fup.max()), "N captured", N, "num_gaussians", mod.num_gaussians)

with torch.no_grad():
    dq_r, _pl = mod.delta_rotation_scale(x0.unsqueeze(0).expand(Mu,-1,-1), xB)
    q1_r, s1_r = mod.apply(dq_r, mod.delta_scale(None, s0), q0, s0)
pl = mod._last_diag
print("diag keys", list(pl.keys()))
for k in ("angle_deg","clipped","gated","coherence"):
    if k in pl:
        v=pl[k]
        print("  %-10s mean %.5f max %.5f"%(k, float(v.float().mean()), float(v.float().max())))

# 原型路径
area=L.face_areas(x0,faces,Mu)
Jf,n1raw=L.face_jacobians(x0,xB,faces)
while Jf.dim()>4: Jf=Jf[0]
while n1raw.dim()>3: n1raw=n1raw[0]
from lam3d.core.rot_sync import polar_rotation
Rf=polar_rotation(Jf)
from lam3d.core.tga_covariance import matrix_to_quat
q_face=F.normalize(matrix_to_quat(Rf.reshape(-1,3,3)),dim=-1).reshape(Mu,-1,4)
qR,coh=L.markley(q_face,area,faces,N)

def qang(q):
    return 2*torch.atan2(q[...,1:].norm(dim=-1), q[...,0].abs().clamp(max=1.))
print("proto qR angle p50 %.3f | repo dq angle p50 %.3f"%(float(qang(qR).median()*180/math.pi), float(qang(dq_r[0]).median()*180/math.pi)))

def kabsch(x0n,xbn,fn):
    n=x0n.shape[0]; f=fn.astype(np.int64)
    a=np.concatenate([f[:,0],f[:,1],f[:,2]]); b=np.concatenate([f[:,1],f[:,2],f[:,0]])
    src=np.concatenate([a,b]); dst=np.concatenate([b,a])
    p=x0n[dst]-x0n[src]; q=xbn[dst]-xbn[src]
    h=np.zeros((n,3,3))
    for i in range(3):
        for j in range(3): h[:,i,j]=np.bincount(src,weights=p[:,i]*q[:,j],minlength=n)
    u,_,vt=np.linalg.svd(h.transpose(0,2,1))
    r=np.einsum("nji,nkj->nik",vt,u)
    bad=np.linalg.det(r)<0
    if bad.any():
        vt2=vt.copy(); vt2[bad,-1,:]*=-1; r=np.einsum("nji,nkj->nik",vt2,u)
    return r
def ang(a,b):
    tr=np.einsum("nij,nij->n",np.swapaxes(a,1,2),b)
    return np.degrees(np.arccos(np.clip((tr-1.)*.5,-1,1)))
X0n=x0.cpu().numpy().astype(np.float64); BXn=xB.cpu().numpy().astype(np.float64); FAn=faces.cpu().numpy()
rr=np.stack([kabsch(X0n,BXn[k],FAn) for k in range(Mu)])
R0=quat_to_matrix(q0)
Rd_repo=(quat_to_matrix(q1_r)@R0.transpose(-1,-2)).double().cpu().numpy()
Rd_proto=quat_to_matrix((qR@0+torch.cat([q0[...,:1],q0[...,1:] if qR.dim()==2 else q0[...,1:]],-1)))  # placeholder

def qmul(a,b):
    w1,x1,y1,z1=a.unbind(-1); w2,x2,y2,z2=b.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2],-1)
q1_p=qmul(qR,q0.unsqueeze(0))
Rd_proto=(quat_to_matrix(q1_p)@R0.transpose(-1,-2)).double().cpu().numpy()
out={}
for k,nm in enumerate(names):
    a_r=ang(rr[k],Rd_repo[k]); a_p=ang(rr[k],Rd_proto[k])
    # repo dq vs proto qR 逐顶点夹角
    dq_repo=(qmul(dq_r[0], torch.cat([qR.new_zeros(0),]),-1) if False else None)
    dot=(dq_r[0]*qR).sum(-1).abs().clamp(max=1.)
    gap=np.degrees(2*np.arccos(dot.cpu().numpy().clip(0,1)))
    out[nm]=dict(repo_p50=float(np.percentile(a_r,50)), repo_p90=float(np.percentile(a_r,90)),
                 proto_p50=float(np.percentile(a_p,50)), proto_p90=float(np.percentile(a_p,90)),
                 dqgap_p50=float(np.percentile(gap,50)), dqgap_p90=float(np.percentile(gap,90)))
    print("%-12s repo rig p50 %7.3f p90 %7.3f | proto rig p50 %7.3f p90 %7.3f | dq gap p50 %7.3f p90 %7.3f"%(
        nm,out[nm]["repo_p50"],out[nm]["repo_p90"],out[nm]["proto_p50"],out[nm]["proto_p90"],out[nm]["dqgap_p50"],out[nm]["dqgap_p90"]))
print("MEAN repo p50 %.4f p90 %.4f | proto p50 %.4f p90 %.4f"%(
    np.mean([v["repo_p50"] for v in out.values()]),np.mean([v["repo_p90"] for v in out.values()]),
    np.mean([v["proto_p50"] for v in out.values()]),np.mean([v["proto_p90"] for v in out.values()])))
json.dump(out, open("/tmp/m4_repocmp.json","w"), indent=1)
