
# -*- coding: utf-8 -*-
"""定位 proto 的 A0(=RotSync 复刻) 与仓库真实 RotSync 的偏差来源。"""
import sys, math, json
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/LAM"); sys.path.insert(0,"/workspace/lam3d/scripts")
import numpy as np, torch, torch.nn.functional as F
import proto_m4_lib as L
from lam3d.core import cov_switch
from lam3d.core.pipeline import SingleImage3DPipeline
from lam3d.core.tga_covariance import quat_to_matrix, matrix_to_quat
from lam3d.core.rot_sync import polar_rotation

torch.set_grad_enabled(True)
pipe=SingleImage3DPipeline(device="cuda", lazy_load=True)
pipe.run(image_path="/workspace/LAM/assets/sample_input/messi.png", job_dir="/tmp/offline_job3", job_id="cmp")
torch.set_grad_enabled(False)
rend=pipe.arkit_session["renderer"]
mod=cov_switch._make("rot_sync", rend, {})
print("mod", type(mod).__name__, mod.average, mod.max_angle_deg, mod.rotation_gain, mod.coherence_gate, mod.deadzone_deg, mod.temporal_smoothing)

x0,xB,faces,q0,s0,names=L.load()
M=xB.shape[0]; N=x0.shape[0]
dev="cuda"

##### 1) 逐段对比：面 J → q_face → Markley
area_r = mod.face_areas(x0.unsqueeze(0))                     # [1,F]
Jf_r   = mod.face_jacobians(x0.unsqueeze(0), xB[:1])          # [1,F,3,3]
q_face_r = F.normalize(matrix_to_quat(polar_rotation(Jf_r)), dim=-1)
dq_k, coh_k, wsum_k = mod._markley_mean(q_face_r, area_r, N)

area_p = L.face_areas(x0, faces, M)                            # [M,F]
Jf_p, n0p = L.face_jacobians(x0, xB, faces)
while Jf_p.dim()>4: Jf_p = Jf_p[0]
q_face_p = F.normalize(matrix_to_quat(polar_rotation(Jf_p.reshape(-1,3,3))), dim=-1).reshape(M,-1,4)
qR_p, coh_p = L.markley(q_face_p, area_p, faces, N)

print("faces equal:", bool(torch.equal(mod.faces.long().cpu(), faces.long().cpu())))
print("area  maxdiff %.3e"%float((area_r[0]-area_p[0]).abs().max()))
print("Jf    maxdiff %.3e"%float((Jf_r[0]-Jf_p[0]).abs().max()))
print("qface maxdiff %.3e"%float((q_face_r[0]-q_face_p[0]).abs().max()))
print("qR    maxdiff %.3e"%float((dq_k[0]-qR_p[0]).abs().max()))
print("coh   maxdiff %.3e"%float((coh_k[0]-coh_p[0]).abs().max()))
def qang(q):
    return 2*torch.atan2(q[...,1:].norm(dim=-1), q[...,0].abs().clamp(max=1.))*180/math.pi
dot=(dq_k[0]*qR_p[0]).sum(-1).abs().clamp(max=1.)
print("angle(dq_repo, qR_proto) p50 %.4f p90 %.4f"%(float(qang(dq_k[0]).median()), 0) )
print("  gap p50 %.4f p90 %.4f"%(float(torch.rad2deg(2*torch.arccos(dot)).median()), float(torch.rad2deg(2*torch.arccos(dot)).quantile(0.9))))

##### 2) 全量：repo 输出 vs proto qR vs 二者 rig 误差
with torch.no_grad():
    dq_all, _ = mod.delta_rotation_scale(x0.unsqueeze(0).expand(M,-1,-1), xB)
q1_all, _ = mod.apply(dq_all, None, q0, s0)
R0 = quat_to_matrix(q0)
Rd_repo = (quat_to_matrix(q1_all)@R0.transpose(-1,-2)).double().cpu().numpy()

def qmul(a,b):
    w1,x1,y1,z1=a.unbind(-1); w2,x2,y2,z2=b.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2],-1)
Rd_proto = (quat_to_matrix(qmul(qR_p,q0.unsqueeze(0)))@R0.transpose(-1,-2)).double().cpu().numpy()

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
rr=np.stack([kabsch(X0n,BXn[k],FAn) for k in range(M)])
out={}
for k,nm in enumerate(names):
    a_r=ang(rr[k],Rd_repo[k]); a_p=ang(rr[k],Rd_proto[k])
    rot_p=ang(np.broadcast_to(np.eye(3),(N,3,3)),Rd_proto[k]); rot_r=ang(np.broadcast_to(np.eye(3),(N,3,3)),Rd_repo[k])
    d=(dq_all[k]*qR_p[k]).sum(-1).abs().clamp(max=1.)
    gap=torch.rad2deg(2*torch.arccos(d)).cpu().numpy()
    out[nm]=dict(repo=dict(p50=float(np.percentile(a_r,50)),p90=float(np.percentile(a_r,90)),
                           applied_p50=float(np.percentile(rot_r,50))),
                 proto=dict(p50=float(np.percentile(a_p,50)),p90=float(np.percentile(a_p,90)),
                            applied_p50=float(np.percentile(rot_p,50))),
                 dqgap=dict(p50=float(np.percentile(gap,50)),p90=float(np.percentile(gap,90))))
    print("%-12s REPO rig %6.3f/%7.3f app %6.3f | PROTO rig %6.3f/%7.3f app %6.3f | dq gap %6.3f/%7.3f"%(
      nm,out[nm]["repo"]["p50"],out[nm]["repo"]["p90"],out[nm]["repo"]["applied_p50"],
      out[nm]["proto"]["p50"],out[nm]["proto"]["p90"],out[nm]["proto"]["applied_p50"],
      out[nm]["dqgap"]["p50"],out[nm]["dqgap"]["p90"]))
print("MEAN repo p50 %.4f p90 %.4f | proto p50 %.4f p90 %.4f"%(
  np.mean([v["repo"]["p50"] for v in out.values()]),np.mean([v["repo"]["p90"] for v in out.values()]),
  np.mean([v["proto"]["p50"] for v in out.values()]),np.mean([v["proto"]["p90"] for v in out.values()])))
json.dump(out, open("/tmp/m4_repocmp2.json","w"), indent=1)
print("saved")
