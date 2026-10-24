
import sys, math
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import numpy as np, torch
import proto_m4_lib as L
from lam3d.core.rot_sync import RotSyncModulator
from lam3d.core.tga_covariance import quat_to_matrix
x0,xB,faces,q0,s0,names=L.load()
BS="/workspace/LAM/model_zoo/human_parametric_models/flame_assets/flame/flame_arkit_bs.npy"
mod=RotSyncModulator(None, BS)
mod.to("cuda").eval()
with torch.no_grad():
    dq,pl=mod.delta_rotation_scale(x0.unsqueeze(0).cuda(), xB.cuda())
    q1,s1=mod.apply(dq, mod.delta_scale(pl,s0.cuda()), q0.cuda(), s0.cuda())
print("dq",tuple(dq.shape),"diag:",{k:(tuple(v.shape) if torch.is_tensor(v) else v) for k,v in pl.items()})
R0=quat_to_matrix(q0.cuda()); r1=quat_to_matrix(q1)
X0n=x0.cpu().numpy().astype(np.float64); BXn=xB.cpu().numpy().astype(np.float64); FAn=faces.cpu().numpy()
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
rr=np.stack([kabsch(X0n,BXn[k],FAn) for k in range(xB.shape[0])])
rd=(r1@R0.transpose(-1,-2)).double().cpu().numpy()
a0=[]
for k,nm in enumerate(names):
    a=ang(rr[k],rd[k]); a0.append(a)
    ra=ang(np.broadcast_to(np.eye(3),(len(a),3,3)),rd[k])
    fm=float(np.mean((a>20)&(ang(np.broadcast_to(np.eye(3),(len(a),3,3)),rr[k])<5)))
    print("%-12s rigid p50 %.3f p90 %.3f | applied p50 %.3f | false %.4f%%"%(nm,np.percentile(a,50),np.percentile(a,90),np.percentile(ra,50),100*fm))
allv=np.concatenate(a0)
print("MEAN-of-presets p50 %.4f p90 %.4f | pooled p50 %.4f"%(np.mean([np.percentile(v,50) for v in a0]),np.mean([np.percentile(v,90) for v in a0]),np.percentile(allv,50)))
