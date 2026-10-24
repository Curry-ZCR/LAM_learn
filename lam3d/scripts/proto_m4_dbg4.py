
import sys, math
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch, numpy as np
import proto_m4_lib as L
torch.set_grad_enabled(False); dev="cuda"
x0,xB,faces,q0,s0,names=L.load()
M,N=xB.shape[0],x0.shape[0]
def kabsch_np(x0n,xbn,facesn):
    n=x0n.shape[0]; f=facesn.astype(np.int64)
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
X0=x0.cpu().numpy().astype(np.float64); BX=xB.cpu().numpy().astype(np.float64); FA=faces.cpu().numpy()
rr=torch.from_numpy(np.stack([kabsch_np(X0,BX[k],FA) for k in range(M)])).float().to(dev)
def ang(A,B):
    R=A.transpose(-1,-2)@B
    tr=R[...,0,0]+R[...,1,1]+R[...,2,2]
    return torch.acos(((tr-1)*.5).clamp(-1.,1.))*180/math.pi
import inspect
from lam3d.core.rot_sync import RotSyncModulator
print(inspect.signature(RotSyncModulator.__init__))
print(inspect.getsource(RotSyncModulator.delta_rotation_scale))
print(inspect.getsource(RotSyncModulator.apply))
