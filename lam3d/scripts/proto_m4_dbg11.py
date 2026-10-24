
import sys, math
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch, numpy as np, torch.nn.functional as F
from lam3d.core.rot_sync import polar_rotation
from lam3d.core.tga_covariance import matrix_to_quat, quat_to_matrix
import proto_m4_lib as L
torch.set_grad_enabled(False); dev="cuda"
x0,xB,faces,q0,s0,names=L.load()
M,N=xB.shape[0],x0.shape[0]; Fn=faces.shape[0]
area=L.face_areas(x0,faces,M); Jf,n1f=L.face_jacobians(x0,xB,faces)
# 顶点级 J：面积加权
Jv9,_=L.scatter_vertex(Jf,area,faces,N); Jv=Jv9.view(M,N,3,3)
# 顶点级 J：邻接平均（另一种）
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
X0=x0.cpu().numpy().astype(np.float64); FA=faces.cpu().numpy(); BX=xB.cpu().numpy().astype(np.float64)
IA=torch.from_numpy(np.stack([kabsch_np(X0,BX[k],FA) for k in range(M)])).float().to(dev)
Rf=polar_rotation(Jf); q_face=F.normalize(matrix_to_quat(Rf.reshape(-1,3,3)),dim=-1).reshape(M,Fn,4)
qR,coh=L.markley(q_face,area,faces,N)
Rv=polar_rotation(Jv)
def ang(A,B):
    R=A.transpose(-1,-2)@B
    tr=R[...,0,0]+R[...,1,1]+R[...,2,2]
    return torch.acos(((tr-1)*.5).clamp(-1.,1.))*180/math.pi
for k,nm in enumerate(names):
    a=ang(IA[k],Rv[k]); b=ang(IA[k],quat_to_matrix(qR[k:k+1])[0]); c=ang(Rv[k],quat_to_matrix(qR[k:k+1])[0])
    print("%-12s kabsch~Jv_polar p50=%6.3f p90=%6.3f | kabsch~face_markley p50=%6.3f | Jv~face p50=%6.3f"%(
        nm,a.median(),torch.quantile(a,.9),b.median(),c.median()))
