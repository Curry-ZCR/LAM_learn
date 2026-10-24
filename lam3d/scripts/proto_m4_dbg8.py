
import sys, math, types
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch, numpy as np, torch.nn.functional as F
from lam3d.core.rot_sync import RotSyncModulator, polar_rotation
from lam3d.core.tga_covariance import matrix_to_quat, quat_to_matrix
import proto_m4_lib as L
torch.set_grad_enabled(False); dev="cuda"
x0,xB,faces,q0,s0,names=L.load()
M,N=xB.shape[0],x0.shape[0]; Fn=faces.shape[0]
obj=types.SimpleNamespace(faces=faces, faces_flat=faces.reshape(-1), eps=1e-12, average="face",
    deadzone_deg=0.0, rotation_gain=1.0, max_angle_deg=45.0,
    coherence_gate=0.0, temporal_smoothing=0.0, _prev_delta=None)
for nm in ("face_areas","face_jacobians","vertex_jacobians","_markley_mean","rotation_field"):
    setattr(obj, nm, types.MethodType(getattr(RotSyncModulator,nm), obj))
# B=1 逐向量调用（repo 语义：x0 与 xB 同 B）
dqs=[]
for k in range(M):
    dq,diag = obj.rotation_field(x0.unsqueeze(0), xB[k:k+1])
    dqs.append(dq[0])
    print("%-12s repo dq angle p50=%7.3f p90=%7.3f"%(names[k], float(diag["angle_deg"].median()), float(torch.quantile(diag["angle_deg"],.9))))
dq=torch.stack(dqs)
area=L.face_areas(x0,faces,M); Jf,n1f=L.face_jacobians(x0,xB,faces)
Rf=polar_rotation(Jf); q_face=F.normalize(matrix_to_quat(Rf.reshape(-1,3,3)),dim=-1).reshape(M,Fn,4)
qR,coh=L.markley(q_face,area,faces,N)
d=2*torch.acos((qR*dq).sum(-1).abs().clamp(max=1.))*180/math.pi
print("my qR vs repo dq: max=%.4f p50=%.6f"%(d.max(), d.median()))
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
def ang(A,B):
    R=A.transpose(-1,-2)@B
    tr=R[...,0,0]+R[...,1,1]+R[...,2,2]
    return torch.acos(((tr-1)*.5).clamp(-1.,1.))*180/math.pi
Rd=quat_to_matrix(dq); Rm=quat_to_matrix(qR); I=torch.eye(3,device=dev).expand(N,3,3)
for k,nm in enumerate(names):
    print("%-12s kabsch_vs_I p50=%6.3f | repo_vs_kabsch p50=%6.3f | mine_vs_kabsch p50=%6.3f"%(
        nm, ang(I,IA[k]).median(), ang(IA[k],Rd[k]).median(), ang(IA[k],Rm[k]).median()))
