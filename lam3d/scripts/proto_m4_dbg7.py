
import sys, math, types, inspect
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch, numpy as np, torch.nn.functional as F
from lam3d.core.rot_sync import RotSyncModulator
from lam3d.core.tga_covariance import quat_to_matrix, matrix_to_quat
import proto_m4_lib as L
torch.set_grad_enabled(False); dev="cuda"
x0,xB,faces,q0,s0,names=L.load()
M,N=xB.shape[0],x0.shape[0]; Fn=faces.shape[0]

# 用 repo 的真实方法绑定出一个轻量对象，跑 repo 的 rotation_field 源码
obj=types.SimpleNamespace(faces=faces, eps=1e-12, average="face",
    deadzone_deg=0.0, rotation_gain=1.0, max_angle_deg=45.0,
    coherence_gate=0.0, temporal_smoothing=0.0, _prev_delta=None)
for nm in ("face_areas","face_jacobians","vertex_jacobians","_markley_mean","rotation_field"):
    setattr(obj, nm, types.MethodType(getattr(RotSyncModulator,nm), obj))
dq,diag = obj.rotation_field(x0.unsqueeze(0), xB)
print("repo delta_q", dq.shape, "identity?", float((dq[0,0]-torch.tensor([1.,0,0,0],device=dev)).abs().max()))

# 我的实现
area=L.face_areas(x0,faces,M); Jf,n1f=L.face_jacobians(x0,xB,faces)
from lam3d.core.rot_sync import polar_rotation
Rf=polar_rotation(Jf); q_face=F.normalize(matrix_to_quat(Rf.reshape(-1,3,3)),dim=-1).reshape(M,Fn,4)
qR,coh=L.markley(q_face,area,faces,N)
print("qR vs repo delta_q max|diff| =", float((qR[...,:]-dq).abs().max()), " median angle diff =",
      float((2*torch.acos((qR*dq).sum(-1).abs().clamp(max=1.))*180/math.pi).median()))
print("diag keys", list(diag.keys())[:12])
print("angle p50 repo", float(diag["angle_deg"].median()))
