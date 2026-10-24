
import sys, math, inspect
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch, numpy as np
from lam3d.core.tga_covariance import quat_to_matrix, matrix_to_quat
from lam3d.core.rot_sync import RotSyncModulator, polar_rotation
import proto_m4_lib as L
torch.set_grad_enabled(False); dev="cuda"
x0,xB,faces,q0,s0,names=L.load()
M,N=xB.shape[0],x0.shape[0]; Fn=faces.shape[0]
area=L.face_areas(x0,faces,M); Jf,n1f=L.face_jacobians(x0,xB,faces)
Rf=polar_rotation(Jf); q_face=matrix_to_quat(Rf.reshape(-1,3,3)).reshape(M,Fn,4)
qR_mine,coh_mine=L.markley(q_face,area,faces,N)

class FM(torch.nn.Module):
    def __init__(s): super().__init__()
fm=FM(); fm.faces=faces
mod=RotSyncModulator(fm)
print("params:", mod.rotation_gain, mod.max_angle_deg, mod.coherence_gate, mod.average)
q_repo,_=mod.delta_rotation_scale(x0.unsqueeze(0), xB)
qm_repo,coh_repo,wsum=mod._markley_mean(q_face, area, N)
print("markley match:", float((qm_repo-qR_mine).abs().max()), "coh match:", float((coh_repo-coh_mine).abs().max()))
# also compare to polar of vertex jacobian
Jv9,_=L.scatter_vertex(Jf,area,faces,N); Jv=Jv9.view(M,N,3,3)

print(inspect.getsource(mod._markley_mean)[:2500])
