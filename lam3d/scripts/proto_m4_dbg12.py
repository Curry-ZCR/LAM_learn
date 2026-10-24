
import sys
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch, torch.nn.functional as F
import proto_m4_lib as L
x0,xB,faces,q0,s0,names=L.load()
x0c=x0.unsqueeze(0); xBc=xB
area=L.face_areas(x0c,faces,5); Jf,n0f=L.face_jacobians(x0c,xBc,faces)
print("area",area.shape,"Jf",Jf.shape,"n0f",n0f.shape)
pf=n0f[0] if (n0f.dim()==3 and n0f.shape[0]==1) else n0f
print("pf",pf.shape,"w",area.shape, "prod",(pf*area[:,:,None]).shape)
