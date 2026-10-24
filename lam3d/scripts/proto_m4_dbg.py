
import sys, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import proto_m4_lib as L
torch.set_grad_enabled(False)
dev="cuda"
x0,xB,faces,q0,s0,names=L.load()
M=xB.shape[0]
Jf,n0=L.face_jacobians(x0,xB,faces)
area=L.face_areas(x0,faces,M)
print("area", area.shape, area.min().item(), area.max().item())
d=torch.linalg.det(Jf.float())
q=[.01,.5,.99]
print("detJ quantile per preset:", [[round(v,4) for v in torch.quantile(d[i], torch.tensor(q,device=dev)).tolist()] for i in range(M)])
print("||J-I||_F p50/p99:", [[round(v,4) for v in torch.quantile((Jf[i]-torch.eye(3,device=dev)).flatten(0,1).norm(dim=-1), torch.tensor([.5,.99],device=dev)).tolist()] for i in range(M)])
Af2 = 0.5*torch.cross(xB[...,faces[:,1],:]-xB[...,faces[:,0],:], xB[...,faces[:,2],:]-xB[...,faces[:,0],:], dim=-1).norm(dim=-1)
print("A'/A p50/p99:", [[round(v,4) for v in torch.quantile((Af2[i]/area[i].clamp(min=1e-20)), torch.tensor([.5,.99],device=dev)).tolist()] for i in range(M)])
print("x0 spread", x0.abs().mean().item(), x0.abs().max().item())
print("xB disp norm p50/p99:", [[round(v,5) for v in torch.quantile((xB[i]-x0).norm(dim=-1), torch.tensor([.5,.99],device=dev)).tolist()] for i in range(M)])
