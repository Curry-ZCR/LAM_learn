
import sys, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import proto_m4_lib as L
torch.set_grad_enabled(False); dev="cuda"
x0,xB,faces,q0,s0,names=L.load(); M,N=xB.shape[0],x0.shape[0]; Fn=faces.shape[0]
area=L.face_areas(x0,faces,M)
Jf,n1f=L.face_jacobians(x0,xB,faces)
Af2=0.5*torch.cross(xB[...,faces[:,1],:]-xB[...,faces[:,0],:],xB[...,faces[:,2],:]-xB[...,faces[:,0],:],dim=-1).norm(dim=-1)
detJ=torch.linalg.det(Jf.float())
ratio=detJ.abs()/(Af2/area.clamp(min=1e-30))
q=torch.tensor([.001,.01,.5,.99,.999],device=dev)
print("ratio quant (all) %s" % [round(float(v),4) for v in torch.quantile(ratio.flatten().double(),q.double())])
mask=(area>1e-8).flatten()
print("area>1e-8 frac %.4f" % mask.float().mean().item())
print("ratio quant (area>1e-8) %s" % [round(float(v),4) for v in torch.quantile(ratio.flatten()[mask].double(),q.double())])
# 用 double 重算 J 判定是否 f32 误差
x0d=x0.double(); xBd=xB.double()
a,b,c=faces.unbind(-1)
e1=x0d[b]-x0d[a]; e2=x0d[c]-x0d[a]
d1=xBd[...,b,:]-xBd[...,a,:]; d2=xBd[...,c,:]-xBd[...,a,:]
n0=F.normalize(torch.cross(e1,e2,dim=-1),dim=-1,eps=1e-30)
n1=F.normalize(torch.cross(d1,d2,dim=-1),dim=-1,eps=1e-30)
Jd=torch.stack((d1.unsqueeze(0) if d1.dim()==2 else d1,),0) if False else None
E=torch.stack((e1,e2,n0),-1).unsqueeze(0).expand(M,-1,-1,-1)
Et=torch.stack((d1,d2,n1),-1)
Jdd=Et@torch.linalg.inv(E)
det2=torch.linalg.det(Jdd)
print("double recompute: max|ratio-1| %.3e" % ((det2.abs()/(Af2.double()/area.double().clamp(min=1e-30))-1).abs().max().item()))
print("f32 Jf vs f64 Jdd max abs diff %.3e" % (Jf.double()-Jdd).abs().max().item())
print("cond(J) p50/p99/max %s" % [round(float(v),2) for v in torch.quantile(torch.linalg.cond(Jdd),q.double())])
