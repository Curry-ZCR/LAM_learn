
import sys, math, torch
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch.nn.functional as F
exec(open("/workspace/lam3d/scripts/proto_m4_final.py").read().split("VAR={")[0])

def jac64(x0v,xBv):
    a,b,c=faces.unbind(-1)
    e1=x0v[b]-x0v[a]; e2=x0v[c]-x0v[a]
    d1=xBv[b]-xBv[a]; d2=xBv[c]-xBv[a]
    n0=F.normalize(torch.cross(e1,e2,dim=-1),dim=-1,eps=1e-12)
    n1=F.normalize(torch.cross(d1,d2,dim=-1),dim=-1,eps=1e-12)
    return torch.stack((d1,d2,n1),-1)@torch.linalg.inv(torch.stack((e1,e2,n0),-1))
a0,b0,c0=faces.unbind(-1)
e1=x0[b0]-x0[a0]; e2=x0[c0]-x0[a0]
n0=F.normalize(torch.cross(e1,e2,dim=-1),dim=-1,eps=1e-12)
n1=F.normalize(torch.cross(xB[0][b0]-xB[0][a0],xB[0][c0]-xB[0][a0],dim=-1),dim=-1,eps=1e-12)
J32=torch.stack((xB[0][b0]-xB[0][a0],xB[0][c0]-xB[0][a0],n1),-1)@torch.linalg.inv(torch.stack((e1,e2,n0),-1))
J64=jac64(x0.double(),xB[0].double()).float()
d=(J32-J64).abs().amax(-1).amax(-1)
ar0=L.face_areas(x0,faces,1).reshape(-1)
print("f32 vs f64 逐面 J 最大元差：p50=%.3e p99=%.3e p99.9=%.3e max=%.3e"%(
  d.median(),torch.quantile(d,.99),torch.quantile(d.double(),.999),d.max()))
big=d>1.0
print("误差>1.0 的面数 = %d (%.4f%%)，其面积占全网格 = %.3e%%"%(
  int(big.sum()),100*big.float().mean(),100*(ar0[big].sum()/ar0.sum())))
print("这些面的静止面积 p50=%.3e  vs 全体 p50=%.3e"%(ar0[big].median() if big.any() else ar0.median(),ar0.median()))
