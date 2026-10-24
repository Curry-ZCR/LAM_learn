
# -*- coding: utf-8 -*-
"""方法四 s 支路：厚度不可辨识性 + 半径门归一化修正的实证。"""
import sys, math, json
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch, numpy as np, torch.nn.functional as F
import proto_m4_lib as L
from lam3d.core.rot_sync import polar_rotation
from lam3d.core.tga_covariance import matrix_to_quat, quat_to_matrix
torch.set_grad_enabled(False); dev="cuda"
x0,xB,faces,q0,s0,names=L.load()
M,N=xB.shape[0],x0.shape[0]; Fn=faces.shape[0]
area=L.face_areas(x0,faces,M)
_,n1f=L.face_jacobians(x0,xB,faces)
# 顶点法向、log 面积比
fa,fb,fc=faces.unbind(-1)
n0f=F.normalize(torch.cross(x0[fb]-x0[fa],x0[fc]-x0[fa],dim=-1),dim=-1,eps=1e-12)
def vsum(pf,w,Mv):
    s=torch.zeros(Mv,N,3,device=dev)
    for k in range(3): s.index_add_(1,faces[:,k],pf*w[:,:,None])
    return s
n0v=F.normalize(vsum(n0f.expand(M,-1,-1),area,M),dim=-1,eps=1e-12)
nv=F.normalize(vsum(n1f if n1f.dim()==3 else n1f[0],area,M),dim=-1,eps=1e-12)
Af2=0.5*torch.cross(xB[...,fb,:]-xB[...,fa,:],xB[...,fc,:]-xB[...,fa,:],dim=-1).norm(dim=-1)
lr=torch.log((Af2/area.clamp(min=1e-20)).clamp(1e-3,1e3))
lk=torch.zeros(M,N,device=dev); ws=torch.zeros(M,N,device=dev)
for k in range(3):
    lk.index_add_(1,faces[:,k],lr*area); ws.index_add_(1,faces[:,k],area)
lk=lk/ws.clamp(min=1e-12)

mu=(x0-x0.mean(0,keepdim=True)).norm(dim=-1)                      # [N]
smean=s0.mean(-1)                                                  # [N]
r_old=mu/smean.clamp(min=1e-9)
mu_ref=mu.median()
r_new=mu/mu_ref
print("=== 半径门归一化对照（真实 capture）===")
for nm,r in (("旧 rho_hat=mu/mean(s0)",r_old),("新 rho_hat=mu/median(mu)",r_new)):
    print("  %-26s p1=%8.3f p50=%8.3f p99=%8.3f  -> G_rho=exp(-.5(r/1)^2) p50=%.3e frac<0.1=%.1f%%"%(
        nm, torch.quantile(r,.01), r.median(), torch.quantile(r,.99),
        torch.exp(-.5*(r.median())**2), 100*float((torch.exp(-.5*r**2)<0.1).float().mean())))
print("  s0.mean p50=%.4e  mu p50=%.4e  mu/s0mean p50=%.2f"%(smean.median(),mu.median(),r_old.median()))
print()
print("=== s 支路三种口径对 shape_err / eig 的影响（只改 s，不动 q）===")
Lm=math.log(1.35)
def project(x):
    return torch.where(x.abs()>1e-9,(Lm/x.abs())*torch.tanh(x.abs()/Lm),torch.ones_like(x))*x
i3=s0.argmin(-1)
i3v=i3.view(1,-1,1).expand(M,-1,1); s0M=s0.unsqueeze(0).expand(M,-1,-1)
si=s0M.gather(-1,i3v).squeeze(-1)
Sq=(s0**2).unsqueeze(0).expand(M,-1,-1)
Sig=(quat_to_matrix(q0).double().unsqueeze(0).expand(M,-1,-1,-1)*Sq.double()[...,None,:])@quat_to_matrix(q0).double().unsqueeze(0).expand(M,-1,-1,-1).transpose(-1,-1)
Jv9,_=L.scatter_vertex(L.face_jacobians(x0,xB,faces)[0],area,faces,N); Jv=Jv9.view(M,N,3,3).double()
Sig_ex=(Jv*Sig)@Jv.transpose(-1,-2)
ev,_=torch.linalg.eigh(Sig_ex); lam_max=ev[...,-1].clamp(min=1e-30)
base=torch.sort(s0,-1).values
def report(tag,snew):
    sfin=s0M.clone().scatter_(-1,i3v,snew.unsqueeze(-1))
    S2=(quat_to_matrix(q0).double().unsqueeze(0).expand(M,-1,-1,-1)*(sfin.double()**2)[...,None,:])@quat_to_matrix(q0).double().unsqueeze(0).expand(M,-1,-1,-1).transpose(-1,-1)
    ev2,_=torch.linalg.eigh(S2)
    rel=((ev2[...,-1].clamp(min=1e-30)-lam_max).abs()/lam_max)
    out=[]
    for k in range(M):
        sz=S2[k].diagonal(dim1=-2,dim2=-1).sum(-1)
    print("  %-40s eigmax_rel p50=%.3e p99=%.3e  s'/s0 p50=%.4f  capfrac=%.2f%%"%(
        tag, rel.median(), torch.quantile(rel.reshape(-1),.99), (sfin/s0M).median(),
        100*float((sfin/s0M>1.349).float().mean())))
gamma=0.5
for tag,lks in (("off (s 不动)", torch.zeros_like(lk)),
                ("gamma=0.5 (均匀缩放一致)", 0.5*lk),
                ("gamma=0.0 (面积守恒)", 1.0*lk),
                ("gamma=0.5 + 新半径门 sigma=3", 0.5*lk)):
    lhat=project(lks)
    if "新半径门" in tag: lhat=lhat*torch.exp(-0.5*(r_new.unsqueeze(0)/3.0)**2)
    report(tag, si*torch.exp(lhat))
print()
print("=== |lk| 分布（顶点级，A'/A 线性面积的 log）===")
al=lk.abs().reshape(-1)
for thr in (0.1,0.3,math.log(1.35),math.log(1.5),math.log(2)):
    print("  |lk|>%.3f : %.2f%%"%(thr,100*float((al>thr).float().mean())))
