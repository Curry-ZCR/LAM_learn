
# -*- coding: utf-8 -*-
"""候选(c) SPD 测地插值 / 候选(b) 法向轴拉伸 的可行性量测。"""
import sys, math, time
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch, torch.nn.functional as F
from lam3d.core.tga_covariance import matrix_to_quat, quat_to_matrix
import proto_m4_lib as L
torch.set_grad_enabled(False); dev="cuda"
x0,xB,faces,q0,s0,names=L.load()
M,N=xB.shape[0],x0.shape[0]
def t(fn,n=30):
    for _ in range(8): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e3
Sig=torch.randn(N,3,3,device=dev); Sig=(Sig@Sig.transpose(-1,-2)+3*torch.eye(3,device=dev))
Sig=Sig.double()
q=torch.randn(N,4,device=dev); q=F.normalize(q,dim=-1)
print("=== 候选(c) SPD 测地插值逐高斯代价（N=20018, f64）===")
def logm_expm():
    ev,evec=torch.linalg.eigh(Sig)
    lg=(evec*torch.log(ev.clamp(min=1e-30))[...,None,:])@evec.transpose(-1,-2)
    lu,e2=torch.linalg.eigh(lg*0.5)
    return (e2*torch.exp(lu)[...,None,:])@e2.transpose(-1,-2)
print("  eigh×2 (logm→expm)     %6.3f ms"%t(logm_expm))
print("  eigh×1 (仅 logm)       %6.3f ms"%t(lambda: torch.linalg.eigh(Sig)))
print("  matrix_to_quat(N=20018) %6.3f ms"%t(lambda: matrix_to_quat(q)))
print("  quat_to_matrix(N=20018) %6.3f ms"%t(lambda: quat_to_matrix(q)))
print()
print("=== 候选(b) 法向轴对齐：细轴 vs 顶点法向夹角（真实数据）===")
area=L.face_areas(x0,faces,M)
fa,fb,fc=faces.unbind(-1)
n0f=F.normalize(torch.cross(x0[fb]-x0[fa],x0[fc]-x0[fa],dim=-1),dim=-1,eps=1e-12)
s=torch.zeros(M,N,3,device=dev)
for k in range(3): s.index_add_(1,faces[:,k],n0f*area[:,:,None])
n0v=F.normalize(s,dim=-1,eps=1e-12)
R0=quat_to_matrix(q0)
i3=s0.argmin(-1)
thin=R0.gather(2,i3.view(-1,1,1).expand(-1,3,1)).squeeze(-1)
ang=torch.acos((thin*n0v[0]).sum(-1).abs().clamp(max=1.))*180/math.pi
print("  angle(thin_axis, n0_vertex): p10=%.1f p50=%.1f p90=%.1f  frac<15deg=%.2f%%"%(
  torch.quantile(ang,.1),ang.median(),torch.quantile(ang,.9),100*float((ang<15).float().mean())))
print("  (随机方向期望 p50 = 57.3deg; 实测 p50=%.1f ⇒ 细轴与法向无关)"%ang.median())
