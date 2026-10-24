
# -*- coding: utf-8 -*-
"""补：尺度支路合成核验（分别开/关 G_amp、G_rho）+ 逐段耗时。"""
import sys, math, time
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch, torch.nn.functional as F
import proto_m4_lib as L
from lam3d.core.rot_sync import polar_rotation
from lam3d.core.tga_covariance import matrix_to_quat, quat_to_matrix
import importlib.util
spec=importlib.util.spec_from_file_location("v","/workspace/lam3d/scripts/proto_m4_verify.py")
torch.set_grad_enabled(False); dev="cuda"
x0,xB,faces,q0,s0,names=L.load()
M,N=xB.shape[0],x0.shape[0]; Fn=faces.shape[0]

# 直接内联 pipeline（与 verify.py 同源）
_src=open("/workspace/lam3d/scripts/proto_m4_verify.py").read()
_src=_src.split('print("=== 1)')[0]
_src=_src[_src.index("def qmul"):]
exec(_src)

print("=== 尺度支路合成核验（xB = x0 (R Su)^T，R=20.05deg）===")
th=torch.tensor(0.35); axv=F.normalize(torch.tensor([0.3,-0.7,0.6],device=dev),dim=0)
c,s_=torch.cos(th),torch.sin(th)
R=torch.stack([torch.stack([c+axv[0]**2*(1-c), axv[0]*axv[1]*(1-c)-axv[2]*s_, axv[0]*axv[2]*(1-c)+axv[1]*s_]),
               torch.stack([axv[1]*axv[0]*(1-c)+axv[2]*s_, c+axv[1]**2*(1-c), axv[1]*axv[2]*(1-c)-axv[0]*s_]),
               torch.stack([axv[2]*axv[0]*(1-c)-axv[1]*s_, axv[2]*axv[1]*(1-c)+axv[0]*s_, c+axv[2]**2*(1-c)])])
for tag,Su in (("iso 1.20",1.2*torch.eye(3,device=dev)),
               ("diag(1,1,1.2)",torch.diag(torch.tensor([1.,1.,1.2],device=dev))),
               ("diag(1.3,0.8,1.1)",torch.diag(torch.tensor([1.3,0.8,1.1],device=dev)))):
    T=(R@Su).float()
    xb=(x0@T.T).unsqueeze(0).expand(5,-1,-1).contiguous()
    row=[]
    for gamp,grho in ((True,True),(False,False)):
        r=pipeline(x0.unsqueeze(0),xb,gamp=gamp,grho=grho)
        rat=(r["s"][0]/s0).median().item()
        lk_med=r["lk"][0].median().item()
        row.append("gamp=%s grho=%s -> s'/s0 p50=%.4f (lk=%+.4f, exp(lk)=%.4f)"%(gamp,grho,rat,lk_med,math.exp(lk_med)))
    print("  %-20s %s"%(tag," | ".join(row)))

print()
print("=== 逐段耗时（B=1，RTX4090，warm 20 次取均值）===")
xb1=xB[0:1]
def t(fn,n=20):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e3
area1=L.face_areas(x0.unsqueeze(0),faces,1)
Jf1,_=L.face_jacobians(x0.unsqueeze(0),xb1,faces)
while Jf1.dim()>4 and Jf1.shape[0]==1: Jf1=Jf1[0]
Rf1=polar_rotation(Jf1)
qf1=F.normalize(matrix_to_quat(Rf1.reshape(-1,3,3)),dim=-1).reshape(1,Fn,4)
qR1,coh1=L.markley(qf1,area1,faces,N)
Jv1,_=L.scatter_vertex(Jf1,area1,faces,N)
from lam3d.core.rot_sync import polar_rotation as pr
print("  face_areas            %6.3f ms"%t(lambda: L.face_areas(x0.unsqueeze(0),faces,1)))
print("  face_jacobians        %6.3f ms"%t(lambda: L.face_jacobians(x0.unsqueeze(0),xb1,faces)))
print("  polar_rotation(8步)   %6.3f ms"%t(lambda: pr(Jf1)))
print("  matrix_to_quat(39904) %6.3f ms"%t(lambda: matrix_to_quat(Rf1.reshape(-1,3,3))))
print("  markley(area-w)       %6.3f ms"%t(lambda: L.markley(qf1,area1,faces,N)))
print("  scatter_vertex(Jv)    %6.3f ms"%t(lambda: L.scatter_vertex(Jf1,area1,faces,N)))
def full(): return pipeline(x0.unsqueeze(0),xb1)
print("  --- 新增项（方法四相对 RotSync 增量）---")
print("  vertex_normal (n0,n1) %6.3f ms"%t(lambda: [F.normalize(torch.cross(x0[faces[:,1]]-x0[faces[:,0]],x0[faces[:,2]]-x0[faces[:,0]],dim=-1),dim=-1,eps=1e-12), F.normalize(torch.cross(xb1[0][faces[:,1]]-xb1[0][faces[:,0]],xb1[0][faces[:,2]]-xb1[0][faces[:,0]],dim=-1),dim=-1,eps=1e-12)]))
def logk():
    fa,fb,fc=faces.unbind(-1)
    Af2=0.5*torch.cross(xb1[...,fb,:]-xb1[...,fa,:],xb1[...,fc,:]-xb1[...,fa,:],dim=-1).norm(dim=-1)
    lr=torch.log((Af2/area1.clamp(min=1e-20)).clamp(1e-3,1e3))
    lk=torch.zeros(1,N,device=dev); ws=torch.zeros(1,N,device=dev)
    for k in range(3):
        lk.index_add_(1,faces[:,k],lr*area1); ws.index_add_(1,faces[:,k],area1)
    return lk/ws.clamp(min=1e-12)
print("  vertex log(k) scatter %6.3f ms"%t(logk))
print("  pipeline FULL (B=1)   %6.3f ms"%t(full))
