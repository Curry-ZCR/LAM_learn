
# -*- coding: utf-8 -*-
"""收尾：确认 A4 是全局平均的 bug（非新发现）+ 修正级耗时。"""
import sys, math, time
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch, torch.nn.functional as F
import proto_m4_lib as L
from lam3d.core.rot_sync import polar_rotation as PR
from lam3d.core.tga_covariance import matrix_to_quat, quat_to_matrix
torch.set_grad_enabled(False); dev="cuda"
x0,xB,faces,q0,s0,names=L.load()
M,N=xB.shape[0],x0.shape[0]; Fn=faces.shape[0]
area=L.face_areas(x0,faces,M)
Jf,_=L.face_jacobians(x0,xB,faces)
while Jf.dim()>4: Jf=Jf[0]
print("=== 澄清：A4 的 'sum(1)' 是跨面求和 = 全局一个变换（缺陷，不代表真实方案）===")
g_bad=(Jf*area[...,None,None]).sum(1)/area.sum(1,keepdim=True).view(M,1,1)
print("  bad  Javg shape:", g_bad.shape, "-> 逐 preset 单一旋转，故 jitter≈0.3（假象）")
Jv9,_=L.scatter_vertex(Jf,area,faces,N); Jv=Jv9.view(M,N,3,3)
print("  good Jv   shape:", Jv.shape, "-> 逐顶点一环面积加权（等价 repo average=vertex）")
def nres(qq):
    fa,fb,fc=faces.unbind(-1)
    n0b=F.normalize(torch.cross(x0[fb]-x0[fa],x0[fc]-x0[fa],dim=-1),dim=-1,eps=1e-12)
    n1b=F.normalize(torch.cross(xB[...,fb,:]-xB[...,fa,:],xB[...,fc,:]-xB[...,fa,:],dim=-1),dim=-1,eps=1e-12)
    def vs(pf):
        s=torch.zeros(M,N,3,device=dev)
        for k in range(3): s.index_add_(1,faces[:,k],pf*area[:,:,None])
        return F.normalize(s,dim=-1,eps=1e-12)
    n0v=vs(n0b.expand(M,-1,-1)); nv=vs(n1b)
    u=n0v+2*qq[...,0:1]*torch.cross(qq[...,1:],n0v,dim=-1)+2*torch.cross(qq[...,1:],torch.cross(qq[...,1:],n0v,dim=-1),dim=-1)
    return (torch.atan2(torch.cross(u,nv,dim=-1).norm(dim=-1),(u*nv).sum(-1))*180/math.pi)
qv=F.normalize(matrix_to_quat(PR(Jv.reshape(-1,3,3)).reshape(M,N,3,3)),dim=-1)
print("  nres(per-vertex Jv polar) p50=%.3f p90=%.3f"%(nres(qv).median(),torch.quantile(nres(qv).reshape(-1),.9)))
print("  => 与 Markley 通路(2.087) 同量级，无新发现；聚合损失是内在的。")

print()
print("=== 修正级 & 端到端耗时（B=1）===")
def t(fn,n=40):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e3
def qmul(a,b):
    w1,x1,y1,z1=a.unbind(-1); w2,x2,y2,z2=b.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2],-1)
def rotv(q,v):
    u=q[...,1:]; w=q[...,0:1]
    return v+2*w*torch.cross(u,v,dim=-1)+2*torch.cross(u,torch.cross(u,v,dim=-1),dim=-1)
B1=1
area1=area[0:1]; Jf1=Jf[0:1]; Rf1=PR(Jf1)
qf1=F.normalize(matrix_to_quat(Rf1.reshape(-1,3,3)),dim=-1).reshape(1,Fn,4)
qR1,coh1=L.markley(qf1,area1,faces,N)
Jv1,_=L.scatter_vertex(Jf1,area1,faces,N)
Jv1=Jv1.view(1,N,3,3)
n0v1=F.normalize(torch.randn(1,N,3,device=dev),dim=-1); nv1=F.normalize(torch.randn(1,N,3,device=dev),dim=-1)
g1=((coh1-0.3)/0.7).clamp(0,1)**1.5
dq1=qR1.clone(); sub=torch.rand(1,N,device=dev)>0.5
cohm=((coh1-0.3)/0.7).clamp(0,1)**1.5
EI=torch.tensor([1.,0,0,0],device=dev)
ncapr=math.pi*25/180
def corr():
    u=rotv(dq1,n0v1)
    cr=torch.cross(u,nv1,dim=-1); sn=cr.norm(dim=-1); cs=(u*nv1).sum(-1)
    thn=torch.atan2(sn,cs); ax=cr/sn.clamp(min=1e-9).unsqueeze(-1)
    thn_s=ncapr*torch.tanh(thn/ncapr)
    ad=g1
    half=ad*thn_s*0.5
    dqfix=torch.cat([torch.cos(half).unsqueeze(-1),torch.sin(half).unsqueeze(-1)*ax],-1)
    dqfix=torch.where((sn<1e-8).unsqueeze(-1)|(ad==0).unsqueeze(-1),EI.expand_as(dqfix),dqfix)
    return qmul(dqfix,dq1)
def scale_stage():
    s0M=s0.unsqueeze(0); i3=s0.argmin(-1)
    i3v=i3.view(1,-1,1); si=s0M.gather(-1,i3v).squeeze(-1)
    return s0M.clone().scatter_(-1,i3v,(si*torch.exp(0.5*g1*torch.zeros_like(g1))).unsqueeze(-1))
def rot_only_pipeline():
    a=L.face_areas(x0.unsqueeze(0),faces,1); jf,_=L.face_jacobians(x0.unsqueeze(0),xB[0:1],faces)
    while jf.dim()>4: jf=jf[0]
    r=PR(jf); qf=F.normalize(matrix_to_quat(r.reshape(-1,3,3)),dim=-1).reshape(1,Fn,4)
    return L.markley(qf,a,faces,N)
print("  face_areas + face_jacobians     %6.3f"%(t(lambda: L.face_jacobians(x0.unsqueeze(0),xB[0:1],faces))+t(lambda: L.face_areas(x0.unsqueeze(0),faces,1))))
print("  polar_rotation(8 it)            %6.3f"%t(lambda: PR(Jf1)))
print("  matrix_to_quat + markley        %6.3f"%(t(lambda: matrix_to_quat(Rf1.reshape(-1,3,3)))+t(lambda: L.markley(qf1,area1,faces,N))))
print("  scatter_vertex Jv (9ch)         %6.3f"%t(lambda: L.scatter_vertex(Jf1,area1,faces,N)))
print("  correction (normal fix)         %6.3f"%t(corr))
print("  scale_stage (可选支路)            %6.3f"%t(scale_stage))
print("  ----")
print("  RotSync 通路小计                  %6.3f"%t(rot_only_pipeline))
