
# -*- coding: utf-8 -*-
"""方法四 RSB：正确性核验（刚体/纯拉伸）、四法同口径对比、逐段耗时。"""
import sys, math, time
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch, numpy as np, torch.nn.functional as F
from lam3d.core.rot_sync import polar_rotation
from lam3d.core.tga_covariance import matrix_to_quat, quat_to_matrix
import proto_m4_lib as L
torch.set_grad_enabled(False); dev="cuda"
x0,xB,faces,q0,s0,names=L.load()
M,N=xB.shape[0],x0.shape[0]; Fn=faces.shape[0]

def qmul(a,b):
    w1,x1,y1,z1=a.unbind(-1); w2,x2,y2,z2=b.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2],-1)
def qang(q): return 2*torch.atan2(q[...,1:].norm(dim=-1), q[...,0].abs().clamp(max=1.))
def slerp_id(q,t):
    q=torch.where(q[...,0:1]<0,-q,q).clone()
    ang=2*torch.acos(q[...,0].clamp(-1,1)); s=torch.sin(ang).clamp(min=1e-8)
    w=torch.where(ang<1e-6, torch.ones_like(ang), torch.sin(ang*t)/s)
    return F.normalize(w.unsqueeze(-1)*q, dim=-1)
def rotv(q,v):
    u=q[...,1:]; w=q[...,0:1]
    return v+2*w*torch.cross(u,v,dim=-1)+2*torch.cross(u,torch.cross(u,v,dim=-1),dim=-1)
def rot_ang(R):
    tr=R[...,0,0]+R[...,1,1]+R[...,2,2]
    return torch.acos(((tr-1)*.5).clamp(-1.,1.))
EI=torch.tensor([1.,0,0,0],device=dev)

# ---------------- 1) 合成正确性核验 ----------------
def synth(R, Su):
    """xB = x0 @ (R Su)^T 的全局相似变换（绕原点），返回逐顶点真值变换。"""
    T=(R@Su).float()
    return (x0@T.T).unsqueeze(0).expand(M,-1,-1).contiguous(), T.unsqueeze(0).expand(N,-1,-1)

def pipeline(x0c, xBc, abar=1.0, tau=0.30, p=1.5, th_cap=45.0, n_cap=25.0,
             La=math.log(1.35), Lb=math.log(1.35), gamp=True, gdir=True, grho=True):
    Mm=xBc.shape[0]
    area=L.face_areas(x0c,faces,Mm); Jf,n0f=L.face_jacobians(x0c,xBc,faces)
    while Jf.dim()>4 and Jf.shape[0]==1: Jf=Jf[0]
    while n0f.dim()>3 and n0f.shape[0]==1: n0f=n0f[0]
    Rf=polar_rotation(Jf); qf=F.normalize(matrix_to_quat(Rf.reshape(-1,3,3)),dim=-1).reshape(Mm,Fn,4)
    q_face,coh=L.markley(qf,area,faces,N)
    Jv9,_=L.scatter_vertex(Jf,area,faces,N); Jv=Jv9.view(Mm,N,3,3)
    # 法向
    def vsum(pf,w):
        s=torch.zeros(Mm,N,3,device=dev)
        for k in range(3): s.index_add_(1,faces[:,k],pf*w[:,:,None])
        return s
    fa,fb,fc=faces.unbind(-1)
    e1=x0c[...,fb,:]-x0c[...,fa,:]; e2=x0c[...,fc,:]-x0c[...,fa,:]
    d1=xBc[...,fb,:]-xBc[...,fa,:]; d2=xBc[...,fc,:]-xBc[...,fa,:]
    n0b=F.normalize(torch.cross(e1,e2,dim=-1),dim=-1,eps=1e-12)
    n1b=F.normalize(torch.cross(d1,d2,dim=-1),dim=-1,eps=1e-12)
    nv=F.normalize(vsum(n1b,area),dim=-1,eps=1e-12)
    n0v=F.normalize(vsum(n0b,area),dim=-1,eps=1e-12)
    Af2=0.5*torch.cross(xBc[...,faces[:,1],:]-xBc[...,faces[:,0],:],xBc[...,faces[:,2],:]-xBc[...,faces[:,0],:],dim=-1).norm(dim=-1)
    logr=torch.log((Af2/area.clamp(min=1e-20)).clamp(min=1e-3,max=1e3))
    lk=torch.zeros(Mm,N,device=dev); ws=torch.zeros(Mm,N,device=dev)
    for k in range(3):
        lk.index_add_(1,faces[:,k],logr*area); ws.index_add_(1,faces[:,k],area)
    lk=lk/ws.clamp(min=1e-12)
    g=((coh-tau)/(1-tau)).clamp(0,1)**p
    th=qang(q_face)*180/math.pi
    t=torch.where(th>0,(th_cap/th.clamp(min=1e-9)),torch.ones_like(th)).clamp(max=1.0)
    dq=slerp_id(q_face,t*g)
    Lm=torch.where(lk>0, torch.full_like(lk,La), torch.full_like(lk,Lb))
    rho=torch.where(lk.abs()>1e-9,(Lm/lk.abs())*torch.tanh(lk.abs()/Lm.clamp(min=1e-9)),torch.ones_like(lk))
    lhat=lk*rho
    i3=s0.argmin(-1); aniso=(s0.max(-1).values/s0.min(-1).values.clamp(min=1e-9))
    G_amp=1.0/(1.0+(aniso/4.0)**2); G_dir=1.0-torch.exp(-aniso/1.5)
    mu=(x0c-x0c.mean(0,keepdim=True)).norm(dim=-1); rho_hat=mu/s0.mean(-1).clamp(min=1e-9)
    G_rho=torch.exp(-0.5*(rho_hat/1.0)**2)
    aa=abar*g*(G_amp if gamp else 1.0)*(G_rho if grho else 1.0)
    u=rotv(dq,n0v)
    cr=torch.cross(u,nv,dim=-1); sn=cr.norm(dim=-1); cs=(u*nv).sum(-1)
    thn=torch.atan2(sn,cs); ax=cr/sn.clamp(min=1e-9).unsqueeze(-1)
    thn_s=n_cap*math.pi/180*torch.tanh(thn/(n_cap*math.pi/180))
    ad=abar*g*t*((G_dir if gdir else 1.0))
    dqfix=torch.cat([torch.cos(ad*thn_s*0.5).unsqueeze(-1), torch.sin(ad*thn_s*0.5).unsqueeze(-1)*ax],-1)
    dqfix=torch.where((sn<1e-8).unsqueeze(-1)|(ad==0).unsqueeze(-1), EI.expand_as(dqfix), dqfix)
    qout=qmul(qmul(dqfix,dq), q0)
    s0M=s0.unsqueeze(0).expand(Mm,-1,-1); i3v=i3.view(1,-1,1).expand(Mm,-1,1)
    si=s0M.gather(-1,i3v).squeeze(-1)
    sfin=s0M.clone().scatter_(-1,i3v,(si*torch.exp(aa*lhat)).unsqueeze(-1))
    return dict(q=qout,s=sfin,coh=coh,g=g,dq=dq,dqfix=dqfix,lk=lk,Jv=Jv,nv=nv,area=area,thn=thn)

def tga_ref(x0c,xBc,str_=1.0,hi=1.35,lo=0.25):
    Mm=xBc.shape[0]
    area=L.face_areas(x0c,faces,Mm); Jf,_=L.face_jacobians(x0c,xBc,faces)
    Jv9,_=L.scatter_vertex(Jf,area,faces,N); Jv=Jv9.view(Mm,N,3,3)
    S0=quat_to_matrix(q0).double().unsqueeze(0).expand(Mm,-1,-1,-1)
    D0=(s0.double()**2).unsqueeze(0).expand(Mm,-1,-1)
    Sig=(S0*D0[...,None,:])@S0.transpose(-1,-2)
    Jd=Jv.double()
    if str_!=1.0: Jd=Jd*str_+(1-str_)*torch.eye(3,device=dev,dtype=Jd.dtype)
    Sn=(Jd*Sig)@Jd.transpose(-1,-2)
    ev,evec=torch.linalg.eigh(Sn)
    ev=ev.clamp(min=1e-30)
    sv=ev.sqrt()
    base=torch.sort(s0.double(),-1).values.unsqueeze(0).expand(Mm,-1,-1)
    r=(sv/base.clamp(min=1e-12)).clamp(lo,hi)
    svf=base*r
    # 符号对齐：让 evec 与 R0 对应（用行列式保证右手）
    evec=torch.where(torch.linalg.det(evec).unsqueeze(-1).unsqueeze(-1)<0,
                     evec*torch.tensor([1.,1.,-1.],device=dev,dtype=evec.dtype),evec)
    qq=matrix_to_quat(evec.reshape(-1,3,3)).reshape(Mm,N,4)
    return dict(q=F.normalize(qq,dim=-1),s=svf)

print("=== 1) 合成正确性核验 ===")
th=torch.tensor(0.35); axv=F.normalize(torch.tensor([0.3,-0.7,0.6],device=dev),dim=0)
R=torch.tensor([[torch.cos(th)+axv[0]**2*(1-torch.cos(th)), axv[0]*axv[1]*(1-torch.cos(th))-axv[2]*torch.sin(th), axv[0]*axv[2]*(1-torch.cos(th))+axv[1]*torch.sin(th)],
                [axv[1]*axv[0]*(1-torch.cos(th))+axv[2]*torch.sin(th), torch.cos(th)+axv[1]**2*(1-torch.cos(th)), axv[1]*axv[2]*(1-torch.cos(th))-axv[0]*torch.sin(th)],
                [axv[2]*axv[0]*(1-torch.cos(th))-axv[1]*torch.sin(th), axv[2]*axv[1]*(1-torch.cos(th))+axv[0]*torch.sin(th), torch.cos(th)+axv[2]**2*(1-torch.cos(th))]],device=dev)
print("R orth err", float((R@R.T-torch.eye(3,device=dev)).abs().max()), "det", float(torch.linalg.det(R)))
for tag,Su in (("rigid", torch.eye(3,device=dev)),
               ("iso1.2", 1.2*torch.eye(3,device=dev)),
               ("aniso diag(1,1,1.2)", torch.diag(torch.tensor([1.,1.,1.2],device=dev))),
               ("aniso diag(1.3,0.8,1.1)", torch.diag(torch.tensor([1.3,0.8,1.1],device=dev)))):
    xb,T=synth(R,Su)
    res=pipeline(x0.unsqueeze(0),xb)
    Rd=quat_to_matrix(res["q"][0:1])[0]@quat_to_matrix(q0).inverse()
    ang=rot_ang(Rd@T[0].transpose(-1,-2))*180/math.pi
    coh=res["coh"][0]
    # 尺度：把 s' 与 s0*Su 的对角比较（仅当 Su 与 R0 主轴对齐时才有意义，这里报范围）
    print("  %-22s applied-rot err p50=%7.3f p90=%7.3f  coh p50=%.4f  cap-hit=%.2f%%"%(
        tag, float(ang.median()), float(torch.quantile(ang,.9)), float(coh.median()), 100*float((qang(res["dq"][0])*180/math.pi>45).float().mean())))
    print("       dq(fix) angle p50=%.4f deg, s'/s0 ratio p50=%.4f"%(
        float((qang(res["dqfix"][0])*180/math.pi).median()), float((res["s"][0]/s0).median())))
