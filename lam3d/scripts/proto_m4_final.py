
# -*- coding: utf-8 -*-
"""方法四最终版：方向支路(必选) + 尺度支路(可选, 修正半径门)；输出文档引用表。"""
import sys, os, json, math, time
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import numpy as np, torch, torch.nn.functional as F
from lam3d.core.rot_sync import polar_rotation
from lam3d.core.tga_covariance import matrix_to_quat, quat_to_matrix
import proto_m4_lib as L
torch.set_grad_enabled(False); dev="cuda"
x0,xB,faces,q0,s0,names=L.load()
M,N=xB.shape[0],x0.shape[0]; Fn=faces.shape[0]
R0=quat_to_matrix(q0).double(); D0=s0.double()**2; lam0=torch.sort(D0,-1).values
area=L.face_areas(x0,faces,M)
Jf,n1raw=L.face_jacobians(x0,xB,faces)
while Jf.dim()>4: Jf=Jf[0]
while n1raw.dim()>3: n1raw=n1raw[0]
Rf=polar_rotation(Jf)
Jv9,_=L.scatter_vertex(Jf,area,faces,N); Jv=Jv9.view(M,N,3,3)
q_face=F.normalize(matrix_to_quat(Rf.reshape(-1,3,3)),dim=-1).reshape(M,Fn,4)
qR,coh=L.markley(q_face,area,faces,N)
fa,fb,fc=faces.unbind(-1)
def vsum(pf,w,Mv=M,B=None):
    s=torch.zeros(Mv,N,3,device=dev)
    for k in range(3): s.index_add_(1,faces[:,k],pf*w[:,:,None])
    return s
n0f=F.normalize(torch.cross(x0[fb]-x0[fa],x0[fc]-x0[fa],dim=-1),dim=-1,eps=1e-12)
n0v=F.normalize(vsum(n0f.expand(M,-1,-1),area),dim=-1,eps=1e-12)
nv=F.normalize(vsum(n1raw,area),dim=-1,eps=1e-12)
Af2=0.5*torch.cross(xB[...,fb,:]-xB[...,fa,:],xB[...,fc,:]-xB[...,fa,:],dim=-1).norm(dim=-1)
lr=torch.log((Af2/area.clamp(min=1e-20)).clamp(1e-3,1e3))

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

# Kabsch 一环参考（仓库口径）
def kabsch_one_ring(x0n,xbn,fn):
    n=x0n.shape[0]; f=fn.astype(np.int64)
    a=np.concatenate([f[:,0],f[:,1],f[:,2]]); b=np.concatenate([f[:,1],f[:,2],f[:,0]])
    src=np.concatenate([a,b]); dst=np.concatenate([b,a])
    p=x0n[dst]-x0n[src]; q=xbn[dst]-xbn[src]
    h=np.zeros((n,3,3))
    for i in range(3):
        for j in range(3): h[:,i,j]=np.bincount(src,weights=p[:,i]*q[:,j],minlength=n)
    u,_,vt=np.linalg.svd(h.transpose(0,2,1))
    r=np.einsum("nji,nkj->nik",vt,u)
    bad=np.linalg.det(r)<0
    if bad.any():
        vt2=vt.copy(); vt2[bad,-1,:]*=-1; r=np.einsum("nji,nkj->nik",vt2,u)
    return r
X0n=x0.cpu().numpy().astype(np.float64); BXn=xB.cpu().numpy().astype(np.float64); FAn=faces.cpu().numpy()
rref=torch.from_numpy(np.stack([kabsch_one_ring(X0n,BXn[k],FAn) for k in range(M)])).float().to(dev)
ref_ang=rot_ang(rref)*180/math.pi
ei=torch.cat([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
key=torch.unique(torch.cat([ei[:,0]*N+ei[:,1],ei[:,1]*N+ei[:,0]]))
ri=torch.div(key,N,rounding_mode="floor"); rj=key%N

i3=s0.argmin(-1); aniso=(s0.max(-1).values/s0.min(-1).values.clamp(min=1e-9))
G_amp=1.0/(1.0+(aniso/4.0)**2); G_dir=1.0-torch.exp(-aniso/1.5)
mu=(x0-x0.mean(0,keepdim=True)).norm(dim=-1)
G_rho_old=torch.exp(-0.5*(mu/s0.mean(-1).clamp(min=1e-9))**2)      # 旧门（失效）
G_rho_new=torch.exp(-0.5*(mu/mu.median()/3.0)**2)                   # 修正门 sigma=3

def build(tau=0.30,p=1.5,th_cap=45.0,n_cap=25.0,do_dir=True,do_scale=True,gs=0.5,
          grho=True,gamp=True,gdir=True,La=math.log(1.35),Lb=math.log(1.35)):
    g=((coh-tau)/(1-tau)).clamp(0,1)**p
    th=qang(qR)*180/math.pi
    t=torch.where(th>0,(th_cap/th.clamp(min=1e-9)),torch.ones_like(th)).clamp(max=1.0)
    dq=slerp_id(qR,t*g)
    u=rotv(dq,n0v)
    cr=torch.cross(u,nv,dim=-1); sn=cr.norm(dim=-1); cs=(u*nv).sum(-1)
    thn=torch.atan2(sn,cs); ax=cr/sn.clamp(min=1e-9).unsqueeze(-1)
    ncapr=n_cap*math.pi/180
    thn_s=ncapr*torch.tanh(thn/ncapr)
    ad=g*t*((G_dir if gdir else 1.0)) if do_dir else torch.zeros_like(g)
    half=ad*thn_s*0.5
    dqfix=torch.cat([torch.cos(half).unsqueeze(-1), torch.sin(half).unsqueeze(-1)*ax],-1)
    dqfix=torch.where((sn<1e-8).unsqueeze(-1)|(ad==0).unsqueeze(-1),EI.expand_as(dqfix),dqfix)
    qout=qmul(qmul(dqfix,dq),q0)
    # 尺度支路
    lk=torch.zeros(M,N,device=dev); ws=torch.zeros(M,N,device=dev)
    for k in range(3):
        lk.index_add_(1,faces[:,k],lr*area); ws.index_add_(1,faces[:,k],area)
    lk=lk/ws.clamp(min=1e-12)
    Lmv=torch.where(lk>0,torch.full_like(lk,La),torch.full_like(lk,Lb))
    rhov=torch.where(lk.abs()>1e-12,(Lmv/lk.abs())*torch.tanh(lk.abs()/Lmv),torch.ones_like(lk))
    lhat=lk*rhov
    GR=G_rho_new if grho else 1.0
    aa=gs*g*(G_amp if gamp else 1.0)*GR
    if not do_scale: aa=torch.zeros_like(aa)
    s0M=s0.unsqueeze(0).expand(M,-1,-1); i3v=i3.view(1,-1,1).expand(M,-1,1)
    si=s0M.gather(-1,i3v).squeeze(-1)
    sfin=s0M.clone().scatter_(-1,i3v,(si*torch.exp(aa*lhat)).unsqueeze(-1))
    return dict(q=qout,s=sfin,dq=dq,dqfix=dqfix,g=g,t=t,aa=aa,lk=lk)

def metrics(res,k,lk_full=None):
    q1=res["q"][k:k+1]; s1=res["s"][k:k+1]
    r1=quat_to_matrix(q1).double(); d1=s1.double()**2
    sig1=(r1*d1[:,:,None,:])@r1.transpose(-1,-2)
    jv=Jv[k:k+1].double()
    sig_ex=(jv*D0[None].unsqueeze(-2))@jv.transpose(-1,-2)
    sh=((sig1-sig_ex).norm(dim=(-2,-1))/sig_ex.norm(dim=(-2,-1)).clamp(min=1e-30))[0]
    lam1=torch.sort(d1,-1).values
    eig=((lam1/lam0.clamp(min=1e-30))-1).abs()[0]
    rd=(r1@R0.transpose(-1,-2))[0].float()
    ang=rot_ang(rref[k].transpose(-1,-2)@rd)*180/math.pi
    fm=((ang>20)&(ref_ang[k]<5)).float().mean().item()
    ratio=(s1/s0.clamp(min=1e-12))[0]
    oor=((ratio>1.35+1e-6)|(ratio<0.741)).float().mean().item()
    qd=qmul(q1[0],torch.cat([q0[...,0:1],-q0[...,1:]],-1))
    qa=qd.index_select(0,ri); qb=qd.index_select(0,rj)
    jit=(2*torch.acos((qa*qb).sum(-1).abs().clamp(max=1.))*180/math.pi).mean()
    return dict(shape=float(torch.quantile(sh.double(),.5)),eigmax=float(eig.max()),
                rigid=float(torch.quantile(ang,.5)),rigid90=float(torch.quantile(ang,.9)),
                fm=fm,oor=oor,jitter=float(jit))

VAR={
 "A0 rot-only (RotSync 基线)": dict(do_dir=False,do_scale=False),
 "A3 方向支路 (本方案核心)": dict(do_dir=True,do_scale=False),
 "A3+尺度支路 (修正门)": dict(do_dir=True,do_scale=True,gs=0.5),
 "A3+尺度支路 (旧门)": dict(do_dir=True,do_scale=True,gs=0.5,grho=False),
}
print("%-26s %7s %7s %8s %7s %7s %7s %6s"%("variant","shp50","eigmax","rig50","rig90","false%","oor%","jit"))
rows={}
for nm,kw in VAR.items():
    if nm.endswith("旧门"): 
        _g=G_rho_new
        G_rho_new=G_rho_old
        res=build(**kw); G_rho_new=_g
    else: res=build(**kw)
    ms=[metrics(res,k) for k in range(M)]
    agg={kk:float(np.mean([m[kk] for m in ms])) for kk in ms[0]}
    rows[nm]=dict(agg=agg,per=ms)
    print("%-26s %7.4f %8.2e %7.3f %7.3f %6.2f %6.2f %6.3f"%(
        nm,agg["shape"],agg["eigmax"],agg["rigid"],agg["rigid90"],100*agg["fm"],100*agg["oor"],agg["jitter"]))
print()
print("per-preset rigid p50:")
for nm in VAR:
    print("  %-26s "%nm + " ".join("%7.3f"%m["rigid"] for m in rows[nm]["per"]), " | ", names)
print("per-preset jitter:")
for nm in VAR:
    print("  %-26s "%nm + " ".join("%7.3f"%m["jitter"] for m in rows[nm]["per"]))
json.dump({k:v for k,v in rows.items()}, open("/tmp/m4_final.json","w"), default=float)
print("\nwrote /tmp/m4_final.json")
