
# -*- coding: utf-8 -*-
"""安慰剂对照：法向修正项是否真的"有方向"，还是仅仅等价于随机扰动/收缩。"""
import sys, math
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
exec(open("/workspace/lam3d/scripts/proto_m4_final.py").read().split("VAR={")[0])

def build2(do_dir=True, mode="real", tau=0.30, p=1.5, th_cap=45.0, n_cap=25.0, seed=0):
    g=((coh-tau)/(1-tau)).clamp(0,1)**p
    th=qang(qR)*180/math.pi
    t=torch.where(th>0,(th_cap/th.clamp(min=1e-9)),torch.ones_like(th)).clamp(max=1.0)
    dq=slerp_id(qR,t*g)
    u=rotv(dq,n0v)
    cr=torch.cross(u,nv,dim=-1); sn=cr.norm(dim=-1); cs=(u*nv).sum(-1)
    thn=torch.atan2(sn,cs); ax=cr/sn.clamp(min=1e-9).unsqueeze(-1)
    ncapr=n_cap*math.pi/180
    thn_s=ncapr*torch.tanh(thn/ncapr)
    ad=g*t*G_dir if do_dir else torch.zeros_like(g)
    if mode=="neg":            # 反号：把 nv 推离 u
        ax=-ax
    elif mode=="random":       # 同幅度随机轴
        torch.manual_seed(seed)
        ax=F.normalize(torch.randn_like(ax),dim=-1,eps=1e-12)
    elif mode=="shrink":       # 只做幅度收缩，不转向
        ad=ad*0
    half=ad*thn_s*0.5
    dqfix=torch.cat([torch.cos(half).unsqueeze(-1), torch.sin(half).unsqueeze(-1)*ax],-1)
    dqfix=torch.where((sn<1e-8).unsqueeze(-1)|(ad==0).unsqueeze(-1),EI.expand_as(dqfix),dqfix)
    qout=qmul(qmul(dqfix,dq),q0)
    return dict(q=qout,s=s0.unsqueeze(0).expand(M,-1,-1).clone(),dq=dq,dqfix=dqfix)

def rg(res,k):
    q1=res["q"][k:k+1]; r1=quat_to_matrix(q1).double()
    rd=(r1@R0.transpose(-1,-2))[0].float()
    ang=rot_ang(rref[k].transpose(-1,-2)@rd)*180/math.pi
    qd=qmul(q1[0],torch.cat([q0[...,0:1],-q0[...,1:]],-1))
    qa=qd.index_select(0,ri); qb=qd.index_select(0,rj)
    jit=(2*torch.acos((qa*qb).sum(-1).abs().clamp(max=1.)).mean()*180/math.pi).item() if False else (2*torch.acos((qa*qb).sum(-1).abs().clamp(max=1.))*180/math.pi).mean().item()
    return float(torch.quantile(ang,.5)),float(torch.quantile(ang,.9)),jit

MODES=[("none (A0)",False,"real"),("real (A3)",True,"real"),
       ("neg-sign 对照",True,"neg"),("random-axis 对照",True,"random"),
       ("shrink-only 对照",True,"shrink")]
print("%-22s %7s %7s %7s"%("mode","rig50","rig90","jitter"))
for nm,dd,md in MODES:
    res=build2(do_dir=dd,mode=md)
    v=[rg(res,k) for k in range(M)]
    print("%-22s %7.3f %7.3f %7.3f"%(nm,np.mean([a[0] for a in v]),np.mean([a[1] for a in v]),np.mean([a[2] for a in v])))
print()
print("修正角 thn（u 与 nv 夹角）分位：p50=%.3f p90=%.3f 度"%(
   float((torch.atan2(cr.norm(dim=-1),(rotv(slerp_id(qR,((coh-0.3)/0.7).clamp(0,1)**1.5),n0v)*nv).sum(-1))).median()*180/math.pi),
   float(torch.quantile(torch.atan2(cr.norm(dim=-1),(rotv(slerp_id(qR,((coh-0.3)/0.7).clamp(0,1)**1.5),n0v)*nv).sum(-1)),.9)*180/math.pi)))
