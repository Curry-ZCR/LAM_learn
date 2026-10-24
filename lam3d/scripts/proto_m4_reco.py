
# -*- coding: utf-8 -*-
"""推荐配置扫描：tau / G_dir / theta_ncap / aniso 门，产出文档最终表。"""
import sys, math, json
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
exec(open("/workspace/lam3d/scripts/proto_m4_final.py").read().split("VAR={")[0])

def run(tau=0.30,p=1.5,th_cap=45.0,n_cap=25.0,gdir=False,aniso_gate=None,do_fix=True):
    g=((coh-tau)/(1-tau)).clamp(0,1)**p
    th=qang(qR)*180/math.pi
    t=torch.where(th>0,(th_cap/th.clamp(min=1e-9)),torch.ones_like(th)).clamp(max=1.0)
    dq=slerp_id(qR,t*g)
    u=rotv(dq,n0v)
    cr=torch.cross(u,nv,dim=-1); sn=cr.norm(dim=-1); cs=(u*nv).sum(-1)
    thn=torch.atan2(sn,cs); ax=cr/sn.clamp(min=1e-9).unsqueeze(-1)
    ncapr=n_cap*math.pi/180; thn_s=ncapr*torch.tanh(thn/ncapr)
    ad=(g*t) if do_fix else torch.zeros_like(g)
    if gdir: ad=ad*G_dir
    if aniso_gate is not None:
        ad=ad*(aniso>aniso_gate).float()
    half=ad*thn_s*0.5
    dqfix=torch.cat([torch.cos(half).unsqueeze(-1),torch.sin(half).unsqueeze(-1)*ax],-1)
    dqfix=torch.where((sn<1e-8).unsqueeze(-1)|(ad==0).unsqueeze(-1),EI.expand_as(dqfix),dqfix)
    return qmul(qmul(dqfix,dq),q0)

def ev(qout):
    r1=quat_to_matrix(qout).double()
    ang=rot_ang(rref.transpose(-1,-2)@(r1@R0.transpose(-1,-2)).float())*180/math.pi
    fm=((ang>20)&(ref_ang<5)).float().mean(1)
    qd=qmul(qout,torch.cat([q0[...,0:1],-q0[...,1:]],-1))
    qa=qd.index_select(1,ri); qb=qd.index_select(1,rj)
    jit=(2*torch.acos((qa*qb).sum(-1).abs().clamp(max=1.))*180/math.pi).mean(1)
    return dict(r50=ang.median(1).values,r90=torch.quantile(ang,.9,dim=1),fm=fm,jit=jit)

CFG=[("A0 基线 RotSync (无修正)",dict(do_fix=False)),
     ("tau=0.3 n=25 Gdir=off",dict(tau=0.3)),
     ("tau=0.5 n=25 Gdir=off",dict(tau=0.5)),
     ("tau=0.3 n=25 Gdir=on ",dict(tau=0.3,gdir=True)),
     ("tau=0.3 n=15 Gdir=off",dict(tau=0.3,n_cap=15)),
     ("tau=0.3 n=35 Gdir=off",dict(tau=0.3,n_cap=35)),
     ("tau=0.3 n=25 aniso>10",dict(tau=0.3,aniso_gate=10.0)),
     ("tau=0.3 n=25 aniso>3 ",dict(tau=0.3,aniso_gate=3.0))]
print("%-24s %7s %7s %7s %6s"%("config","rig50","rig90","jit","false%"))
best={}
for nm,kw in CFG:
    r=ev(run(**kw))
    print("%-24s %7.3f %7.3f %7.3f %6.2f"%(nm,r["r50"].mean(),r["r90"].mean(),r["jit"].mean(),100*r["fm"].mean()))
    best[nm]=r
print()
print("per-preset rig50 (推荐配置 tau=0.3 n=25 Gdir=off):")
r=best["tau=0.3 n=25 Gdir=off"]
print("  "+" ".join("%7.3f"%v for v in r["r50"].tolist()),"  vs A0:")
print("  "+" ".join("%7.3f"%v for v in best["A0 基线 RotSync (无修正)"]["r50"].tolist()))
print("  reduction%: "+" ".join("%6.1f"%v for v in (100*(1-r["r50"]/best["A0 基线 RotSync (无修正)"]["r50"])).tolist()))
print("  presets:", names)
