# -*- coding: utf-8 -*-
"""安慰剂对照 v2：推荐配置（G_dir 关）下的方向性检验。"""
import sys, math, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
exec(open("/workspace/lam3d/scripts/proto_m4_final.py").read().split("VAR={")[0])

def build5(mode="real", do_fix=True, tau=0.30, p=1.5, th_cap=45.0, n_cap=25.0, seed=0):
    g=((coh-tau)/(1-tau)).clamp(0,1)**p
    th=qang(qR)*180/math.pi
    t=torch.where(th>0,(th_cap/th.clamp(min=1e-9)),torch.ones_like(th)).clamp(max=1.0)
    dq=slerp_id(qR,t*g)
    u=rotv(dq,n0v)
    cr=torch.cross(u,nv,dim=-1); sn=cr.norm(dim=-1); cs=(u*nv).sum(-1)
    thn=torch.atan2(sn,cs); ax=cr/sn.clamp(min=1e-9).unsqueeze(-1)
    ncapr=n_cap*math.pi/180; thn_s=ncapr*torch.tanh(thn/ncapr)
    ad=g*t if do_fix else torch.zeros_like(g)
    if mode=="neg": ax=-ax
    elif mode=="random":
        torch.manual_seed(seed); ax=F.normalize(torch.randn_like(ax),dim=-1,eps=1e-12)
    elif mode=="shrink": ad=ad*0
    half=ad*thn_s*0.5
    dqfix=torch.cat([torch.cos(half).unsqueeze(-1),torch.sin(half).unsqueeze(-1)*ax],-1)
    dqfix=torch.where((sn<1e-8).unsqueeze(-1)|(ad==0).unsqueeze(-1),EI.expand_as(dqfix),dqfix)
    qout=qmul(qmul(dqfix,dq),q0)
    return dict(q=qout,s=s0.unsqueeze(0).expand(M,-1,-1).clone())

print("%-22s %7s %7s %7s %7s"%("mode","rig50","rig90","jitter","fm%"))
rows={}
for nm,df,md in (("none (A0)",False,"real"),("real (REC)",True,"real"),
                 ("neg-sign",True,"neg"),("random-axis",True,"random"),
                 ("shrink-only",True,"shrink")):
    res=build5(mode=md,do_fix=df)
    per=[metrics(res,k) for k in range(M)]
    avg={kk:sum(pp[kk] for pp in per)/M for kk in per[0]}
    rows[nm]=dict(per=per,avg=avg)
    print("%-22s %7.3f %7.3f %7.3f %7.4f"%(nm,avg["rigid"],avg["rigid90"],avg["jitter"],100*avg["fm"]))
json.dump(rows, open("/tmp/m4_placebo2.json","w"), ensure_ascii=False, indent=1)
print("saved")
