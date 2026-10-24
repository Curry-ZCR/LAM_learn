# -*- coding: utf-8 -*-
"""推荐配置（G_dir 关）逐 preset 明细，供文档 §10.1 使用。"""
import sys, math, json
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
exec(open("/workspace/lam3d/scripts/proto_m4_final.py").read().split("VAR={")[0])

def build4(tau=0.30,p=1.5,th_cap=45.0,n_cap=25.0,gdir=False,use_dir=True):
    g=((coh-tau)/(1-tau)).clamp(0,1)**p
    th=qang(qR)*180/math.pi
    t=torch.where(th>0,(th_cap/th.clamp(min=1e-9)),torch.ones_like(th)).clamp(max=1.0)
    dq=slerp_id(qR,t*g)
    u=rotv(dq,n0v)
    cr=torch.cross(u,nv,dim=-1); sn=cr.norm(dim=-1); cs=(u*nv).sum(-1)
    thn=torch.atan2(sn,cs); ax=cr/sn.clamp(min=1e-9).unsqueeze(-1)
    ncapr=n_cap*math.pi/180; thn_s=ncapr*torch.tanh(thn/ncapr)
    ad=g*t*((G_dir if gdir else 1.0)) if use_dir else torch.zeros_like(g)
    half=ad*thn_s*0.5
    dqfix=torch.cat([torch.cos(half).unsqueeze(-1),torch.sin(half).unsqueeze(-1)*ax],-1)
    dqfix=torch.where((sn<1e-8).unsqueeze(-1)|(ad==0).unsqueeze(-1),EI.expand_as(dqfix),dqfix)
    qout=qmul(qmul(dqfix,dq),q0)
    return dict(q=qout,s=s0.unsqueeze(0).expand(M,-1,-1).clone()), g, t, thn*180/math.pi, (ad*thn_s)*180/math.pi

out={}
for tag,kw in (("A0",dict(use_dir=False)),("REC",dict())):
    res,g,t,thn_deg,applied=build4(**kw)
    per=[metrics(res,k) for k in range(M)]
    avg={kk:sum(p[kk] for p in per)/M for kk in per[0]}
    out[tag]=dict(per=per,avg=avg)
    print(tag, " ".join("%s=%.4f"%(k,avg[k]) for k in ("rigid","rigid90","jitter","fm","shape")))
    for nm,p in zip(names,per):
        print("   %-11s rig50=%7.3f rig90=%8.3f jit=%6.3f fm=%.4f%%"%(nm,p["rigid"],p["rigid90"],p["jitter"],100*p["fm"]))
# 修正角统计（推荐配置）
print("thn_deg quantiles:", [round(float(torch.quantile(thn_deg,q)),3) for q in (0.1,0.25,0.5,0.75,0.9,0.99)])
print("frac>5/15/25:", [round(float((thn_deg>x).float().mean()),4) for x in (5,15,25)])
print("applied quantiles:", [round(float(torch.quantile(applied,q)),3) for q in (0.5,0.9,0.99)])
json.dump(out, open("/tmp/m4_rec.json","w"), ensure_ascii=False, indent=1)
print("saved /tmp/m4_rec.json")
