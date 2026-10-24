# -*- coding: utf-8 -*-
"""统一口径扫描：G_dir 关（推荐）下的 tau / n_cap / aniso 门。"""
import sys, math, json
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
exec(open("/workspace/lam3d/scripts/proto_m4_final.py").read().split("VAR={")[0])

def build3(tau=0.30,p=1.5,th_cap=45.0,n_cap=25.0,gdir=False,aniso_gate=None,use_dir=True):
    g=((coh-tau)/(1-tau)).clamp(0,1)**p
    th=qang(qR)*180/math.pi
    t=torch.where(th>0,(th_cap/th.clamp(min=1e-9)),torch.ones_like(th)).clamp(max=1.0)
    dq=slerp_id(qR,t*g)
    u=rotv(dq,n0v)
    cr=torch.cross(u,nv,dim=-1); sn=cr.norm(dim=-1); cs=(u*nv).sum(-1)
    thn=torch.atan2(sn,cs); ax=cr/sn.clamp(min=1e-9).unsqueeze(-1)
    ncapr=n_cap*math.pi/180; thn_s=ncapr*torch.tanh(thn/ncapr)
    ad=g*t*((G_dir if gdir else 1.0)) if use_dir else torch.zeros_like(g)
    if aniso_gate is not None:
        ad=ad*(aniso>aniso_gate).float()
    half=ad*thn_s*0.5
    dqfix=torch.cat([torch.cos(half).unsqueeze(-1),torch.sin(half).unsqueeze(-1)*ax],-1)
    dqfix=torch.where((sn<1e-8).unsqueeze(-1)|(ad==0).unsqueeze(-1),EI.expand_as(dqfix),dqfix)
    qout=qmul(qmul(dqfix,dq),q0)
    return dict(q=qout,s=s0.unsqueeze(0).expand(M,-1,-1).clone())

rows={}
def run(tag,**kw):
    res=build3(**kw)
    m=[metrics(res,k) for k in range(M)]
    avg={kk:sum(mm[kk] for mm in m)/M for kk in m[0]}
    rows[tag]=dict(per=m,avg=avg)
    print("%-26s rig50=%7.3f rig90=%8.3f jit=%6.3f fm=%.4f%%"%(tag,avg["rigid"],avg["rigid90"],avg["jitter"],100*avg["fm"]))

run("REC gdir=off tau=0.3 cap=25")
run("gdir=on  tau=0.3 cap=25", gdir=True)
for tc in (0.5,0.7,0.9):
    run("tau=%.1f"%tc, tau=tc)
for nc in (0.0,12.0,15.0,18.0,25.0,30.0,35.0,45.0):
    run("cap=%.0f"%nc, n_cap=nc)
run("aniso>3 only", aniso_gate=3.0)
run("aniso>10 only", aniso_gate=10.0)
json.dump(rows, open("/tmp/m4_scan2.json","w"), ensure_ascii=False, indent=1)
print("saved")
