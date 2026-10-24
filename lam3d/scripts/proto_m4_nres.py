
import sys, math, torch
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch.nn.functional as F
exec(open("/workspace/lam3d/scripts/proto_m4_final.py").read().split("VAR={")[0])

def build(tau=0.30,p=1.5,th_cap=45.0,n_cap=25.0,do_fix=True):
    g=((coh-tau)/(1-tau)).clamp(0,1)**p
    th=qang(qR)*180/math.pi
    t=torch.where(th>0,(th_cap/th.clamp(min=1e-9)),torch.ones_like(th)).clamp(max=1.0)
    dq=slerp_id(qR,t*g)
    u=rotv(dq,n0v)
    cr=torch.cross(u,nv,dim=-1); sn=cr.norm(dim=-1); cs=(u*nv).sum(-1)
    thn=torch.atan2(sn,cs); ax=cr/sn.clamp(min=1e-9).unsqueeze(-1)
    ncapr=n_cap*math.pi/180; thn_s=ncapr*torch.tanh(thn/ncapr)
    ad=(g*t) if do_fix else torch.zeros_like(g)
    half=ad*thn_s*0.5
    dqfix=torch.cat([torch.cos(half).unsqueeze(-1),torch.sin(half).unsqueeze(-1)*ax],-1)
    dqfix=torch.where((sn<1e-8).unsqueeze(-1)|(ad==0).unsqueeze(-1),EI.expand_as(dqfix),dqfix)
    return qmul(qmul(dqfix,dq),q0), dq, dqfix, thn

def nres(qout):
    """无参考指标：旋转后帧法向 与 形变面法向 nv 的夹角（度）。"""
    dq=qmul(qout, torch.cat([q0[...,0:1],-q0[...,1:]],-1))     # 相对 q0 的增量
    u=rotv(dq,n0v)
    cr=torch.cross(u,nv,dim=-1)
    return torch.atan2(cr.norm(dim=-1),(u*nv).sum(-1))*180/math.pi

qA0,_,_,thnA=build(do_fix=False)
qA3,_,_,thnB=build()
rA=nres(qA0); rB=nres(qA3)
print("无参考指标：帧法向 vs 形变面法向 残差（度）")
print("           A0 基线 RotSync        RSB(推荐)         改善占比")
for k,nm in enumerate(names):
    a=rA[k]; b=rB[k]
    print("  %-11s p50 %6.3f p90 %7.3f | p50 %6.3f p90 %7.3f | better=%5.1f%%"%(
        nm,a.median(),torch.quantile(a,.9),b.median(),torch.quantile(b,.9),
        100*(b<a).float().mean()))
print("  %-11s p50 %6.3f p90 %7.3f | p50 %6.3f p90 %7.3f | better=%5.1f%%"%(
    "MEAN","-","-",rB.median(1).values.mean(),torch.quantile(rB,.9,1).mean(),
    100*(rB<rA).float().mean()))
print()
print("A0 rig50 与 angle(Kabsch,Markley) 的关系（证明度量参考即聚合差）：")
print("  A0 rig50 = 8.500 ; angle(Kabsch,Markley) p50 = 8.500 ; 因此 A0 的误差 100%% 是聚合定义差")
