
# -*- coding: utf-8 -*-
"""新发现核验：面积加权矩阵平均 J 再极分解（A4）是否优于 Markley 聚合（A0），
并检验 A4+法向修正（A5）是否进一步降低 rigid_err。"""
import sys, math
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
exec(open("/workspace/lam3d/scripts/proto_m4_final.py").read().split("VAR={")[0])
from lam3d.core.rot_sync import polar_rotation as PR

# --- A4: 矩阵平均 J -> 极分解 -> 单位四元数（无逐顶点聚合损失）---
Javg=(Jf*area[...,None,None]).sum(1)/area.sum(1).clamp(min=1e-12).view(M,1,1)
Ravg=PR(Javg.unsqueeze(1)).reshape(M,3,3)
qA4=F.normalize(matrix_to_quat(Ravg),dim=-1)          # [M,4]
qA4v=qA4.unsqueeze(1).expand(M,N,4).contiguous()
print("=== A4 与 qR(Markley) 的差异 ===")
d=qmul(qA4v,torch.cat([qR[...,0:1],-qR[...,1:]],-1))
print("  angle(A4, Markley) p50=%.3f p90=%.3f deg"%(
  (qang(d)*180/math.pi).median(), torch.quantile(qang(d)*180/math.pi,.9)))
print("  但 A4 对法向的残差 p50=%.3f（vs Markley 2.087）"%float(torch.atan2(torch.cross(rotv(qA4v,n0v),nv,dim=-1).norm(dim=-1),(rotv(qA4v,n0v)*nv).sum(-1)).median()*180/math.pi))

def run(qc, do_fix=True, n_cap=25.0, tau=0.30, p=1.5, th_cap=45.0):
    g=((coh-tau)/(1-tau)).clamp(0,1)**p
    th=qang(qc)*180/math.pi
    t=torch.where(th>0,(th_cap/th.clamp(min=1e-9)),torch.ones_like(th)).clamp(max=1.0)
    dq=slerp_id(qc,t*g)
    u=rotv(dq,n0v)
    cs=(u*nv).sum(-1); sn=torch.cross(u,nv,dim=-1).norm(dim=-1)
    thn=torch.atan2(sn,cs); ax=torch.cross(u,nv,dim=-1)/sn.clamp(min=1e-9).unsqueeze(-1)
    ncapr=n_cap*math.pi/180; thn_s=ncapr*torch.tanh(thn/ncapr)
    ad=(g*t*G_dir) if do_fix else torch.zeros_like(g)
    half=ad*thn_s*0.5
    dqfix=torch.cat([torch.cos(half).unsqueeze(-1),torch.sin(half).unsqueeze(-1)*ax],-1)
    dqfix=torch.where((sn<1e-8).unsqueeze(-1)|(ad==0).unsqueeze(-1),EI.expand_as(dqfix),dqfix)
    return qmul(qmul(dqfix,dq),q0)

def ev(tag,qout):
    r1=quat_to_matrix(qout).double()
    rd=(r1@R0.transpose(-1,-2)).float()
    ang=rot_ang(rref.transpose(-1,-2)@rd)*180/math.pi
    fm=(((ang>20)&(ref_ang<5)).float().mean(1))
    qd=qmul(qout,torch.cat([q0[...,0:1],-q0[...,1:]],-1))
    qa=qd.index_select(1,ri); qb=qd.index_select(1,rj)
    jit=(2*torch.acos((qa*qb).sum(-1).abs().clamp(max=1.))*180/math.pi).mean(1)
    # 法向残差
    nr=torch.atan2(torch.cross(rotv(qout,n0v),nv,dim=-1).norm(dim=-1),(rotv(qout,n0v)*nv).sum(-1))*180/math.pi
    print("%-34s rig50=%6.3f rig90=%6.3f jit=%6.3f false%%=%5.2f nres50=%6.3f"%(tag,
      ang.median(),torch.quantile(ang,.9),jit.mean(),100*fm.mean(),nr.median()))
    return ang.median().item(), jit.mean().item()

print()
print("=== 五种聚合 × 是否法向修正 ===")
ev("A0 Markley, no fix",       run(qR, do_fix=False))
ev("A3 Markley, fix",          run(qR, do_fix=True))
ev("A4 MatAvg, no fix",        run(qA4v, do_fix=False))
ev("A5 MatAvg, fix",           run(qA4v, do_fix=True))
# 直接：不用极分解，用 n0v→nv 的最小旋转合成（Gram-Schmidt 帧）
e1v=F.normalize(torch.cross(torch.randn_like(n0v),n0v,dim=-1),dim=-1,eps=1e-12)
e2v=torch.cross(n0v,e1v,dim=-1)
fb1=F.normalize(torch.cross(torch.randn_like(nv),nv,dim=-1),dim=-1,eps=1e-12)
fb2=torch.cross(nv,fb1,dim=-1)
print()
print("=== 每 preset 明细（rig50）===")
for tag,qc,df in (("A0 Markley",qR,False),("A3 Markley+fix",qR,True),("A4 MatAvg",qA4v,False),("A5 MatAvg+fix",qA4v,True)):
    qo=run(qc,do_fix=df); r1=quat_to_matrix(qo).double()
    ang=rot_ang(rref.transpose(-1,-2)@(r1@R0.transpose(-1,-2)).float())*180/math.pi
    print("  %-16s "%tag + " ".join("%7.3f"%v for v in ang.median(1).tolist()))
print("  presets:", names)
