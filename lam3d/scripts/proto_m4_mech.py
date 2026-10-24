
# -*- coding: utf-8 -*-
"""机理确认：法向残差 = "聚合损失"（Markley 均值与面积均值的不可交换性）。"""
import sys, math
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
exec(open("/workspace/lam3d/scripts/proto_m4_final.py").read().split("VAR={")[0])

def resid(dq_in):
    u=rotv(dq_in,n0v)
    cs=(u*nv).sum(-1); sn=torch.cross(u,nv,dim=-1).norm(dim=-1)
    return torch.atan2(sn,cs)*180/math.pi

# (i) 顶点级 J 的极分解（等价"每面极分解后再平均矩阵"的另一路径）
Rvp=polar_rotation(Jv)
dqv=F.normalize(matrix_to_quat(Rvp.reshape(-1,3,3)),dim=-1).reshape(M,N,4)
r_v=resid(dqv)
# (ii) 逐面极分解 → Markley 面积加权四元数平均（RotSync 通路）
r_m=resid(slerp_id(qR,torch.ones_like(coh)))
# (iii) 纯面积加权矩阵平均后再极分解（第三种聚合）
from lam3d.core.rot_sync import polar_rotation as PR
Javg=(Jf*area[...,None,None]).sum(1)/area.sum(1).clamp(min=1e-12).view(M,1,1)
dqa=F.normalize(matrix_to_quat(PR(Javg.unsqueeze(1)).reshape(M,3,3)),dim=-1).unsqueeze(1).expand(M,N,4).contiguous()
r_a=resid(dqa)
print("=== 法向残差 angle(R_agg·n0v, n1v) 按聚合方式（deg）===")
for nm,r in (("(i) 顶点J极分解",r_v),("(ii) 面极分解+Markley(本方案)",r_m),("(iii) 面积加权矩阵平均+极分解",r_a)):
    print("  %-32s p50=%7.3f p90=%7.3f p99=%7.3f"%(nm,torch.quantile(r.reshape(-1),.5),torch.quantile(r.reshape(-1),.9),torch.quantile(r.reshape(-1),.99)))
print()
print("=== 若 n1v 用【同一聚合】重算，残差是否为 0（一致性检查）===")
# 用 Jv 直接映射 n0v：n_pred = Jv^{-T} n0v 归一（余切迹法向），与面积平均 n1v 的差
Jvd=Jv.double()
n_pred=torch.linalg.solve(Jvd.transpose(-1,-2), n0v.double().unsqueeze(-1)).squeeze(-1)
n_pred=F.normalize(n_pred,dim=-1)
a1=torch.acos((n_pred*nv.double()).sum(-1).abs().clamp(max=1.))*180/math.pi
print("  angle(Jv^{-T}n0v, n1v_areaAvg) p50=%.3f p90=%.3f (应≈0 若两者一致)"%(torch.quantile(a1.reshape(-1),.5),torch.quantile(a1.reshape(-1),.9)))
