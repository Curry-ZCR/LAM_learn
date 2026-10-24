
import sys, math, time, torch
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch.nn.functional as F
exec(open("/workspace/lam3d/scripts/proto_m4_final.py").read().split("VAR={")[0])

def bench(fn,it=100,w=25):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3

# 冻结的 mock：x0/xB 用真实 capture 的第 0 个 preset 展开成 [1,...]
x0b=x0.unsqueeze(0).expand(1,-1,-1).contiguous()
xBb=xB[0:1].contiguous()

# 逐段：A0 通路（复用 proto_m4_final 里已算好的量不现实，这里重算关键段）
def seg_rotfield():
    a=L.face_areas(x0b,faces,1)
    J=torch.stack([torch.stack([xB[0:1][...,b,:]-xB[0:1][...,a2,:] for b,a2 in [(0,0)]],0)],0)  # placeholder
    return a
# 直接测量推荐配置的调制器本体（从 Jf 之后的聚合开始算）
def rsb_block():
    g=((coh-tau_)/(1-tau_)).clamp(0,1)**1.5
    th=qang(qR)*180/math.pi
    t=torch.where(th>0,(45.0/th.clamp(min=1e-9)),torch.ones_like(th)).clamp(max=1.0)
    dq=slerp_id(qR,t*g)
    return dq
tau_=0.30
print("推荐配置 RSB 方向块（从 qR/coh 到 delta_q，含修正级）: %.4f ms"%bench(rsb_block))
def full_corr():
    g=((coh-tau_)/(1-tau_)).clamp(0,1)**1.5
    th=qang(qR)*180/math.pi
    t=torch.where(th>0,(45.0/th.clamp(min=1e-9)),torch.ones_like(th)).clamp(max=1.0)
    dq=slerp_id(qR,t*g)
    u=rotv(dq,n0v)
    cr=torch.cross(u,nv,dim=-1); sn=cr.norm(dim=-1)
    thn=torch.atan2(sn,(u*nv).sum(-1)); ax=cr/sn.clamp(min=1e-9).unsqueeze(-1)
    ncapr=25.0*math.pi/180; thn_s=ncapr*torch.tanh(thn/ncapr)
    ad=g*t; half=ad*thn_s*0.5
    dqfix=torch.cat([torch.cos(half).unsqueeze(-1),torch.sin(half).unsqueeze(-1)*ax],-1)
    dqfix=torch.where((sn<1e-8).unsqueeze(-1)|(ad==0).unsqueeze(-1),EI.expand_as(dqfix),dqfix)
    return qmul(qmul(dqfix,dq),q0)
print("同一块 + 法向修正级 + 两次 qmul（不含 apply 的 normalize）: %.4f ms"%bench(full_corr))
print("=> 增量 = %.4f ms"%(bench(full_corr)-bench(rsb_block)))
