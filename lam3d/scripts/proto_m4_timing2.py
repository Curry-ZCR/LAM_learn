
import sys, math, time
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch, torch.nn.functional as F
import proto_m4_lib as L
torch.set_grad_enabled(False); dev="cuda"
x0,xB,faces,q0,s0,names=L.load()
M,N=5,x0.shape[0]; Fn=faces.shape[0]
area=L.face_areas(x0.unsqueeze(0),faces,1)
Jf1=L.face_jacobians(x0.unsqueeze(0),xB[0:1],faces)[0]
while Jf1.dim()>4: Jf1=Jf1[0]
fa,fb,fc=faces.unbind(-1)
A=L.face_areas(x0.unsqueeze(0),faces,1)
Ap=0.5*torch.cross(xB[0][fb]-xB[0][fa],xB[0][fc]-xB[0][fa],dim=-1).norm(dim=-1).view(1,Fn)
lr=torch.log((Ap/A).clamp(1e-3,1e3))
flat=faces.reshape(-1)
def sc_generic(per_face, C):
    vals=(per_face*area[...,None]).repeat_interleave(3,1)
    idx=flat.view(1,-1,1).expand(1,-1,C)
    out=torch.zeros(1,N,C,device=dev); out.scatter_add_(1,idx,vals)
    den=torch.zeros(1,N,device=dev); den.scatter_add_(1,flat.view(1,-1).expand(1,-1),area.repeat_interleave(3,1))
    return out/den.clamp(min=1e-12).unsqueeze(-1)
J9=Jf1.reshape(1,Fn,9); J10=torch.cat([J9,lr[...,None]],-1)
def t(fn,n=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e3
for tag,pt,C in (("9ch (仅 Jv)",J9,9),("10ch (Jv+logk 融合)",J10,10),
                 ("1ch (仅 logk)",lr[...,None],1)):
    print("  scatter %-20s %6.4f ms"% (tag, t(lambda pt=pt,C=C: sc_generic(pt,C))))
print("  9ch + 1ch 分开          %6.4f ms"%t(lambda:(sc_generic(J9,9),sc_generic(lr[...,None],1))))
