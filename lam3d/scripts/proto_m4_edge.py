
import sys, math, time, torch
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
import torch.nn.functional as F
exec(open("/workspace/lam3d/scripts/proto_m4_final.py").read().split("VAR={")[0])

area0=L.face_areas(x0,faces,1).reshape(-1)
print("静止面积: min=%.3e  frac(==0)=%.3e  frac(<1e-12)=%.3e"%(
   float(area0.min()), float((area0==0).float().mean()), float((area0<1e-12).float().mean())))
# 孤立顶点
cnt=torch.zeros(N,dtype=torch.long,device=faces.device)
for k in range(3): cnt.index_add_(0,faces[:,k],torch.ones(Fn,dtype=torch.long,device=faces.device))
print("未被任何面引用的顶点数 = %d ; 仅1个面引用 = %d"%(int((cnt==0).sum()),int((cnt==1).sum())))
# 零面积面对 J / polar 的污染检查
bad=(area0<=0)
if int(bad.sum())>0:
    Jb=Jf[:,bad]; print("零面积面 J 有限=%s  max|J|=%.3e"%(bool(torch.isfinite(Jb).all()), float(Jb.abs().max())))
else:
    print("capture 中无零面积面；仍需在实现里加 where(area<=eps, I, q_face) 硬化（0*nan=nan 风险）")
# dispatch 开销标定：单个小算子的耗时
def bench(fn,it=200,w=30):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
a=torch.randn(20018,3,device="cuda")
print()
print("dispatch 标定（N=20018 小算子，单次 ms）:")
print("  cross            %.4f"%bench(lambda: torch.cross(a,a,dim=-1)))
print("  atan2            %.4f"%bench(lambda: torch.atan2(a[:,0],a[:,1])))
print("  10x cross        %.4f"%bench(lambda:[torch.cross(a,a,dim=-1) for _ in range(10)]))
print("  => 每 ~13us/op，说明调制器是 dispatch-bound 而非 compute-bound")
