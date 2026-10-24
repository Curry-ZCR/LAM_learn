
import sys, math, json
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
exec(open("/workspace/lam3d/scripts/proto_m4_final.py").read().split("VAR={")[0])
g=((coh-0.30)/0.70).clamp(0,1)**1.5
th=qang(qR)*180/math.pi
t=torch.where(th>0,(45.0/th.clamp(min=1e-9)),torch.ones_like(th)).clamp(max=1.0)
dq=slerp_id(qR,t*g)
u=rotv(dq,n0v)
cs=(u*nv).sum(-1); sn=torch.cross(u,nv,dim=-1).norm(dim=-1)
thn=torch.atan2(sn,cs)*180/math.pi
ad=g*t*G_dir
print("=== 修正角 thn = angle(rot(dq)·n0_hat, n1_hat) 分位 (deg) ===")
for q in (0.1,0.25,0.5,0.75,0.9,0.99):
    print("  p%-4s %7.3f"%(int(q*100), float(torch.quantile(thn.reshape(-1),q))))
print("  >5deg %.2f%%  >15deg %.2f%%  >25deg %.2f%%"%(
  100*float((thn>5).float().mean()),100*float((thn>15).float().mean()),100*float((thn>25).float().mean())))
print("  有效修正角 ad*thn_s: p50=%.4f p90=%.3f p99=%.3f deg"%(
  float((ad*thn_s).median()) if 'thn_s' in dir() else -1,0,0)) if False else None
ncapr=math.pi*25/180
thn_s=ncapr*torch.tanh(thn*math.pi/180/ncapr)*180/math.pi
print("  有效修正角 ad*thn_s: p50=%.4f p90=%.3f p99=%.3f deg"%(
  float((ad*thn_s).median()),float(torch.quantile((ad*thn_s).reshape(-1),.9)),float(torch.quantile((ad*thn_s).reshape(-1),.99))))
print()
print("=== 门控覆盖率 ===")
for tau in (0.0,0.3,0.5,0.7,0.9):
    gg=((coh-tau)/(1-tau)).clamp(0,1)**1.5
    print("  tau=%.1f  g==0 占 %.2f%%  g>0.5 占 %.2f%%"%(tau,100*float((gg<=1e-6).float().mean()),100*float((gg>0.5).float().mean())))
print()
print("=== 关键量程 ===")
print("  coh p0.1=%.4f p1=%.4f p5=%.4f p50=%.5f"%(torch.quantile(coh.reshape(-1),.001),torch.quantile(coh.reshape(-1),.01),torch.quantile(coh.reshape(-1),.05),coh.median()))
print("  aniso p50=%.3f p90=%.1f p99=%.1f G_amp p50=%.4f G_dir p50=%.4f"%(aniso.median(),torch.quantile(aniso,.9),torch.quantile(aniso,.99),G_amp.median(),G_dir.median()))
print("  G_rho_new p1=%.3e p50=%.3f p99=%.4f"%(torch.quantile(G_rho_new,.01),G_rho_new.median(),torch.quantile(G_rho_new,.99)))
print("  s0 min axis p50=%.3e (clamp 1e-9 必需)"%(s0.min(-1).values.median()))
