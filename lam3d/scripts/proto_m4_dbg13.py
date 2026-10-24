
import sys
sys.path.insert(0,"/workspace/lam3d"); sys.path.insert(0,"/workspace/lam3d/scripts")
exec(open("/workspace/lam3d/scripts/proto_m4_final.py").read().split("VAR={")[0])
print("Jf",Jf.shape,"area",area.shape,"Jv",Jv.shape,"qR",qR.shape,"n0v",n0v.shape,"nv",nv.shape)
print("coh",coh.shape, "q0",q0.shape, "s0",s0.shape)
