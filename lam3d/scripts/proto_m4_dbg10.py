
import json
d=json.load(open("/tmp/m4_eval.json"))
names=["ones","half","extreme","rand","rand_sparse"]
print("%-22s"%"variant"+"".join("%12s"%n for n in names))
for tag in ("A0 rot-only (abar=0)","A3 RSB full"):
    r=d[tag]["per"]
    print("%-22s"%tag+"".join("%12.3f"%r[k]["rigid"] for k in range(5)))
    print("%-22s"%("  rigid90")+"".join("%12.3f"%r[k]["rigid90"] for k in range(5)))
    print("%-22s"%("  shape50")+"".join("%12.4f"%r[k]["shape"] for k in range(5)))
    print("%-22s"%("  thin50")+"".join("%12.2f"%r[k]["thin"] for k in range(5)))
    print("%-22s"%("  jitter")+"".join("%12.3f"%r[k]["jitter"] for k in range(5)))
    print("%-22s"%("  false%")+"".join("%12.2f"%(100*r[k]["fm"]) for k in range(5)))
