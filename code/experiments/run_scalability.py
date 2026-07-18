"""Scalability: N up to 100K. OC-Local (O=z) vs Ding vs Gonzalez."""
import sys,os,ast,time,math
import numpy as np, pandas as pd
sys.path.insert(0,'/Users/rudrabhardwaj/Downloads/K-Centre')
os.chdir('/Users/rudrabhardwaj/Downloads/K-Centre')
import drg as drg_mod, drg1
from gonzalez import gonzalez_standard
import dingvOC as ding_mod

def oc_local(X,k,z):
    _,Q=drg_mod.overcluster_drg(X,k,O=z)
    c=drg1.local_search_fast(X,Q,k,z)
    return drg_mod.robust_radius(X,c,z)

SIZES=[500,1000,2000,5000,10000,20000,50000,99000]
rows=[]
for ds,fname in [('diabetes','diabetes_dataset.py'),('covertype','covertype_dataset.py')]:
    with open(fname) as f:
        X_full=np.asarray(ast.literal_eval(f.read()),dtype=np.float32)
    for N in [s for s in SIZES if s<=len(X_full)]:
        X=X_full[:N]; k=20; z=int(0.10*N); T=math.ceil(math.log(0.01)/math.log(1-1/(z+1)))
        print(f"{ds} N={N} k={k} z={z} T={T}",flush=True)
        t0=time.perf_counter(); oc_r=oc_local(X,k,z); oc_t=time.perf_counter()-t0
        t0=time.perf_counter(); g_r=gonzalez_standard(X,k,z,seed=42); g_t=time.perf_counter()-t0
        if N<=20000:
            t0=time.perf_counter(); d_r,d_t,_,_=ding_mod.ding_et_al(X,k,z,seed=42); d_t=time.perf_counter()-t0
        else:
            d_r,d_t=None,None
        print(f"  OC={oc_r:.4f}({oc_t:.2f}s) Gonz={g_r:.4f}({g_t:.3f}s)"
              +(" Ding=N/A (too slow)" if d_r is None else f" Ding={d_r:.4f}({d_t:.1f}s)"),flush=True)
        rows.append({'dataset':ds,'N':N,'k':k,'z':z,'oc_r':round(oc_r,6),'oc_t':round(oc_t,4),
                     'gonz_r':round(g_r,6),'gonz_t':round(g_t,6),
                     'd_r':round(d_r,6) if d_r else None,'d_t':round(d_t,2) if d_t else None})
        pd.DataFrame(rows).to_csv('paper/experiments/scalability_results.csv',index=False)
print("Done.")
