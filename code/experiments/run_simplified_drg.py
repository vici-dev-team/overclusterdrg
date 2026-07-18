"""Compare SimplifiedDRG vs OC-Local (O=z) vs OC-Forward."""
import sys,os,ast,time
import numpy as np, pandas as pd
sys.path.insert(0,'/Users/rudrabhardwaj/Downloads/K-Centre')
os.chdir('/Users/rudrabhardwaj/Downloads/K-Centre')
import drg as drg_mod, drg1

def oc_forward(X,k,z):
    centers,_=drg_mod.overcluster_drg(X,k,O=z)
    return drg_mod.robust_radius(X,centers,z)

def oc_local(X,k,z):
    _,Q=drg_mod.overcluster_drg(X,k,O=z)
    c=drg1.local_search_fast(X,Q,k,z)
    return drg_mod.robust_radius(X,c,z)

def simplified_drg(X,k,z):
    """SimplifiedDRG: Gonzalez + outlier-aware step selection."""
    r,_,_=drg_mod.simplified_drg(X,k,O=z)
    return r

CONFIGS=[(500,10,.10),(500,20,.20),(1000,10,.10),(1000,20,.20),(1000,50,.10),
         (2000,10,.10),(2000,20,.20),(5000,10,.10),(5000,20,.20),(5000,50,.10)]
rows=[]
for ds,fname in [('adult','adult_final_dataset.py'),
                 ('diabetes','diabetes_dataset.py'),('covertype','covertype_dataset.py')]:
    with open(fname) as f:
        X_full=np.asarray(ast.literal_eval(f.read()),dtype=np.float32)
    for N,k,op in CONFIGS:
        if N>len(X_full): continue
        X=X_full[:N]; z=int(op*N)
        t0=time.perf_counter(); sdrg=simplified_drg(X,k,z); sdrg_t=time.perf_counter()-t0
        t0=time.perf_counter(); ocf=oc_forward(X,k,z);   ocf_t=time.perf_counter()-t0
        t0=time.perf_counter(); ocl=oc_local(X,k,z);     ocl_t=time.perf_counter()-t0
        print(f"{ds} N={N} k={k} z={z}: SDRG={sdrg:.4f}({sdrg_t:.3f}s) "
              f"OC-Fwd={ocf:.4f}({ocf_t:.3f}s) OC-Local={ocl:.4f}({ocl_t:.3f}s)",flush=True)
        rows.append({'dataset':ds,'N':N,'k':k,'z':z,'op':op,
                     'sdrg_r':round(sdrg,6),'sdrg_t':round(sdrg_t,4),
                     'ocf_r':round(ocf,6),'ocf_t':round(ocf_t,4),
                     'ocl_r':round(ocl,6),'ocl_t':round(ocl_t,4)})
        pd.DataFrame(rows).to_csv('paper/experiments/simplified_drg_results.csv',index=False)
print("Done.")
