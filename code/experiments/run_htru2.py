"""Download HTRU2 dataset and run full comparison."""
import sys,os,ast,time,urllib.request,zipfile,io
import numpy as np, pandas as pd
sys.path.insert(0,'/Users/rudrabhardwaj/Downloads/K-Centre')
os.chdir('/Users/rudrabhardwaj/Downloads/K-Centre')
import drg as drg_mod, drg1
from gonzalez import gonzalez_standard
from rkc import charikar_kcenter_outliers

def oc_local(X,k,z):
    _,Q=drg_mod.overcluster_drg(X,k,O=z)
    c=drg1.local_search_fast(X,Q,k,z)
    return drg_mod.robust_radius(X,c,z)

# Download HTRU2
htru_path='htru2_dataset.py'
if not os.path.exists(htru_path):
    print("Downloading HTRU2...")
    url='https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip'
    with urllib.request.urlopen(url,timeout=60) as r:
        data=r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z_file:
        fname=[n for n in z_file.namelist() if n.endswith('.csv')][0]
        with z_file.open(fname) as f:
            import csv
            rows_raw=list(csv.reader(f))
    # standardise numeric features (all 8 cols, last is label)
    X_raw=np.array([[float(v) for v in r[:8]] for r in rows_raw if len(r)>=8],dtype=np.float64)
    mu,sd=X_raw.mean(0),X_raw.std(0); sd[sd==0]=1
    X_std=((X_raw-mu)/sd).tolist()
    with open(htru_path,'w') as f:
        f.write('[\n')
        for i,r in enumerate(X_std):
            f.write(f"    {r}{',' if i<len(X_std)-1 else ''}\n")
        f.write(']\n')
    print(f"Saved {len(X_std)} rows → {htru_path}")

with open(htru_path) as f:
    X_full=np.asarray(ast.literal_eval(f.read()),dtype=np.float32)
print(f"HTRU2: {X_full.shape}")

CONFIGS=[(500,10,.10),(500,20,.20),(1000,10,.10),(1000,20,.20),(1000,50,.10),
         (2000,10,.10),(2000,20,.20),(5000,10,.10),(5000,20,.20),(5000,50,.10),
         (10000,10,.10),(10000,20,.20),(17000,10,.10),(17000,20,.20)]
rows=[]
for N,k,op in CONFIGS:
    if N>len(X_full): continue
    X=X_full[:N]; z=int(op*N)
    t0=time.perf_counter(); oc_r=oc_local(X,k,z); oc_t=time.perf_counter()-t0
    t0=time.perf_counter(); g_r=gonzalez_standard(X,k,z,seed=42); g_t=time.perf_counter()-t0
    t0=time.perf_counter(); c_r=charikar_kcenter_outliers(X,k,z); c_t=time.perf_counter()-t0
    print(f"N={N} k={k} z={z}: OC={oc_r:.4f}({oc_t:.2f}s) Gonz={g_r:.4f}({g_t:.4f}s) "
          f"Char3r*={3*c_r:.4f}({c_t:.2f}s)",flush=True)
    rows.append({'N':N,'k':k,'z':z,'op':op,'oc_r':round(oc_r,6),'oc_t':round(oc_t,4),
                 'gonz_r':round(g_r,6),'gonz_t':round(g_t,6),
                 'char_rstar':round(c_r,6),'char_3r':round(3*c_r,6),'char_t':round(c_t,4)})
    pd.DataFrame(rows).to_csv('paper/experiments/htru2_results.csv',index=False)
print("Done.")
