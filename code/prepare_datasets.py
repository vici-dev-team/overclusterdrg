"""
Download and preprocess Shuttle and Covertype datasets into the project's
.py list-of-lists format (z-score standardised, no label column).
"""
import urllib.request
import gzip
import io
import os
import numpy as np

OUT_DIR = "."

# ── helpers ──────────────────────────────────────────────────────────────────

def standardise(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def save_as_py(X, path):
    rows = X.tolist()
    with open(path, "w") as f:
        f.write("[\n")
        for i, row in enumerate(rows):
            comma = "," if i < len(rows) - 1 else ""
            f.write(f"    {row}{comma}\n")
        f.write("]\n")
    print(f"  Saved {len(rows)} rows x {X.shape[1]} cols  →  {path}")

# ── Shuttle ───────────────────────────────────────────────────────────────────
# 9 numeric features + 1 class label (last col); rows from train + test splits

def build_shuttle():
    out = os.path.join(OUT_DIR, "shuttle_dataset.py")
    if os.path.exists(out):
        print(f"[shuttle] already exists, skipping")
        return

    print("[shuttle] downloading …")
    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/"
    chunks = []
    for fname in ("shuttle.trn.Z", "shuttle.tst"):
        url = base + fname
        print(f"  fetching {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                raw = r.read()
        except Exception as e:
            raise RuntimeError(f"Download failed for {url}: {e}")

        # shuttle.trn.Z is Unix-compressed (.Z); shuttle.tst is plain text
        if fname.endswith(".Z"):
            # Python stdlib has no .Z decompressor; try zlib raw or fallback
            try:
                import subprocess, tempfile
                with tempfile.NamedTemporaryFile(suffix=".Z", delete=False) as tmp:
                    tmp.write(raw); tmp_path = tmp.name
                result = subprocess.run(["uncompress", "-c", tmp_path],
                                        capture_output=True)
                if result.returncode != 0:
                    raise RuntimeError("uncompress failed")
                text = result.stdout.decode()
                os.unlink(tmp_path)
            except Exception:
                # fallback: download the plain-text mirror on openml
                print("  .Z decompression unavailable; fetching plain-text mirror …")
                url2 = "https://www.openml.org/data/get_csv/4965243/shuttle.arff"
                with urllib.request.urlopen(url2, timeout=60) as r:
                    arff_text = r.read().decode()
                # skip ARFF header
                lines = [l for l in arff_text.splitlines()
                         if l and not l.startswith("@") and not l.startswith("%")]
                rows = [[float(v) for v in l.split(",")] for l in lines if l.strip()]
                arr = np.array(rows, dtype=np.float64)
                X = standardise(arr[:, :-1])   # drop class label
                save_as_py(X, out)
                return
        else:
            text = raw.decode()

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        rows = [[float(v) for v in l.split()] for l in lines]
        chunks.extend(rows)

    arr = np.array(chunks, dtype=np.float64)
    X = standardise(arr[:, :-1])   # drop class label (last col)
    save_as_py(X, out)

# ── Covertype (first 10 continuous features, subsample to 30 k) ───────────────

def build_covertype(n_keep=30000):
    out = os.path.join(OUT_DIR, "covertype_dataset.py")
    if os.path.exists(out):
        print(f"[covertype] already exists, skipping")
        return

    print("[covertype] downloading (≈11 MB gzip) …")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    with urllib.request.urlopen(url, timeout=120) as r:
        raw = r.read()

    print("  decompressing …")
    with gzip.open(io.BytesIO(raw)) as gz:
        text = gz.read().decode()

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    rows = [[float(v) for v in l.split(",")] for l in lines]
    arr = np.array(rows, dtype=np.float64)

    # first 10 columns are continuous; last column is class label
    X_cont = arr[:, :10]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_cont), size=min(n_keep, len(X_cont)), replace=False)
    idx.sort()
    X = standardise(X_cont[idx])
    save_as_py(X, out)

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    build_shuttle()
    build_covertype()
    print("\nDone. New dataset files ready.")
