import os, sys, glob, numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final2 import normal_scores, z, group_of, cov_with_y, EY, EY2, VY, SDY

work = sys.argv[1]
G = {"neural": [], "matrix": []}
for sub in ("out2", "out3", "out4", "out5", "out6", "out7"):
    for f in sorted(glob.glob(os.path.join(work, sub, "*.npz"))):
        g = group_of(f)
        if g: G[g].append(normal_scores(np.load(f)["cos"]))

def plain(lst): return z(normal_scores(np.mean(lst, axis=0)))
def loaded(lst):
    A = np.array([z(x) for x in lst])
    C = np.corrcoef(A)
    w_, v = np.linalg.eigh(C)
    L = v[:, -1] * np.sign(v[:, -1].sum())
    return z(normal_scores((np.clip(L, 0, None)[:, None] * A).sum(0)))

for name, fn in (("plain", plain), ("loaded", loaded)):
    n, m = fn(G["neural"]), fn(G["matrix"])
    comp = z(normal_scores(0.536 * n + 0.464 * m))
    np.save(os.path.join(work, f"comp_{name}.npy"), comp)
    print(f"{name}: corr(neural,matrix)={np.corrcoef(n,m)[0,1]:.3f}")

a = np.load(os.path.join(work, "comp_plain.npy"))
b = np.load(os.path.join(work, "comp_loaded.npy"))
print(f"\ncorr(plain composite, loading-weighted composite) = {np.corrcoef(a,b)[0,1]:.4f}")
