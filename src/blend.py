"""Factor-analytic consensus across all model families, anchored on the probed predictor."""
import os, sys, glob, numpy as np, pandas as pd
from scipy.stats import norm

def ns(x):
    r = pd.Series(x).rank(method="average").to_numpy()
    return norm.ppf(r / (len(x) + 1.0))

work = sys.argv[1]
names, Z = [], []
for sub in ("out2", "out3", "out4"):
    for f in sorted(glob.glob(os.path.join(work, sub, "*.npz"))):
        names.append(sub + "/" + os.path.basename(f)[:-4])
        Z.append(ns(np.load(f)["cos"]))
Z = np.array(Z)
Z = (Z - Z.mean(1, keepdims=True)) / Z.std(1, keepdims=True)
print("models:", len(names))

fam = lambda n: ("ffnn" if n.startswith("out2") else
                 ("svd" if "/svd" in n or "/ppmifull" in n else "sgns"))
C = np.corrcoef(Z)
# first principal component across models = shared signal
w, v = np.linalg.eigh(C)
load = v[:, -1] * np.sign(v[:, -1].sum())
print(f"PC1 explains {w[-1]/len(names):.3f} of variance")

p = np.load(os.path.join(work, "p_rank.npy"))        # probed predictor (rank-uniform)
zp = ns(p); zp = (zp - zp.mean()) / zp.std()
R_PY = 0.441                                          # measured corr(p, y) from probe 3

print("\ntop PC1 loadings:")
idx = np.argsort(-load)
for i in idx[:12]:
    print(f"  {load[i]:.3f}  {names[i]}")
print("bottom:")
for i in idx[-5:]:
    print(f"  {load[i]:.3f}  {names[i]}")

print("\nmean |corr| with probed p, by family:")
for f in ("ffnn", "svd", "sgns"):
    sel = [i for i, n in enumerate(names) if fam(n) == f]
    cs = [np.corrcoef(Z[i], zp)[0, 1] for i in sel]
    print(f"  {f:5s} n={len(sel):3d}  corr(model,p) mean={np.mean(cs):.3f} max={np.max(cs):.3f}")

# consensus = PC1-weighted average, built per family then averaged to avoid family-size bias
cons_fam = {}
for f in ("ffnn", "svd", "sgns"):
    sel = [i for i, n in enumerate(names) if fam(n) == f]
    wts = np.clip(load[sel], 0, None)
    cons_fam[f] = ns((wts[:, None] * Z[sel]).sum(0))
for f in cons_fam:
    print(f"corr(cons_{f}, p) = {np.corrcoef(cons_fam[f], zp)[0,1]:.3f}")
print("family consensus mutual corr:")
fs = list(cons_fam)
for i, a in enumerate(fs):
    for b in fs[i+1:]:
        print(f"  {a}-{b}: {np.corrcoef(cons_fam[a], cons_fam[b])[0,1]:.3f}")

cons = ns(sum(cons_fam.values()))
cons = (cons - cons.mean()) / cons.std()
r_cons_p = np.corrcoef(cons, zp)[0, 1]
print(f"\ncorr(grand consensus, p) = {r_cons_p:.3f}")
print(f"implied corr(consensus, y) ~= {R_PY / r_cons_p:.3f}  (if p informs y only via shared signal)")
np.save(os.path.join(work, "cons_z.npy"), cons)
for f in cons_fam:
    np.save(os.path.join(work, f"cons_{f}.npy"), cons_fam[f])
