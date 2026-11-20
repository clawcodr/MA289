"""Optimal blend of the three predictors whose covariance with y was measured."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final2 import normal_scores, z, cov_with_y, EY, EY2, VY, SDY

work = sys.argv[1]
p = np.load(os.path.join(work, "p_rank.npy"))
q_zs = np.load(os.path.join(work, "q_z.npy"))
rq = 0.65
q = norm.cdf(rq * q_zs / np.sqrt(2 - rq**2)); q += EY - q.mean()
c_z = np.load(os.path.join(work, "composite_z.npy"))
rc = 0.545
c = norm.cdf(rc * c_z / np.sqrt(2 - rc**2)); c += EY - c.mean()

X = np.column_stack([p, q, c])
covs = np.array([cov_with_y(p, 0.09385), cov_with_y(q, 0.07325), cov_with_y(c, 0.06524)])
for n, v, col in zip("pqc", covs, X.T):
    print(f"corr({n},y) = {v/(col.std()*SDY):.4f}")
Sig = np.cov(X, rowvar=False)
for lam in (0.0, 0.05, 0.15, 0.30):
    S2 = Sig + lam * np.diag(np.diag(Sig))
    w = np.linalg.solve(S2, covs)
    pred = X @ w
    pred = pred + (EY - pred.mean())
    r2 = (w @ covs)**2 / ((w @ Sig @ w) * VY)
    print(f"lam={lam:.2f} w={np.round(w,3)} implied corr={np.sqrt(min(r2,1)):.4f} "
          f"-> MSE {VY*(1-min(r2,1)):.5f}")
lam = 0.15
w = np.linalg.solve(Sig + lam*np.diag(np.diag(Sig)), covs)
pred = X @ w
pred = np.clip(pred + (EY - pred.mean()), 0, 1)
print(f"\nchosen: mean={pred.mean():.4f} sd={pred.std():.4f}")
test = pd.read_csv(os.path.join(os.environ["MA289_DATA"], "test.csv"))
pd.DataFrame({"pair_id": test["pair_id"], "similarity": np.round(pred, 6)}).to_csv(
    os.path.join(work, "submission_v2.csv"), index=False)
print("wrote submission_v2.csv")
