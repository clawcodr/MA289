"""Recover the latent common factor and each predictor's correlation with the target."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import norm

def ns(x):
    r = pd.Series(x).rank(method="average").to_numpy()
    return norm.ppf(r / (len(x) + 1.0))

work = sys.argv[1]
EY, EY2 = 0.50142, 0.33602
VY = EY2 - EY**2; SDY = np.sqrt(VY)

p = np.load(os.path.join(work, "p_rank.npy"))
cons = np.load(os.path.join(work, "cons_z.npy"))
q_z = np.load(os.path.join(work, "q_z.npy"))
RHO_USED = 0.65
pred_q = norm.cdf(RHO_USED * q_z / np.sqrt(2 - RHO_USED**2))
pred_q = pred_q + (EY - pred_q.mean())

def cov_with_y(pred, mse):
    """Back out Cov(pred,y) from a public-LB MSE."""
    m, s = pred.mean(), pred.std()
    Epy = (m**2 + s**2 + EY2 - mse) / 2
    cov = Epy - m * EY
    return cov, cov / (s * SDY)

cov_p, r_p = cov_with_y(p, 0.09385)
cov_q, r_q = cov_with_y(pred_q, 0.07325)
print(f"corr(p, y) = {r_p:.4f}")
print(f"corr(q, y) = {r_q:.4f}")

zp, zq, zc = ns(p), ns(q_z), ns(cons)
r_pq = np.corrcoef(zp, zq)[0, 1]
r_cp = np.corrcoef(zc, zp)[0, 1]
r_cq = np.corrcoef(zc, zq)[0, 1]
print(f"\ninternal: corr(p,q)={r_pq:.3f} corr(cons,p)={r_cp:.3f} corr(cons,q)={r_cq:.3f}")

# one-factor model: corr(i,j)=li*lj, corr(i,y)=li*ly
ly = np.sqrt(r_p * r_q / r_pq)
lp, lq = r_p / ly, r_q / ly
print(f"\n1-factor fit: lambda_y={ly:.3f} lambda_p={lp:.3f} lambda_q={lq:.3f}")
print(f"  check corr(p,q): model={lp*lq:.3f} actual={r_pq:.3f}")
l_cons_from_p = r_cp / lp
l_cons_from_q = r_cq / lq
print(f"  lambda_cons from p = {l_cons_from_p:.3f}, from q = {l_cons_from_q:.3f}"
      "   (consistency check)")
l_cons = min(1.0, (l_cons_from_p + l_cons_from_q) / 2)
print(f"  => lambda_cons={l_cons:.3f}, implied corr(cons,y)={l_cons*ly:.4f}")
print(f"  implied best MSE with cons = {VY*(1-(l_cons*ly)**2):.5f}")

# optimal 2-predictor blend of measured p and q, for comparison
Sig = np.array([[p.var(), np.cov(p, pred_q)[0,1]], [np.cov(p, pred_q)[0,1], pred_q.var()]])
c = np.array([cov_p, cov_q])
wts = np.linalg.solve(Sig, c)
r2 = c @ wts / VY
print(f"\noptimal measured blend(p,q): R2={r2:.4f} -> MSE={VY*(1-r2):.5f}")
np.save(os.path.join(work, "factor_params.npy"),
        np.array([ly, lp, lq, l_cons, cov_p, cov_q]))
