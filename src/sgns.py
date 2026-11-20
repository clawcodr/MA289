"""Skip-gram with negative sampling (word2vec style) in PyTorch."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, pair_cos

def make_pairs(corpus, w, rng, dynamic=True, subsample=None, freq=None, total=None):
    a, b = [], []
    for line in corpus:
        arr = np.asarray(line)
        if subsample:
            p = freq[arr] / total
            keep = (np.sqrt(p / subsample) + 1) * (subsample / p)
            arr = arr[rng.random(len(arr)) < np.minimum(keep, 1.0)]
        n = len(arr)
        if n < 2:
            continue
        for off in range(1, w + 1):
            if dynamic:
                m = rng.random(n - off) < (1.0 - (off - 1) / w)
                x, y = arr[:n-off][m], arr[off:][m]
            else:
                x, y = arr[:n-off], arr[off:]
            a.append(x); b.append(y); a.append(y); b.append(x)
    return np.concatenate(a), np.concatenate(b)

def train(cfg, D):
    V = D["V"]
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    freq = np.bincount(np.concatenate([np.asarray(l) for l in D["corpus"]]), minlength=V).astype(np.float64)
    total = freq.sum()
    noise = torch.tensor(freq ** 0.75 / (freq ** 0.75).sum(), dtype=torch.float)
    d, k = cfg["d"], cfg["neg"]
    Win = torch.nn.Embedding(V, d); Wout = torch.nn.Embedding(V, d)
    torch.nn.init.uniform_(Win.weight, -0.5 / d, 0.5 / d)
    torch.nn.init.zeros_(Wout.weight)
    opt = torch.optim.Adam(list(Win.parameters()) + list(Wout.parameters()), lr=cfg["lr"])
    bs = 4096
    for ep in range(cfg["epochs"]):
        c, o = make_pairs(D["corpus"], cfg["w"], rng, True, cfg.get("sub"), freq, total)
        perm = rng.permutation(len(c))
        c, o = torch.tensor(c[perm]), torch.tensor(o[perm])
        for i in range(0, len(c), bs):
            cb, ob = c[i:i+bs], o[i:i+bs]
            nb = torch.multinomial(noise, len(cb) * k, replacement=True).view(len(cb), k)
            v = Win(cb)
            pos = torch.sum(v * Wout(ob), -1)
            neg = torch.bmm(Wout(nb), v.unsqueeze(2)).squeeze(2)
            loss = -(torch.nn.functional.logsigmoid(pos).mean()
                     + torch.nn.functional.logsigmoid(-neg).sum(1).mean())
            opt.zero_grad(); loss.backward(); opt.step()
    return Win.weight.detach().numpy(), Wout.weight.detach().numpy()

if __name__ == "__main__":
    D = load(); outdir = sys.argv[1]
    for w in (2, 5, 10):
        for d in (50, 100):
            for seed in (0, 1, 2):
                cfg = dict(w=w, d=d, seed=seed, neg=5, lr=0.005, epochs=5, sub=1e-3)
                Ein, Eout = train(cfg, D)
                for name, E in (("in", Ein), ("sum", Ein + Eout)):
                    c = pair_cos(E, D["t1"], D["t2"])
                    tag = f"sgns_w{w}_d{d}_s{seed}_{name}"
                    np.savez_compressed(os.path.join(outdir, tag + ".npz"), cos=c)
                print("done", w, d, seed, flush=True)
