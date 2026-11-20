"""Softmax FFNN (lesson architecture) on full-corpus pairs with wider windows."""
import os, sys, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, pair_cos

def pairs_from_corpus(corpus, w):
    a, b = [], []
    for line in corpus:
        arr = np.asarray(line); n = len(arr)
        for off in range(1, w + 1):
            a.append(arr[:n-off]); b.append(arr[off:])
            a.append(arr[off:]);   b.append(arr[:n-off])
    return np.concatenate(a), np.concatenate(b)

if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    D = load(); V = D["V"]
    c, o = pairs_from_corpus(D["corpus"], cfg["w"])
    torch.manual_seed(cfg["seed"])
    emb = nn.Embedding(V, cfg["d"]); out = nn.Linear(cfg["d"], V)
    opt = torch.optim.Adam(list(emb.parameters()) + list(out.parameters()), lr=cfg["lr"])
    crit = nn.CrossEntropyLoss()
    ct = torch.tensor(c); ot = torch.tensor(o)
    N = len(ct); bs = 4096
    g = torch.Generator().manual_seed(cfg["seed"])
    for ep in range(1, cfg["epochs"] + 1):
        perm = torch.randperm(N, generator=g)
        tot = 0.0
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = crit(out(emb(ct[idx])), ot[idx])
            loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if ep in cfg["snapshots"]:
            E = emb.weight.detach().numpy().copy()
            tag = f"ffnnw_w{cfg['w']}_d{cfg['d']}_s{cfg['seed']}_e{ep}"
            np.savez_compressed(os.path.join(cfg["outdir"], tag + ".npz"),
                                cos=pair_cos(E, D["t1"], D["t2"]))
            print(tag, f"loss={tot/N:.4f}", flush=True)
