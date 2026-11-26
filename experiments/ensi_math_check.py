#!/usr/bin/env python3
"""
ENSI math cross-check (no new large simulation).
1. Brute-force check of GPT's counterexample.
2. Brute-force check of the identity  E[Xbar_{N_T}; N_T >= m] = E[X w_m(L)]  on small discrete laws.
3. For V0.3 Gaussian first-exit trades (cells 5, 8, 9, 19, 44): estimate h(l)=E[X|L=l] and
   p(l)=P(L=l) from 2M i.i.d. trades, check monotonicity / log-concavity, evaluate the exact
   formula B_m(T+1) and compare with the V0.2/V0.3 window simulation (same test paths).
Usage: python ensi_math_check.py out_v02/raw_test.parquet out_v03
"""
import sys, os, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sr_stopping_v03 import CFG, grid, simulate_E_cell

out = sys.argv[2] if len(sys.argv) > 2 else "out_v03"
lines = []
def say(s=""):
    print(s); lines.append(s)

# ---------------------------------------------------------------- helpers for discrete laws
def cdf_sums(pL, tmax, kmax):
    """F[k][t] = P(S_k <= t) for t=0..tmax, k=0..kmax; pL indexed by l=0..len-1 (pL[0]=0)."""
    p = np.zeros(tmax + 1); p[:min(len(pL), tmax + 1)] = pL[:tmax + 1]
    dens = np.zeros(tmax + 1); dens[0] = 1.0           # law of S_0
    F = [np.cumsum(dens)]
    for _ in range(kmax):
        dens = np.convolve(dens, p)[:tmax + 1]
        F.append(np.cumsum(dens))
    return np.array(F)

def B_exact(support, T, mmax):
    """support: list of (l, x, prob). Returns B_m = E[X w_m(L)] / F_m(T) for m=1..mmax."""
    lmax = max(l for l, _, _ in support)
    pL = np.zeros(lmax + 1); EXl = np.zeros(lmax + 1)
    for l, x, pr in support:
        pL[l] += pr; EXl[l] += pr * x
    F = cdf_sums(pL, T, mmax)
    res = {}
    for m in range(1, mmax + 1):
        if F[m][T] <= 0: break
        num = sum(EXl[l] * F[m - 1][T - l] for l in range(1, min(lmax, T) + 1))
        res[m] = num / F[m][T]
    return res

def brute(support, T, mmax, nmax):
    """Enumerate cycle sequences; return E[X1|N>=m] and E[Xbar_N|N>=m]."""
    acc = {m: [0.0, 0.0, 0.0] for m in range(1, mmax + 1)}   # P, E[X1;.], E[Xbar;.]
    for seq in itertools.product(support, repeat=nmax):
        pr = np.prod([s[2] for s in seq]); S = 0; N = 0
        for (l, _, _) in seq:
            if S + l <= T: S += l; N += 1
            else: break
        assert N < nmax, "increase nmax"
        if N == 0: continue
        xbar = np.mean([s[1] for s in seq[:N]])
        for m in range(1, min(N, mmax) + 1):
            acc[m][0] += pr; acc[m][1] += pr * seq[0][1]; acc[m][2] += pr * xbar
    return {m: (a[1] / a[0], a[2] / a[0]) for m, a in acc.items() if a[0] > 0}

# ---------------------------------------------------------------- 1. counterexample
say("== 1. GPT counterexample: (L,X) in {(1,-1),(2,4),(3,-3)} w.p. 1/3, T=2")
cx = [(1, -1.0, 1/3), (2, 4.0, 1/3), (3, -3.0, 1/3)]
EX = sum(x * p for _, x, p in cx); EL = sum(l * p for l, _, p in cx); ELX = sum(l * x * p for l, x, p in cx)
say(f"E[X]={EX:.4f}  Cov(L,X)={ELX-EL*EX:.4f}")
bf = brute(cx, 2, 2, 4); ex = B_exact(cx, 2, 2)
for m in (1, 2):
    say(f"m={m}: brute E[X1|N>=m]={bf[m][0]:.4f}  brute E[Xbar|N>=m]={bf[m][1]:.4f}  formula={ex[m]:.4f}")

# ---------------------------------------------------------------- 2. identity on random small laws
say("\n== 2. Identity E[Xbar_N | N>=m] = E[X1 | N>=m] = E[X w_m(L)]/F_m(T): random discrete laws")
rng = np.random.default_rng(1)
worst = 0.0
for trial in range(30):
    k = rng.integers(2, 4)
    ls = rng.integers(1, 4, size=k); xs = rng.normal(size=k); ps = rng.dirichlet(np.ones(k))
    xs = xs - np.dot(xs, ps)                                  # E[X]=0
    sup = list(zip(ls.tolist(), xs.tolist(), ps.tolist()))
    T = int(rng.integers(3, 6))
    bf = brute(sup, T, 3, T + 1); ex = B_exact(sup, T, 3)
    for m in bf:
        worst = max(worst, abs(bf[m][0] - ex[m]), abs(bf[m][1] - ex[m]))
say(f"30 random laws, m<=3, T in 3..5: max |brute - formula| over X1 and Xbar = {worst:.2e}")

# ---------------------------------------------------------------- 3. Gaussian first-exit cells
say("\n== 3. V0.3 Gaussian first-exit model (L = tau + 1, completed count = N_{T+1}, T+1 = 251)")
raw = pd.read_parquet(sys.argv[1], columns=["path", "cell", "n_complete", "total_pnl", "forced_close_pnl"])
TPf, SLf, sig = grid(CFG); Tp = CFG["T"] + 1
rng = np.random.default_rng(np.random.SeedSequence([CFG["master_seed"], 4]))
ms = [1, 2, 5, 10, 20, 30]
rows = []
for k in [5, 8, 9, 19, 44]:
    X, tau = simulate_E_cell(TPf[k], SLf[k], sig, 2_000_000, rng)
    L = tau + 1
    lmax = int(L.max())
    cnt = np.bincount(L, minlength=lmax + 1).astype(float)
    sx = np.bincount(L, weights=X, minlength=lmax + 1); sx2 = np.bincount(L, weights=X**2, minlength=lmax + 1)
    pL = cnt / cnt.sum(); EXl = sx / cnt.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        h = sx / cnt; se = np.sqrt((sx2 / cnt - h**2) / cnt)
    # monotonicity of h on l with >= 2000 obs: count significant increases between adjacent l
    good = np.where(cnt >= 2000)[0]
    inc = [(a, b) for a, b in zip(good[:-1], good[1:]) if b == a + 1 and h[b] - h[a] > 3 * np.hypot(se[a], se[b])]
    # log-concavity of p on l with >= 2000 obs: second difference of log p
    lp = np.log(pL[good]); d2 = lp[2:] - 2 * lp[1:-1] + lp[:-2]
    d2_se = np.sqrt(1 / cnt[good][2:] + 4 / cnt[good][1:-1] + 1 / cnt[good][:-2])
    viol = int(np.sum(d2 > 3 * d2_se))
    F = cdf_sums(pL, Tp, max(ms))
    r = raw[raw.cell == k]
    n = r.n_complete.to_numpy(); S1 = (r.total_pnl - r.forced_close_pnl).to_numpy().astype(float)
    for m in ms:
        num = sum(EXl[l] * F[m - 1][Tp - l] for l in range(1, min(lmax, Tp) + 1))
        pred = num / F[m][Tp]
        sel = n >= m
        xbar = S1[sel] / n[sel]
        rows.append(dict(cell=k, TP_sigma=TPf[k] / sig, SL_sigma=SLf[k] / sig, m=m,
                         P_N_ge_m_formula=F[m][Tp], P_N_ge_m_sim=sel.mean(), paths=int(sel.sum()),
                         B_formula=pred, Xbar_sim_mean=xbar.mean() if sel.any() else np.nan,
                         Xbar_sim_se=xbar.std(ddof=1) / np.sqrt(sel.sum()) if sel.sum() > 1 else np.nan))
    say(f"cell {k:2d} TP={TPf[k]/sig:g}σ SL={SLf[k]/sig:g}σ : E[X]={X.mean():+.5f}  corr(L,X)={np.corrcoef(L,X)[0,1]:+.3f}  "
        f"h(l) significant increases={len(inc)} (first: {inc[:3]})  log-concavity violations={viol}/{len(d2)}  "
        f"h at l=2,5,10,20,40: {np.round(h[[2,5,10,20,40]] / sig, 2).tolist()} (σ units)")
df = pd.DataFrame(rows)
df["z"] = (df.Xbar_sim_mean - df.B_formula) / df.Xbar_sim_se
df.to_csv(os.path.join(out, "ensi_exact_vs_sim.csv"), index=False)
say("\nExact formula vs window simulation (10,000 test paths):")
say(df.round(5).to_string(index=False))
open(os.path.join(out, "ensi_math_check.txt"), "w").write("\n".join(lines))