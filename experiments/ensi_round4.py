#!/usr/bin/env python3
"""
ENSI round 4 (no enlarged simulation; trade pools regenerated deterministically with the
same simulate_E_cell routine, 2M i.i.d. trades per cell).
Part 1  exact identification check of q = P(L > b(T,m)) from filtered complete paths
        (GPT counterexample T=2, m=2; and the proposed sufficient condition T >= (m+1) l_min).
Part 2  near-non-identification: q0, q_eps, inverse-weight moments, tail moments of X,
        for T+1 in {126, 251, 501} and m in {1, 10, 20, 30, 40}.
Usage: python ensi_round4.py out_v03
"""
import sys, os, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sr_stopping_v03 import CFG, grid, simulate_E_cell

out = sys.argv[1] if len(sys.argv) > 1 else "out_v03"
lines = []
def say(s=""):
    print(s); lines.append(str(s))

# ------------------------------------------------------------------ Part 1
def filtered_path_law(pL, T, m, nmax=12):
    """Exact law of the observed L-path (L_1..L_N, N) given N_T >= m (X ignored).
    pL: dict l -> prob. Paths are enumerated as sequences; the last (incomplete) event
    contributes only through survival P(L > T - S_N)."""
    sup = sorted(pL)
    surv = lambda t: sum(p for l, p in pL.items() if l > t)
    law = {}
    def rec(prefix, S, pr):
        n = len(prefix)
        stop = surv(T - S)                       # next event does not finish
        if n >= m and stop > 0:
            law[tuple(prefix)] = law.get(tuple(prefix), 0) + pr * stop
        if n >= nmax: return
        for l in sup:
            if S + l <= T:
                rec(prefix + [l], S + l, pr * pL[l])
    rec([], 0, 1.0)
    Z = sum(law.values())
    return {k: v / Z for k, v in law.items()}

def tv(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)

say("== Part 1: identification of q from filtered complete paths")
P0 = {1: 0.9, 3: 0.1}; P1 = {1: 0.1, 3: 0.9}
say(f"GPT counterexample T=2, m=2:  TV distance of observed laws = {tv(filtered_path_law(P0,2,2), filtered_path_law(P1,2,2)):.3e}"
    f"   (q0=0.1 vs q1=0.9)")
say("Same two populations, other (T, m):")
for T, m in [(3, 2), (4, 2), (4, 3), (5, 3), (5, 4)]:
    lmin = 1; cond = T >= (m + 1) * lmin
    say(f"  T={T}, m={m}, T>=(m+1)l_min: {cond!s:5}  TV = {tv(filtered_path_law(P0,T,m), filtered_path_law(P1,T,m)):.3e}")
# systematic: for random 3-point laws, can two laws with different q give the same filtered law?
# check local injectivity via numerical Jacobian rank of the observed law w.r.t. (p1,p2,p3) on the simplex
say("Local identifiability (Jacobian rank of observed law wrt free params of a 3-point law L in {1,2,5}):")
rng = np.random.default_rng(3)
for T, m in [(2, 2), (3, 2), (4, 2), (4, 3), (6, 3)]:
    ranks = []
    for _ in range(20):
        p = rng.dirichlet(np.ones(3)); base = filtered_path_law({1: p[0], 2: p[1], 5: p[2]}, T, m)
        keys = sorted(base); J = []
        for j in range(2):                                   # 2 free params: move mass from atom 2 (l=5) to j
            d = np.zeros(3); d[j] += 1e-6; d[2] -= 1e-6; pp = p + d
            nl = filtered_path_law({1: pp[0], 2: pp[1], 5: pp[2]}, T, m)
            J.append([(nl.get(k, 0) - base[k]) / 1e-6 for k in keys])
        ranks.append(np.linalg.matrix_rank(np.array(J), tol=1e-6))
    b = T - (m - 1)
    say(f"  T={T}, m={m}, b={b}, T>=(m+1): {T >= m + 1!s:5}  Jacobian rank (need 2 for full ID): {sorted(set(ranks))}")

# ------------------------------------------------------------------ Part 2
def cdf_sums(pL, tmax, kmax):
    p = np.zeros(tmax + 1); p[:min(len(pL), tmax + 1)] = pL[:tmax + 1]
    d = np.zeros(tmax + 1); d[0] = 1.0; F = [np.cumsum(d)]
    for _ in range(kmax):
        d = np.convolve(d, p)[:tmax + 1]; F.append(np.cumsum(d))
    return np.array(F)

say("\n== Part 2: near-non-identification in the V0.3 first-exit model (L = tau + 1, l_min = 2)")
TPf, SLf, sig = grid(CFG)
rng = np.random.default_rng(np.random.SeedSequence([CFG["master_seed"], 8]))
rows = []
for k in [9, 19, 44, 99]:
    X, tau = simulate_E_cell(TPf[k], SLf[k], sig, 2_000_000, rng)
    L = tau + 1; Lmax = int(L.max())
    cnt = np.bincount(L, minlength=Lmax + 1).astype(float); pL = cnt / cnt.sum()
    for Tp in [126, 251, 501]:
        F = cdf_sums(pL, Tp, 41)
        for m in [1, 10, 20, 30, 40]:
            if F[m][Tp] < 1e-12: continue
            # w_m(l) = F_{m-1}(Tp - l) for l <= Tp, 0 beyond
            wl = np.zeros(Lmax + 1)
            idx = np.arange(1, min(Lmax, Tp) + 1); wl[idx] = F[m - 1][Tp - idx]
            w = wl[L]
            row = dict(cell=k, TP_sigma=TPf[k] / sig, SL_sigma=SLf[k] / sig, T_plus_1=Tp, m=m,
                       b=Tp - 2 * (m - 1), P_pass=F[m][Tp], q0=float(np.mean(w == 0)))
            for eps in [1e-2, 1e-3, 1e-4]:
                row[f"q_eps_{eps:g}"] = float(np.mean(w <= eps))
            pos = w > 0
            v = F[m][Tp] / w[pos]                        # inverse weight on observable support
            # moments under the observed (filtered) law Q: E_Q[f] = E_P[f w]/F_m
            EQ_v = float(np.mean(w[pos] * v) / F[m][Tp]) if pos.any() else np.nan       # = P(w>0)
            EQ_v2 = float(np.mean(w[pos] * v**2) / F[m][Tp]) if pos.any() else np.nan
            row.update(EQ_v=EQ_v, EQ_v2=EQ_v2, kish_ess_ratio=EQ_v**2 / EQ_v2 if EQ_v2 else np.nan,
                       v_q99_under_Q=float(np.quantile(v, 0.99)) if pos.any() else np.nan)
            for lab, reg in [("unobs", w == 0), ("near_1e-3", (w > 0) & (w <= 1e-3))]:
                if reg.any():
                    row[f"{lab}_mass"] = float(reg.mean())
                    row[f"{lab}_EX_over_sig"] = float(X[reg].mean() / sig)
                    row[f"{lab}_EX2_over_sig2"] = float((X[reg]**2).mean() / sig**2)
                    row[f"{lab}_contrib_EX_over_sig"] = float((X * reg).mean() / sig)
                else:
                    row[f"{lab}_mass"] = 0.0
            row["X_min_over_sig"] = float(X.min() / sig); row["X_max_over_sig"] = float(X.max() / sig)
            row["EX2_over_sig2"] = float((X**2).mean() / sig**2)
            rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, "ensi_round4_near_nonid.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
show = ["cell", "T_plus_1", "m", "b", "P_pass", "q0", "q_eps_0.01", "q_eps_0.001", "q_eps_0.0001",
        "EQ_v2", "kish_ess_ratio", "v_q99_under_Q", "unobs_contrib_EX_over_sig", "near_1e-3_mass"]
say(df[show].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
open(os.path.join(out, "ensi_round4.txt"), "w").write("\n".join(lines))