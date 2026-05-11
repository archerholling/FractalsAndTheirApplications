
"""

Fractal and Stochastic Analysis of Silver Price Dynamics
Archer Holling – RMIT Honours Thesis 2026

"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yfinance as yf
from scipy.stats import kstest, jarque_bera
from statsmodels.tsa.stattools import acf
from fbm import FBM


# 1) Configuration
OUTDIR = "outputs/silver"
os.makedirs(OUTDIR, exist_ok=True)

START = os.getenv("START", "2000-08-30")
END   = os.getenv("END",   "2026-04-23")  # date of analysis in thesis

CANDIDATES = [
    "SI=F",      # COMEX futures (continuous)
]

INTERVAL = "1d"        
FREQ_RESAMPLE = None  

# Rolling Hurst settings
ROLLING_WINDOW = 252 * 3  # ~3 years of daily data
ROLLING_STEP   = 21       # compute every ~month

# DFA/MF-DFA scales and q-range
SCALES = np.unique(np.logspace(1.2, 3, 20).astype(int))  # ~16..1000
Q_RANGE = np.linspace(-4, 4, 17)  # includes 0 as special case

# -----------------------------
# 2) Utility Functions
# -----------------------------
def safe_log_returns(series: pd.Series) -> pd.Series:
    """Compute log returns, drop inf/NaN."""
    r = np.log(series).diff()
    return r.replace([np.inf, -np.inf], np.nan).dropna()

def aggregate_returns(returns: pd.Series, m: int) -> pd.Series:
    """Aggregate log returns over non-overlapping blocks of length m."""
    n = len(returns) // m
    r = returns.iloc[: n * m]
    agg = r.values.reshape(n, m).sum(axis=1)
    idx = r.index[::m][:n]
    return pd.Series(agg, index=idx)

# -----------------------------
# 3) Hurst via Rescaled Range (R/S)
# -----------------------------
def hurst_rs(series: pd.Series, window_sizes=None, min_size=16):
    """
    Estimate H via classic rescaled range analysis.
    Returns (H, df) where df has columns ['n','RS','log_n','log_RS'].
    """
    x = series.values.astype(float)
    N = len(x)

    if window_sizes is None:
        window_sizes = np.unique(
            np.logspace(np.log10(min_size), np.log10(max(min_size + 1, N / 4)), 20).astype(int)
        )

    rows = []
    for n in window_sizes:
        if n < min_size or n > N // 2:
            continue

        K = N // n
        RS_vals = []

        for k in range(K):
            seg = x[k * n:(k + 1) * n]
            seg = seg - np.mean(seg)
            y = np.cumsum(seg)
            R = np.max(y) - np.min(y)
            S = np.std(seg, ddof=1)

            if S > 0:
                RS_vals.append(R / S)

        if RS_vals:
            RS = np.mean(RS_vals)
            rows.append((n, RS, np.log(n), np.log(RS)))

    df = pd.DataFrame(rows, columns=["n", "RS", "log_n", "log_RS"])
    H = np.polyfit(df["log_n"], df["log_RS"], 1)[0] if len(df) >= 2 else np.nan
    return H, df


# -----------------------------
# 4) Detrended Fluctuation Analysis (DFA, order 1)
# -----------------------------
def dfa(series: pd.Series, scales=SCALES, order=1):
    """DFA to estimate H (for fGn: H ~ slope). Returns (H, df)."""
    x = series.values.astype(float)
    x = x - np.mean(x)
    y = np.cumsum(x)  # profile

    Fs = []
    for s in scales:
        if s < 8 or s >= len(y):
            continue

        n_segments = len(y) // s
        if n_segments < 2:
            continue

        rms = []
        t = np.arange(s)

        for i in range(n_segments):
            seg = y[i * s:(i + 1) * s]
            coeff = np.polyfit(t, seg, order)
            trend = np.polyval(coeff, t)
            detrended = seg - trend
            rms.append(np.sqrt(np.mean(detrended**2)))

        F_s = np.sqrt(np.mean(np.array(rms)**2))
        Fs.append((s, F_s, np.log(s), np.log(F_s)))

    df = pd.DataFrame(Fs, columns=["s", "F", "log_s", "log_F"])
    H = np.polyfit(df["log_s"], df["log_F"], 1)[0] if len(df) >= 2 else np.nan
    return H, df

# -----------------------------
# 5) Multifractal DFA (MF-DFA) - basic implementation
# -----------------------------
def mfdfa(series: pd.Series, q_vals=Q_RANGE, scales=SCALES, order=1):
    """Basic MF-DFA. Returns hq, tau_q, alpha, f_alpha, qs."""
    x = series.values.astype(float)
    x = x - np.mean(x)
    y = np.cumsum(x)

    Fqs = {q: [] for q in q_vals if q != 0}
    F0s = []  # q -> 0 (log-average)

    for s in scales:
        if s < 8 or s >= len(y):
            continue

        n_segments = len(y) // s
        if n_segments < 2:
            continue

        rms_list = []
        t = np.arange(s)

        for i in range(n_segments):
            seg = y[i * s:(i + 1) * s]
            coeff = np.polyfit(t, seg, order)
            trend = np.polyval(coeff, t)
            detrended = seg - trend
            rms_list.append(np.sqrt(np.mean(detrended**2)))

        rms_arr = np.array(rms_list)
        rms_arr = rms_arr[rms_arr > 0]
        if len(rms_arr) == 0:
            continue

        for q in Fqs.keys():
            Fq = (np.mean(rms_arr**q)) ** (1.0 / q)
            Fqs[q].append((s, Fq))

        F0s.append((s, np.exp(np.mean(np.log(rms_arr)))))  # q -> 0

    hq = {}
    for q, pairs in Fqs.items():
        if len(pairs) < 2:
            hq[q] = np.nan
            continue
        s_vals = np.array([p[0] for p in pairs])
        Fq_vals = np.array([p[1] for p in pairs])
        hq[q] = np.polyfit(np.log(s_vals), np.log(Fq_vals), 1)[0]

    if len(F0s) >= 2:
        s_vals0 = np.array([p[0] for p in F0s])
        F0_vals = np.array([p[1] for p in F0s])
        hq[0.0] = np.polyfit(np.log(s_vals0), np.log(F0_vals), 1)[0]
    else:
        hq[0.0] = np.nan

    tau_q = {q: (q * hq[q] - 1) if not np.isnan(hq[q]) else np.nan for q in hq.keys()}
    qs = np.array(sorted(hq.keys()))
    tau_vals = np.array([tau_q[q] for q in qs])

    alpha = np.gradient(tau_vals, qs)
    f_alpha = qs * alpha - tau_vals

    return hq, tau_q, alpha, f_alpha, qs


# -----------------------------
# 6) Simulations: BM and fBM
# -----------------------------
def simulate_bm_like(returns: pd.Series, n_steps: int, seed=42):
    """Simulate GBM-like path using empirical mu, sigma on log returns."""
    np.random.seed(seed)
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    z = np.random.normal(0, 1, size=n_steps)
    r_sim = mu + sigma * z
    prices = np.exp(np.cumsum(np.insert(r_sim, 0, 0.0)))
    return pd.Series(prices), pd.Series(r_sim)

def simulate_fbm_like(returns: pd.Series, n_steps: int, H=0.7, seed=43):
    """Simulate fBM increments as log returns, then exponentiate to get price path."""
    np.random.seed(seed)
    sigma = np.std(returns, ddof=1)
    fbm_gen = FBM(n=n_steps-1, hurst=H, length=1.0, method="daviesharte")
    path = fbm_gen.fbm()
    fgn = np.diff(path)
    r_sim = sigma * (fgn / np.std(fgn, ddof=1))
    prices = np.exp(np.cumsum(np.insert(r_sim, 0, 0.0)))
    return pd.Series(prices), pd.Series(r_sim)

# -----------------------------
# 7) Robust Download Helper
# -----------------------------
def download_silver_series():
    """Try multiple tickers & methods; return a single best-effort price series."""
    price_frames, worked, failed = [], [], []
    for t in CANDIDATES:
        print(f"Trying {t} ...")
        df = None
        try:
            df = yf.download(t, start=START, end=END, interval=INTERVAL, progress=False)
            if (not isinstance(df, pd.DataFrame)) or df.empty:
                raise RuntimeError("Empty from yf.download")
        except Exception:
            # Fallback: Ticker.history(period='max') sometimes works when download fails
            try:
                hist = yf.Ticker(t).history(period="max", interval="1d", auto_adjust=False)
                df = hist if isinstance(hist, pd.DataFrame) and not hist.empty else None
            except Exception as e2:
                failed.append((t, f"{type(e2).__name__}"))
                continue

        if df is None or df.empty:
            failed.append((t, "No data"))
            continue

        col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if not col:
            failed.append((t, "No Close/Adj Close column"))
            continue

        s = df[col].copy().dropna()
        if s.empty:
            failed.append((t, "No price data after dropna"))
            continue

        s.name = t
        
    # ensure we always append a DataFrame
        if isinstance(s, pd.DataFrame):
            price_frames.append(s)
        else:
            price_frames.append(s.to_frame())

        worked.append(t)

        if not price_frames:
             raise RuntimeError(f"No data downloaded for any candidate tickers. Failures: {failed}")

    print("Downloaded:", worked)
    if failed:
        print("Skipped:", failed)

    # Combine into single series preferring spot, then futures, then ETFs
    df_all = pd.concat(price_frames, axis=1)
    order = ["XAGUSD=X", "SI=F", "SLV", "SIVR"]
    series = None
    for name in order:
        if name in df_all.columns:
            series = df_all[name] if series is None else series.combine_first(df_all[name])
    if series is None:
        series = df_all.select_dtypes(include=[float, int]).iloc[:, 0]
    series = series.sort_index().dropna()
    series.name = "SilverUSD_like"
    return series

# -----------------------------
# 8) Main Workflow
# -----------------------------
def main():
    # ---- Download data (robust) ----
    silver = download_silver_series()
    silver = silver[silver.index >= "2000-08-30"]

    print("Silver starts:", silver.index.min())
    print("Silver ends:", silver.index.max())
    print("Last 5 rows of silver:")
    print(silver.tail())

    if FREQ_RESAMPLE:
        silver = silver.resample(FREQ_RESAMPLE).last()

    print("Saving silver_prices.csv to:")
    print(os.path.abspath(os.path.join(OUTDIR, "silver_prices.csv")))

    silver.to_csv(os.path.join(OUTDIR, "silver_prices.csv"))

    # ---- Returns ----
    r = safe_log_returns(silver)
    r.name = "log_return"
    r.to_csv(os.path.join(OUTDIR, "silver_log_returns.csv"))

    # ---- H via R/S ----
    H_rs, df_rs = hurst_rs(r, window_sizes=None, min_size=16)
    df_rs.to_csv(os.path.join(OUTDIR, "hurst_rs.csv"), index=False)

    # ---- H via DFA ----
    H_dfa, df_dfa = dfa(r, scales=SCALES, order=1)
    df_dfa.to_csv(os.path.join(OUTDIR, "hurst_dfa.csv"), index=False)

    D_from_rs = 2 - H_rs if not np.isnan(H_rs) else np.nan
    D_from_dfa = 2 - H_dfa if not np.isnan(H_dfa) else np.nan

    # ---- Rolling H (DFA) ----
    rolling_H, idxs = [], []
    for start in range(0, len(r) - ROLLING_WINDOW, ROLLING_STEP):
        sub = r.iloc[start:start + ROLLING_WINDOW]
        if len(sub) < max(16, int(SCALES.min())):
            continue
        H_sub, _ = dfa(sub, scales=SCALES, order=1)
        rolling_H.append(H_sub)
        idxs.append(sub.index[-1])
    rolling_H = pd.Series(rolling_H, index=idxs, name="H_rolling_dfa")
    rolling_H.to_csv(os.path.join(OUTDIR, "rolling_hurst_dfa.csv"))

    # ---- MF-DFA ----
    hq, tau_q, alpha, f_alpha, qs = mfdfa(r, q_vals=Q_RANGE, scales=SCALES, order=1)
    q_table = pd.DataFrame({
        "q": qs,
        "h_q": [hq[q] for q in qs],
        "tau_q": [tau_q[q] for q in qs],
        "alpha": alpha,
        "f_alpha": f_alpha
    })

    q_table = q_table.round({
        "q": 2,
        "h_q": 4,
        "tau_q": 4,
        "alpha": 4,
        "f_alpha": 4
    })

    q_table.to_csv(os.path.join(OUTDIR, "q_table.csv"), index=False)
    q_table.to_excel(os.path.join(OUTDIR, "q_table.xlsx"), index=False)

    with open(os.path.join(OUTDIR, "q_table.tex"), "w", encoding="utf-8") as f:
        f.write(q_table.to_latex(index=False, float_format="%.4f"))

    print("Saved q_table.csv, q_table.xlsx, and q_table.tex")

    q_keep = [-4, -2, -1, 0, 1, 2, 4]
    q_table_small = q_table[q_table["q"].isin(q_keep)].copy()

    q_table_small.to_csv(os.path.join(OUTDIR, "q_table_small.csv"), index=False)
    q_table_small.to_excel(os.path.join(OUTDIR, "q_table_small.xlsx"), index=False)

    with open(os.path.join(OUTDIR, "q_table_small.tex"), "w", encoding="utf-8") as f:
        f.write(q_table_small.to_latex(index=False, float_format="%.4f"))

    print("Saved q_table_small.csv, q_table_small.xlsx, and q_table_small.tex")
    
    # ---- Simulations ----
    n_steps = len(r)
    bm_price, bm_ret = simulate_bm_like(r, n_steps=n_steps, seed=42)
    fbm_price, fbm_ret = simulate_fbm_like(r, n_steps=n_steps, H=max(min(H_dfa, 0.99), 0.01), seed=43)

    # ---- Distributional tests ----
    jb_emp = jarque_bera(r.values)
    ks_emp = kstest((r - r.mean()) / r.std(ddof=1), "norm")

    jb_bm  = jarque_bera(bm_ret.values)
    ks_bm  = kstest((bm_ret - bm_ret.mean()) / bm_ret.std(ddof=1), "norm")

    jb_fbm = jarque_bera(fbm_ret.values)
    ks_fbm = kstest((fbm_ret - fbm_ret.mean()) / fbm_ret.std(ddof=1), "norm")

    with open(os.path.join(OUTDIR, "distribution_tests.txt"), "w") as f:
        f.write("Jarque-Bera (stat, p-value):\n")
        f.write(f"Empirical: {jb_emp}\nBM: {jb_bm}\nFBM: {jb_fbm}\n\n")
        f.write("KS against Normal (stat, p-value):\n")
        f.write(f"Empirical: {ks_emp}\nBM: {ks_bm}\nFBM: {ks_fbm}\n")

    # ---- ACF & Variance Scaling ----
    max_lag = 60
    acf_emp = acf(r, nlags=max_lag, fft=True)
    acf_bm  = acf(bm_ret, nlags=max_lag, fft=True)
    acf_fbm = acf(fbm_ret, nlags=max_lag, fft=True)
    pd.DataFrame(
        {"lag": np.arange(len(acf_emp)), "emp": acf_emp, "bm": acf_bm, "fbm": acf_fbm}
    ).to_csv(os.path.join(OUTDIR, "acf_compare.csv"), index=False)

    # variance vs aggregation scale m
    ms = np.unique(np.logspace(0, 2, 15).astype(int))  # 1..100
    rows = []
    for m in ms:
        rows.append(("empirical", m, np.var(aggregate_returns(r, m), ddof=1)))
        rows.append(("bm", m, np.var(aggregate_returns(bm_ret, m), ddof=1)))
        rows.append(("fbm", m, np.var(aggregate_returns(fbm_ret, m), ddof=1)))
    pd.DataFrame(rows, columns=["series", "m", "var"]).to_csv(
        os.path.join(OUTDIR, "variance_scaling.csv"), index=False
    )

    # ----------------------
    # 9) Plots
    # ----------------------
    print("Silver series starts:", silver.index.min())
    print("Silver series ends:", silver.index.max())
    plt.figure(figsize=(11, 5))
    ax = plt.gca()

    silver.plot(ax=ax, color='black', linewidth=1.0)

    ax.set_title(
    f"Daily Silver Price (USD), {silver.index.min().year}–{silver.index.max().year}",
    fontsize=12,
    fontweight='bold'
)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price", fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "01_price_series.png"), dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(11, 5))
    ax = plt.gca()

    r.plot(ax=ax, color='#2166ac', linewidth=0.8)

    ax.set_title(
        f"Silver Log Returns, {r.index.min().year}–{r.index.max().year}",
        fontsize=12,
        fontweight='bold'
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Log Return", fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "02_log_returns.png"), dpi=200, bbox_inches='tight')
    plt.close()

        # R/S log-log
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.scatter(
        df_rs["log_n"], df_rs["log_RS"],
        s=55, color="#2166ac", label="Data points", zorder=3
    )

    b_rs = np.polyfit(df_rs["log_n"], df_rs["log_RS"], 1)[1]
    xline_rs = np.linspace(df_rs["log_n"].min(), df_rs["log_n"].max(), 200)
    yline_rs = H_rs * xline_rs + b_rs

    ss_res_rs = np.sum((df_rs["log_RS"] - (H_rs * df_rs["log_n"] + b_rs)) ** 2)
    ss_tot_rs = np.sum((df_rs["log_RS"] - df_rs["log_RS"].mean()) ** 2)
    r2_rs = 1 - ss_res_rs / ss_tot_rs

    ax.plot(
        xline_rs, yline_rs,
        color="#d73027",
        linewidth=2.0,
        label=f"y = {H_rs:.4f}x + {b_rs:.4f}\n$R^2$ = {r2_rs:.4f}",
        zorder=2
    )

    ax.set_title("R/S Scaling (Global) – Silver Daily Log Returns",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel(r"$\log_{10}(N)$", fontsize=14)
    ax.set_ylabel(r"$\log_{10}(E[R/S])$", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "03_rs_scaling.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved 03_rs_scaling.png")


    # DFA log-log
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.scatter(
        df_dfa["log_s"], df_dfa["log_F"],
        s=55, color="#2166ac", label="Data points", zorder=3
    )

    b_dfa = np.polyfit(df_dfa["log_s"], df_dfa["log_F"], 1)[1]
    xline_dfa = np.linspace(df_dfa["log_s"].min(), df_dfa["log_s"].max(), 200)
    yline_dfa = H_dfa * xline_dfa + b_dfa

    ss_res_dfa = np.sum((df_dfa["log_F"] - (H_dfa * df_dfa["log_s"] + b_dfa)) ** 2)
    ss_tot_dfa = np.sum((df_dfa["log_F"] - df_dfa["log_F"].mean()) ** 2)
    r2_dfa = 1 - ss_res_dfa / ss_tot_dfa

    ax.plot(
        xline_dfa, yline_dfa,
        color="#d73027",
        linewidth=2.0,
        label=f"y = {H_dfa:.4f}x + {b_dfa:.4f}\n$R^2$ = {r2_dfa:.4f}",
        zorder=2
    )

    ax.set_title("DFA Scaling (Global) – Silver Daily Log Returns",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel(r"$\log_{10}(s)$", fontsize=14)
    ax.set_ylabel(r"$\log_{10}(F(s))$", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "04_dfa_scaling.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved 04_dfa_scaling.png")

    # Rolling H (DFA)
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        rolling_H.index, rolling_H.values,
        color="#2166ac",
        linewidth=1.2,
        label="Rolling Hurst Exponent (DFA)",
        zorder=3
    )

    ax.axhline(
        0.5,
        color="black",
        linewidth=1.0,
        linestyle="--",
        label="H = 0.5 benchmark",
        zorder=2
    )

    ax.set_title(
        f"Rolling Hurst Exponent (DFA) – Silver ({rolling_H.index.min().year}–{rolling_H.index.max().year})",
        fontsize=12,
        fontweight="bold"
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Hurst Exponent H", fontsize=11)

    ax.set_ylim(0.22, 0.58)
    ax.set_yticks([0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55])

    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "05_rolling_hurst.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved 05_rolling_hurst.png")

    # Simulated vs empirical (prices & returns)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    ax[0].plot(np.arange(len(silver)), silver / silver.iloc[0], label="Empirical")
    ax[0].plot(np.arange(len(bm_price)), bm_price, label="BM")
    ax[0].plot(np.arange(len(fbm_price)), fbm_price, label="fBM")
    ax[0].set_title("Normalised Price Paths")
    ax[0].legend()

    ax[1].plot(r.values, label="Empirical")
    ax[1].plot(bm_ret.values, label="BM")
    ax[1].plot(fbm_ret.values, label="fBM")
    ax[1].set_title("Log Returns")
    ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "06_simulated_compare.png")); plt.close()

    # ACF comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stem(np.arange(len(acf_emp)), acf_emp, markerfmt=" ", basefmt=" ")
    ax.stem(np.arange(len(acf_bm)), acf_bm, markerfmt=" ", basefmt=" ")
    ax.stem(np.arange(len(acf_fbm)), acf_fbm, markerfmt=" ", basefmt=" ")
    ax.set_title("ACF comparison (Emp, BM, fBM)")
    ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "07_acf_compare.png")); plt.close()

    # Variance scaling plot (log-log)
    vs = pd.read_csv(os.path.join(OUTDIR, "variance_scaling.csv"))
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, sub in vs.groupby("series"):
        ax.scatter(np.log(sub["m"]), np.log(sub["var"]), label=name, s=20)
    ax.set_xlabel("log(m)")
    ax.set_ylabel("log(Var[Σ r_t])")
    ax.legend()
    ax.set_title("Variance scaling with aggregation scale")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "08_variance_scaling.png")); plt.close()

 # MF-DFA spectrum
    spec = pd.read_csv(os.path.join(OUTDIR, "mfdfo_spectrum.csv"))   # change filename if needed

    # Sort by alpha so the spectrum is drawn cleanly left-to-right
    spec = spec.sort_values("alpha").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        spec["alpha"],
        spec["f_alpha"],
        color="#2166ac",
        linewidth=1.8,
        zorder=3
    )

    ax.set_title(
        "Multifractal Spectrum (MF-DFA) – Silver Daily Log Returns",
        fontsize=12,
        fontweight="bold"
    )
    ax.set_xlabel(r"$\alpha$", fontsize=11)
    ax.set_ylabel(r"$f(\alpha)$", fontsize=11)

    ax.grid(axis="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "09_mfdfa_spectrum.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved 09_mfdfa_spectrum.png")

    # Summary
    with open(os.path.join(OUTDIR, "summary.txt"), "w") as f:
        f.write(f"H (R/S): {H_rs:.4f} | Fractal D: {2 - H_rs if not np.isnan(H_rs) else np.nan:.4f}\n")
        f.write(f"H (DFA): {H_dfa:.4f} | Fractal D: {2 - H_dfa if not np.isnan(H_dfa) else np.nan:.4f}\n")
        f.write("See distribution_tests.txt, *_csv files and figures in outputs/.\n")

    print("Done. Outputs written to:", OUTDIR)

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    main()
