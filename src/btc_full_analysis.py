"""
Full Bitcoin Fractal Analysis
Archer Holling – RMIT Honours Thesis 2026

Combines:
  1. Data download / loading         (from btc_price.py)
  2. R/S scaling table               (from E_RS.py)
  3. Global & rolling Hurst (R/S)    (from test.py)
  4. Structural breakpoint analysis  (new – btc_breakpoint.py)

Outputs (all saved to OUTPUT_DIR):
  BTC-USD_closing.xlsx               – raw price + log return data
  BTC_RS_Table.xlsx                  – R/S scaling table
  fig_global_rs.png                  – global R/S log-log plot  (Figure 4)
  fig_rolling_hurst.png              – rolling Hurst exponent   (Figure 5)
  fig_rolling_hurst_breakpoints.png  – rolling H + breakpoints  (Figure 15)
  fig_regime_boxplot.png             – per-regime H distribution(Figure 16)
  fig_pre_post_event.png             – pre/post event analysis  (Figure 17)
  BTC_breakpoint_results.xlsx        – full breakpoint numerics

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path

OUTPUT_DIR = Path("outputs/btc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TICKER      = "BTC-USD"
START_DATE  = "2015-01-01"
END_DATE    = None                

ROLLING_WINDOW = 365             # days for rolling Hurst window
ROLLING_STEP   = 7               # re-estimate every N days

# SECTION 1 – DATA DOWNLOAD / LOAD

xlsx_path = OUTPUT_DIR / "BTC-USD_closing.xlsx"

if not xlsx_path.exists():
    raise FileNotFoundError(
        f"BTC-USD_closing.xlsx not found in {OUTPUT_DIR}. Put the file there and re-run."
    )

print(f"[1] Found existing data at {xlsx_path} – loading Excel file.")
df = pd.read_excel(xlsx_path)

df = df.sort_values("Date").reset_index(drop=True)
df["Date"] = pd.to_datetime(df["Date"])

if "LogReturn" not in df.columns:
    df["LogReturn"] = np.log(df["Close"]).diff()

returns = df["LogReturn"].dropna().to_numpy()
dates   = df["Date"].iloc[1:].reset_index(drop=True)

print(f"    Loaded {len(df)} rows  ({df['Date'].iloc[0].date()} – {df['Date'].iloc[-1].date()})")


# SECTION 2 – HELPER: HURST R/S

def e_rs_for_N(x: np.ndarray, N: int) -> float:
    """Mean R/S for segments of length N (from E_RS.py)."""
    k = len(x) // N
    if k < 2:
        return np.nan
    xk = x[: k * N].reshape(k, N)
    m  = xk.mean(axis=1, keepdims=True)
    s  = xk.std(axis=1, ddof=1, keepdims=True)
    y  = np.cumsum(xk - m, axis=1)
    R  = y.max(axis=1) - y.min(axis=1)
    S  = s.squeeze()
    rs = R / S
    return float(np.nanmean(rs))


def hurst_rs(series: np.ndarray, n_lags: int = 20) -> float:
    """
    Estimate the Hurst exponent via R/S analysis (from test.py).
    Returns H, or np.nan if too few valid lags.
    """
    n = len(series)
    if n < 20:
        return np.nan
    lags = np.unique(
        np.logspace(np.log10(10), np.log10(max(n // 2, 11)), n_lags).astype(int)
    )
    log_rs, log_n = [], []
    for lag in lags:
        k = n // lag
        if k < 2:
            continue
        segs = series[: k * lag].reshape(k, lag)
        m    = segs.mean(axis=1, keepdims=True)
        dev  = np.cumsum(segs - m, axis=1)
        R    = dev.max(axis=1) - dev.min(axis=1)
        S    = segs.std(axis=1, ddof=1)
        mask = S > 0
        if mask.sum() < 2:
            continue
        log_rs.append(np.log10(np.mean(R[mask] / S[mask])))
        log_n.append(np.log10(lag))
    if len(log_n) < 4:
        return np.nan
    H, _ = np.polyfit(log_n, log_rs, 1)
    return H


# SECTION 3 – R/S SCALING TABLE  (E_RS.py)
print("\n[2] Computing R/S scaling table …")

N_LIST = [16, 32, 64, 128, 256, 512, 1024]
rs_rows = []
for N in N_LIST:
    ers = e_rs_for_N(returns, N)
    rs_rows.append({
        "N": N, "E_RS": ers,
        "log10N": np.log10(N), "log10E_RS": np.log10(ers)
    })
rs_table = pd.DataFrame(rs_rows)
rs_table.to_excel(OUTPUT_DIR / "BTC_RS_Table.xlsx", index=False)
print(rs_table.to_string(index=False))


# SECTION 4 – GLOBAL HURST + FIGURE 4  (test.py)
print("\n[3] Estimating global Hurst exponent …")

lags_global = np.unique(
    np.logspace(np.log10(10), np.log10(len(returns) // 2), 30).astype(int)
)
log_rs_g, log_n_g = [], []
for lag in lags_global:
    k = len(returns) // lag
    if k < 2:
        continue
    segs = returns[: k * lag].reshape(k, lag)
    m    = segs.mean(axis=1, keepdims=True)
    dev  = np.cumsum(segs - m, axis=1)
    R    = dev.max(axis=1) - dev.min(axis=1)
    S    = segs.std(axis=1, ddof=1)
    mask = S > 0
    if mask.sum() < 2:
        continue
    log_rs_g.append(np.log10(np.mean(R[mask] / S[mask])))
    log_n_g.append(np.log10(lag))

H_global, intercept = np.polyfit(log_n_g, log_rs_g, 1)
print(f"    Global H = {H_global:.4f}")

fit_line = np.array(log_n_g) * H_global + intercept
r2 = 1 - np.sum((np.array(log_rs_g) - fit_line) ** 2) / \
         np.sum((np.array(log_rs_g) - np.mean(log_rs_g)) ** 2)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(log_n_g, log_rs_g, color='#2166ac', zorder=3, label='Data points')
ax.plot(log_n_g, fit_line, color='#d73027', linewidth=1.8,
        label=f'y = {H_global:.4f}x + {intercept:.4f}\nR² = {r2:.4f}')
ax.set_xlabel('Log₁₀(N)', fontsize=11)
ax.set_ylabel('Log₁₀(E[R/S])', fontsize=11)
ax.set_title('R/S Scaling (Global) – Bitcoin Daily Log Returns', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_global_rs.png', dpi=200, bbox_inches='tight')
plt.close()
print("    Saved fig_global_rs.png")


# SECTION 5 – ROLLING HURST + FIGURE 5  (test.py)
print(f"\n[4] Computing rolling Hurst (window={ROLLING_WINDOW}d, step={ROLLING_STEP}d) …")

roll_h = []
roll_d = []
roll_start_dates = []
roll_end_dates = []

for start in range(0, len(returns) - ROLLING_WINDOW, ROLLING_STEP):
    seg = returns[start: start + ROLLING_WINDOW]
    h   = hurst_rs(seg)

    start_date = dates.iloc[start]
    end_date   = dates.iloc[start + ROLLING_WINDOW - 1]
    mid_date   = dates.iloc[start + ROLLING_WINDOW // 2]

    roll_h.append(h)
    roll_d.append(mid_date)
    roll_start_dates.append(start_date)
    roll_end_dates.append(end_date)

roll_h = np.array(roll_h)
roll_d = pd.DatetimeIndex(roll_d)
roll_start_dates = pd.DatetimeIndex(roll_start_dates)
roll_end_dates = pd.DatetimeIndex(roll_end_dates)

mask = ~np.isnan(roll_h)
roll_h = roll_h[mask]
roll_d = roll_d[mask]
roll_start_dates = roll_start_dates[mask]
roll_end_dates = roll_end_dates[mask]

print(f"    {len(roll_h)} rolling estimates  "
      f"({roll_d[0].date()} – {roll_d[-1].date()})")
print(f"    Mean H = {roll_h.mean():.4f},  "
      f"Min = {roll_h.min():.4f},  Max = {roll_h.max():.4f}")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(roll_d, roll_h, color='#2166ac', linewidth=1.1,
        label=f'Rolling Hurst Exponent (R/S, {ROLLING_WINDOW}-day window)')
ax.axhline(0.5, color='black', linewidth=1.0, linestyle='--', label='H = 0.5 benchmark')

ax.set_ylabel('Hurst Exponent H', fontsize=11)
ax.set_xlabel('Date', fontsize=11)
ax.set_title(
    f'Rolling Hurst Exponent – Bitcoin (Window = {ROLLING_WINDOW} days, Step = {ROLLING_STEP} days)',
    fontsize=12, fontweight='bold'
)

ax.set_ylim(0.45, 0.70)
ax.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.70])

ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_rolling_hurst.png', dpi=200, bbox_inches='tight')
plt.close()
print("    Saved fig_rolling_hurst.png")

# SECTION 6 – STRUCTURAL BREAKPOINT ANALYSIS 
print("\n[5] Running structural breakpoint analysis …")

# 6a.  PELT-style binary segmentation 
def pelt_mean_shift(signal, min_size=26, penalty=None):
    n = len(signal)
    if penalty is None:
        penalty = np.log(n) * np.var(signal) * 2

    def cost(i, j):
        return np.var(signal[i:j]) * (j - i)

    breakpoints = []
    queue = [(0, n)]
    while queue:
        lo, hi = queue.pop()
        if hi - lo < 2 * min_size:
            continue
        base = cost(lo, hi)
        best_gain, best_k = 0, None
        for k in range(lo + min_size, hi - min_size):
            gain = base - cost(lo, k) - cost(k, hi) - penalty
            if gain > best_gain:
                best_gain, best_k = gain, k
        if best_k is not None:
            breakpoints.append(best_k)
            queue.append((lo, best_k))
            queue.append((best_k, hi))
    return sorted(breakpoints)

bps      = pelt_mean_shift(roll_h)
bp_dates = [roll_d[b] for b in bps]

print(f"    Detected {len(bps)} breakpoints:")
for d in bp_dates:
    print(f"      {d.date()}")

# 6b.  Regime statistics 
regime_edges  = [roll_d[0]] + bp_dates + [roll_d[-1]]
regime_labels, regime_data, regime_stats_rows = [], [], []

for i in range(len(regime_edges) - 1):
    s, e  = regime_edges[i], regime_edges[i + 1]
    seg   = roll_h[(roll_d >= s) & (roll_d < e)]
    lbl   = f"R{i+1}\n{s.strftime('%b %y')}–{e.strftime('%b %y')}"
    regime_labels.append(lbl)
    regime_data.append(seg)
    regime_stats_rows.append({
        'Regime': f'R{i+1}', 'Start': s.date(), 'End': e.date(),
        'Mean_H': round(seg.mean(), 4), 'Std_H': round(seg.std(), 4), 'N_obs': len(seg)
    })
    print(f"    R{i+1} ({s.date()} – {e.date()}):  "
          f"mean H = {seg.mean():.4f}, std = {seg.std():.4f}, n = {len(seg)}")

# 6c.  Adjacent t-tests 
print("\n    Pairwise t-tests (adjacent regimes):")
ttest_rows = []
for i in range(len(regime_data) - 1):
    t, p = stats.ttest_ind(regime_data[i], regime_data[i + 1], equal_var=False)
    sig  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"      R{i+1} vs R{i+2}: t = {t:.3f}, p = {p:.4f}  {sig}")
    ttest_rows.append({'Comparison': f'R{i+1} vs R{i+2}',
                       't-stat': round(t, 4), 'p-value': round(p, 6), 'Significance': sig})

# ── 6d.  Event study 
EVENTS = {
    '2017 Bull Peak':    pd.Timestamp('2017-12-17'),
    'COVID Crash':       pd.Timestamp('2020-03-12'),
    '2021 ATH':          pd.Timestamp('2021-11-10'),
    'FTX Collapse':      pd.Timestamp('2022-11-08'),
    '2024 ETF Approval': pd.Timestamp('2024-01-10'),
}
EVENT_WINDOW = 180
event_rows = []

print("\n    Pre/post event analysis (descriptive, ±180 days):")
for name, edate in EVENTS.items():

    # Pre-event windows: windows that END before the event
    pre = roll_h[
        (roll_end_dates >= edate - pd.Timedelta(days=EVENT_WINDOW)) &
        (roll_end_dates < edate)
    ]

    # Post-event windows: windows that START after/on the event
    post = roll_h[
        (roll_start_dates >= edate) &
        (roll_start_dates < edate + pd.Timedelta(days=EVENT_WINDOW))
    ]

    if len(pre) < 3 or len(post) < 3:
        print(f"      {name}: insufficient data, skipped")
        continue

    pre_median = float(np.median(pre))
    post_median = float(np.median(post))
    pre_mean = float(np.mean(pre))
    post_mean = float(np.mean(post))

    if post_median > pre_median:
        direction = "increase"
    elif post_median < pre_median:
        direction = "decrease"
    else:
        direction = "no change"

    print(
        f"      {name}: "
        f"pre median H = {pre_median:.4f}, post median H = {post_median:.4f} "
        f"({direction})"
    )

    event_rows.append({
        'Event': name,
        'Pre_median_H': round(pre_median, 4),
        'Post_median_H': round(post_median, 4),
        'Pre_mean_H': round(pre_mean, 4),
        'Post_mean_H': round(post_mean, 4),
        'Direction': direction,
        'N_pre': len(pre),
        'N_post': len(post)
    })

# SECTION 7 – FIGURES 15, 16, 17
print("\n[6] Generating breakpoint figures …")

REGIME_COLS = ['#d0e8f1','#fde8cc','#d6f0d6','#f5d5e0','#e8d6f5','#fffacc','#ddeedd']

# Figure 15: rolling H + breakpoints + events 
fig, ax = plt.subplots(figsize=(14, 5))

for i in range(len(regime_edges) - 1):
    ax.axvspan(regime_edges[i], regime_edges[i + 1],
               alpha=0.20, color=REGIME_COLS[i % len(REGIME_COLS)], zorder=0)
    mid = regime_edges[i] + (regime_edges[i + 1] - regime_edges[i]) / 2
    ax.text(mid, 0.72, f'R{i+1}', ha='center', va='bottom',
            fontsize=8, color='#444444', fontweight='bold')

ax.plot(roll_d, roll_h, color='#2166ac', linewidth=1.1,
        label='Rolling Hurst Exponent (R/S, 365-day window)', zorder=3)
ax.axhline(0.5, color='black', linewidth=1.0, linestyle='--',
           label='Random walk (H = 0.5)', zorder=2)

for bd in bp_dates:
    ax.axvline(bd, color='#d73027', linewidth=1.4, linestyle='-', zorder=4, alpha=0.7)

event_cols = ['#e66101','#5e3c99','#1a9641','#d01c8b','#f4a582']
for (name, edate), col in zip(EVENTS.items(), event_cols):
    if roll_d[0] <= edate <= roll_d[-1]:
        ax.axvline(edate, color=col, linewidth=1.3, linestyle=':', zorder=4, alpha=0.9)
        ax.text(edate, 0.6, name.replace(' ', '\n'), ha='center', va='top',
                fontsize=6.0, color=col)

bp_patch = mpatches.Patch(color='#d73027', label='Estimated Breakpoint')
ax.legend(handles=[
    plt.Line2D([0],[0], color='#2166ac', lw=1.5, label='Rolling Hurst Exponent (R/S, 365-day)'),
    plt.Line2D([0],[0], color='black',   lw=1.0, ls='--', label='H = 0.5 benchmark'),
    bp_patch
], fontsize=8, loc='upper right')

ax.set_ylabel('Hurst Exponent H', fontsize=11)
ax.set_xlabel('Date', fontsize=11)
ax.set_title('Rolling Hurst Exponent with Estimated Breakpoints – Bitcoin (2015–2025)',
             fontsize=12, fontweight='bold')
ax.set_ylim(0.48, 0.78)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_rolling_hurst_breakpoints.png', dpi=200, bbox_inches='tight')
plt.close()
print("    Saved fig_rolling_hurst_breakpoints.png")

# Figure 16: regime boxplot 
fig, ax = plt.subplots(figsize=(11, 5))

bp_obj = ax.boxplot(
    regime_data,
    patch_artist=True,
    notch=False,
    medianprops=dict(color='black', linewidth=2)
)

for patch, col in zip(bp_obj['boxes'], REGIME_COLS):
    patch.set_facecolor(col)
    patch.set_alpha(0.75)

ax.axhline(
    0.5,
    color='black',
    linewidth=1.0,
    linestyle='--',
    label='H = 0.5 benchmark'
)

ax.set_xticklabels(regime_labels, fontsize=8)
ax.set_ylabel('Hurst Exponent H', fontsize=11)
ax.set_title(
    'Distribution of Rolling H by Estimated Regime – Bitcoin',
    fontsize=12,
    fontweight='bold'
)

ax.set_ylim(0.45, 0.70)
ax.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.70])

ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_regime_boxplot.png', dpi=200, bbox_inches='tight')
plt.close()
print("    Saved fig_regime_boxplot.png")

# Figure 17: pre/post event (descriptive)
valid_events = [r for r in event_rows]

if valid_events:
    fig, axes = plt.subplots(1, len(valid_events), figsize=(14, 4), sharey=True)
    if len(valid_events) == 1:
        axes = [axes]

    for ax, row in zip(axes, valid_events):
        edate = EVENTS[row['Event']]

        pre = roll_h[
            (roll_end_dates >= edate - pd.Timedelta(days=EVENT_WINDOW)) &
            (roll_end_dates < edate)
        ]

        post = roll_h[
            (roll_start_dates >= edate) &
            (roll_start_dates < edate + pd.Timedelta(days=EVENT_WINDOW))
        ]

        bp2 = ax.boxplot(
            [pre, post],
            patch_artist=True,
            medianprops=dict(color='black', linewidth=2)
        )

        bp2['boxes'][0].set_facecolor('#a6cee3')
        bp2['boxes'][0].set_alpha(0.8)
        bp2['boxes'][1].set_facecolor('#fb9a99')
        bp2['boxes'][1].set_alpha(0.8)

        ax.axhline(0.5, color='black', linewidth=0.9, linestyle='--')
        ax.set_xticklabels(['Pre', 'Post'], fontsize=9)

        pre_med = np.median(pre)
        post_med = np.median(post)

        if post_med > pre_med:
            change_text = "Median increase"
        elif post_med < pre_med:
            change_text = "Median decrease"
        else:
            change_text = "No median change"

        ax.set_title(
            f"{row['Event']}\n"
            f"{pre_med:.3f} → {post_med:.3f}\n"
            f"{change_text}",
            fontsize=8,
            fontweight='bold'
        )

        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Hurst Exponent H', fontsize=11)
    fig.suptitle(
        'Pre- vs Post-event Rolling H (descriptive, ±180 days) – Key Bitcoin Market Events',
        fontsize=11,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_pre_post_event.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("    Saved fig_pre_post_event.png")


# SECTION 8 – SAVE ALL RESULTS TO EXCEL
print("\n[7] Saving results to Excel …")

with pd.ExcelWriter(OUTPUT_DIR / 'BTC_breakpoint_results.xlsx') as writer:
    pd.DataFrame({'Date': roll_d, 'Hurst_H': roll_h}).to_excel(
        writer, sheet_name='RollingHurst', index=False)
    pd.DataFrame({'Breakpoint_Date': bp_dates}).to_excel(
        writer, sheet_name='Breakpoints', index=False)
    pd.DataFrame(regime_stats_rows).to_excel(
        writer, sheet_name='RegimeStats', index=False)
    pd.DataFrame(ttest_rows).to_excel(
        writer, sheet_name='Ttests_Regimes', index=False)
    pd.DataFrame(event_rows).to_excel(
        writer, sheet_name='Ttests_Events', index=False)
    rs_table.to_excel(writer, sheet_name='RS_ScalingTable', index=False)

print("Saved BTC_breakpoint_results.xlsx")

print("All Done")
