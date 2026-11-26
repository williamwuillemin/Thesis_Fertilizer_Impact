"""
crop_panel.py — do crops that need more nitrogen react more to fertilizer prices?

IDEA
The FAO-index regressions (model.py) cannot separate the fertilizer channel from oil, gas,
the dollar or the Hormuz closure itself, because all of them move in the same months.
Here we compare CROPS within the SAME month. Maize, rice and wheat need a lot of nitrogen
fertilizer; soybeans fix nitrogen from the air and need very little. If fertilizer prices
matter for food prices, a fertilizer price increase should raise maize, rice and wheat
prices MORE than soybean prices. Soybeans are the placebo (control) crop.

MODEL (panel: 4 crops x ~300 months; prices in monthly % changes = 100 * log-difference)

    dp_{c,t} = SUM_k beta_k * (fert_{t-k} x N_c) + r * dp_{c,t-1}
               + crop-specific energy slopes + crop fixed effect_c + month fixed effect_t + u_{c,t}

    dp_{c,t}      % change of the world price of crop c in month t
    N_c           nitrogen intensity of crop c, scaled 0 (soybeans) to 1 (maize, the most intensive)
    fert_{t-k}    % change of the fertilizer price k months ago (default: US PPI nitrogen fertilizer)
    beta_k        how much MORE a crop at N = 1 (maize) reacts than soybeans (N = 0), k months after
                  a +1% fertilizer price change. SUM of beta_k = cumulative extra effect after 12 months.
    month FE      one dummy per month: absorbs EVERYTHING common to all crops in that month
                  (oil, gas, freight, the dollar, COVID, the Hormuz closure, general fertilizer price
                  level...). This is why the fertilizer price itself never appears alone: its common
                  effect is inside the month dummies; only its DIFFERENTIAL effect by N is estimated.
    crop FE       one dummy per crop: absorbs crop-specific average price growth (trends).
    energy slopes oil and gas x crop dummies: maize, soybeans and vegetable oils are biofuel feedstocks,
                  so oil can move crops differently. Fertilizer prices move with gas, so without
                  these terms "fert x N" could pick up "energy x crop". Controlling for them keeps
                  the comparison about NITROGEN, not about biofuels.
    r             persistence of each crop's own price changes.

SECOND QUESTION: the Hormuz difference-in-differences
    dp_{c,t} = theta * (Hormuz_t x N_c) + same controls
    theta  = extra monthly price growth of maize (N = 1) vs soybeans (N = 0) during the closure.
    Event study: one coefficient per month around the closure, to check that the nitrogen-intensive
    crops did NOT already rise faster before March 2026 (parallel pre-trends).

Usage (from the repository root, after  python src/extract.py worldbank fred):
    python src/crop_panel.py                     # US PPI nitrogen, 12 lags
    python src/crop_panel.py --fert wb_urea      # robustness: World Bank urea (FOB origin) price
    python src/crop_panel.py --lags 6 --no-show
"""

# =============================================================================
# %% IMPORTS
# =============================================================================

# argparse: reads the options typed on the command line (--fert, --lags, ...).
import argparse

# numpy: logarithms, square roots; pandas: tables (DataFrames).
import numpy as np
import pandas as pd

# statsmodels formula API: regressions written as formulas, "y ~ x1 + x2".
import statsmodels.formula.api as smf

# Normal distribution, to turn a z-statistic into a p-value (long-run effect).
from scipy.stats import norm

# matplotlib: figures (event study) and the academic-style table image.
import matplotlib.pyplot as plt

# This repository: settings (episode dates, nitrogen rates) and the database reader.
import config
from database import monthly_panel


# =============================================================================
# %% STEP 1 — SETTINGS AND COMMAND-LINE OPTIONS
# =============================================================================

parser = argparse.ArgumentParser(description="Crop panel: nitrogen-intensive crops vs soybeans.",
                                 allow_abbrev=False)
# Fertilizer price: US PPI nitrogen fertilizer by default (full 2000-2026 sample, no series break,
# a market price on the side of the trade that went short). The World Bank urea price is quoted
# FOB Middle East since 2022, i.e. at the blocked origin, and collapsed during the closure.
parser.add_argument("--fert", default="fred_ppi_nitrogen", help="fertilizer price series")
# 12 lags = one crop year: fertilizer is bought before planting and affects prices through
# the NEXT harvest (expected yields), which can take most of a year.
parser.add_argument("--lags", type=int, default=12, help="number of fertilizer lags (default 12)")
parser.add_argument("--no-show", action="store_true", help="do not open figure windows")
# parse_known_args ignores the extra arguments of Jupyter / the VS Code interactive window.
# allow_abbrev=False (above) is needed because VS Code passes "--f=kernel.json", which argparse
# would otherwise read as an abbreviation of --fert / --food.
args = parser.parse_known_args()[0]
FERT, LAGS = args.fert, args.lags

# The four crops and their nitrogen rates (kg N/ha), defined with their sources in config.py.
CROPS = config.CROP_N_RATE                     # {"wb_maize": ("Maize", 118.8), ...}

# Last month where all series are available; same sample end as model.py.
END = "2026-09-01"

# Event-study window around the closure: 12 months before (Mar 2025 - Feb 2026)
# and every month since (Mar 2026 - Sep 2026). Feb 2026 is the reference month (path = 0).
EVENT_START, EVENT_REF = "2025-03-01", "2026-02-01"

# Driscoll-Kraay standard errors (statsmodels "hac-groupsum"), with up to 6 months of
# autocorrelation. Why: all four crops share world shocks every month (weather, demand, freight),
# so their errors are correlated ACROSS crops within a month, and price shocks persist OVER time.
# Driscoll-Kraay is robust to both. Clustering by crop would be invalid with only 4 crops.
DK_LAGS = 6

TABLE_DIR = config.PROJECT_DIR / "outputs" / "tables"
FIG_DIR = config.PROJECT_DIR / "outputs" / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# %% STEP 2 — LOAD THE PRICES
# =============================================================================
# World Bank Pink Sheet, nominal US$ per tonne, monthly:
#   wb_maize      US maize, Gulf export price          (benchmark for world corn trade)
#   wb_wheat_hrw  US hard red winter wheat, Gulf        (benchmark for world wheat trade)
#   wb_rice       Thai 5% broken rice, Bangkok          (benchmark for world rice trade)
#   wb_soybeans   US soybeans, export price             (the low-nitrogen placebo)
# plus the fertilizer price and the two energy prices used as controls.
# Nominal prices are fine: the month fixed effects absorb general inflation and the dollar,
# because they affect all crops in the same month.
df = monthly_panel(list(CROPS) + [FERT, "wb_brent", "wb_ngas_eu"], start=config.START_DATE, end=END)


def pct_change(series):
    """Monthly % change as 100 * log-difference (an elasticity scale; NaN in the first month)."""
    return 100 * np.log(series).diff()


# =============================================================================
# %% STEP 3 — NITROGEN INTENSITY N_c, SCALED 0 TO 1
# =============================================================================
# Raw rates in kg N/ha: maize 118.8, rice 113.8, wheat 97.8, soybeans 16.
# Scaling: N_c = (rate_c - rate_soybeans) / (rate_maize - rate_soybeans)
#   -> soybeans 0, wheat ~0.80, rice ~0.95, maize 1.
# Economic reason for the scaling: beta then reads directly as "how much more maize reacts than
# soybeans". The ranking and the relative distances come from agronomy, not from the price data,
# so the measure is set BEFORE looking at the results (no data mining).
rates = pd.Series({sid: rate for sid, (_, rate) in CROPS.items()})
N = (rates - rates.min()) / (rates.max() - rates.min())
print("Nitrogen intensity (0 = soybeans, 1 = maize):\n", N.round(3).to_string(), "\n")


# =============================================================================
# %% STEP 4 — BUILD THE PANEL (one row per crop and month)
# =============================================================================
# Variables common to all crops in a month (they will be multiplied by crop characteristics).
fert = pct_change(df[FERT])
brent = pct_change(df["wb_brent"])
gas = pct_change(df["wb_ngas_eu"])

# Hormuz closure dummy: 1 from March 2026 (first full month closed) to the end of the sample.
hormuz = (df.index >= config.HORMUZ_START).astype(int)

rows = []
for sid, (name, _) in CROPS.items():
    part = pd.DataFrame(index=df.index)
    part["crop"] = name                                    # crop label (for the crop fixed effect)
    part["N"] = N[sid]                                     # nitrogen intensity of this crop
    part["dp"] = pct_change(df[sid])                       # dependent variable: crop price % change
    part["dp_l1"] = part["dp"].shift(1)                    # last month's change of the SAME crop
    # Fertilizer x nitrogen interactions, lags 0..LAGS.
    # fxN_k = fertilizer % change k months ago x N_c. For soybeans (N = 0) these are always 0,
    # so soybeans identify the counterfactual: how a crop that does not need nitrogen moved.
    for k in range(LAGS + 1):
        part[f"fxN_{k}"] = fert.shift(k) * N[sid]
    # Hormuz x nitrogen: the difference-in-differences regressor.
    part["hxN"] = hormuz * N[sid]
    # Crop-specific energy slopes. One column per crop EXCEPT soybeans: the common energy effect
    # is already in the month dummies, so with all four crops the columns would be collinear.
    # Each coefficient = how much MORE this crop reacts to oil (gas) than soybeans.
    for other_sid, (other, _) in CROPS.items():
        if other_sid == "wb_soybeans":
            continue
        part[f"brent_{other}"] = brent * (sid == other_sid)
        part[f"gas_{other}"] = gas * (sid == other_sid)
    rows.append(part)

# Stack the four crops on top of each other -> long panel. The date becomes a normal column.
panel = pd.concat(rows).rename_axis("date").reset_index()

# Event-study regressors: one column per month of the window, = N_c in that month, 0 otherwise.
# Their coefficients trace the extra price growth of nitrogen-intensive crops month by month,
# relative to the average differential in all months OUTSIDE the window (the baseline).
# No month needs to be left out: the months outside the window already identify the baseline.
event_months = pd.date_range(EVENT_START, END, freq="MS")
event_cols = []
for m in event_months:
    col = f"ev_{m:%Y_%m}"
    panel[col] = (panel["date"] == m) * panel["N"]
    event_cols.append(col)

# One common sample for every regression: drop rows with a missing lag or price.
# (The 12 fertilizer lags and the own lag remove the first 13 months.)
fert_cols = [f"fxN_{k}" for k in range(LAGS + 1)]
energy_cols = [c for c in panel.columns if c.startswith(("brent_", "gas_"))]
panel = panel.dropna(subset=["dp", "dp_l1"] + fert_cols + energy_cols).reset_index(drop=True)

# Month fixed effects need a month identifier; Driscoll-Kraay needs integer periods in time order.
panel["month"] = panel["date"].dt.strftime("%Y-%m")
time_id = panel["date"].rank(method="dense").astype(int).values - 1

print(f"Panel: {panel['crop'].nunique()} crops x {panel['month'].nunique()} months = {len(panel)} rows, "
      f"{panel['date'].min():%Y-%m} to {panel['date'].max():%Y-%m}\n")


# =============================================================================
# %% STEP 5 — HELPERS
# =============================================================================
FE = "C(crop) + C(month)"                      # crop and month fixed effects
ENERGY = " + ".join(energy_cols)               # crop-specific oil and gas slopes
FERT_TERMS = " + ".join(fert_cols)             # fxN_0 + ... + fxN_12


def fit(formula):
    """OLS with Driscoll-Kraay standard errors (robust to cross-crop and serial correlation)."""
    return smf.ols(formula, data=panel).fit(cov_type="hac-groupsum",
                                            cov_kwds={"time": time_id, "maxlags": DK_LAGS})


def fert_effects(res):
    """
    Short-run cumulative effect S = beta_0 + ... + beta_L (extra % change of maize vs soybeans
    after L months, for a +1% fertilizer price change), and the long-run effect
    LR = S / (1 - r) once the crop's own price persistence r has played out.
    LR standard error by the delta method: Var(LR) = g' V g, with
        dLR/dbeta_k = 1/(1-r)   and   dLR/dr = S/(1-r)^2.
    """
    t = res.t_test(f"{FERT_TERMS} = 0")
    out = {"S": t.effect.item(), "S_se": t.sd.item(), "S_p": t.pvalue.item()}
    if "dp_l1" in res.params.index:
        r, S = res.params["dp_l1"], out["S"]
        g = pd.Series(0.0, index=res.params.index)
        g[fert_cols] = 1 / (1 - r)
        g["dp_l1"] = S / (1 - r) ** 2
        se = float(np.sqrt(g @ res.cov_params() @ g))
        out.update({"LR": S / (1 - r), "LR_se": se, "LR_p": 2 * (1 - norm.cdf(abs(S / (1 - r) / se)))})
    return out


def stars(p):
    """*** 1%, ** 5%, * 10% significance."""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# =============================================================================
# %% STEP 6 — THE REGRESSIONS
# =============================================================================
specs = {
    # (1) MAIN: fertilizer x nitrogen, 12 lags, own persistence, energy slopes, both fixed effects.
    #     Question: when fertilizer prices rise, do nitrogen-intensive crops rise more than soybeans?
    "(1) Fertilizer x N": f"dp ~ {FERT_TERMS} + dp_l1 + {ENERGY} + {FE}",

    # (2) Same without the energy slopes: shows how much the answer depends on separating
    #     the biofuel/energy channel from the nitrogen channel.
    "(2) No energy slopes": f"dp ~ {FERT_TERMS} + dp_l1 + {FE}",

    # (3) Hormuz difference-in-differences: did nitrogen-intensive crops grow faster than soybeans
    #     during the closure? The month dummies absorb everything the closure did to ALL crops
    #     (oil, freight, panic buying); theta is only the part that differs by nitrogen need.
    "(3) Hormuz x N": f"dp ~ hxN + dp_l1 + {ENERGY} + {FE}",

    # (4) Hormuz DiD controlling for the fertilizer price channel. If theta falls from (3) to (4),
    #     the closure hit nitrogen-intensive crops THROUGH fertilizer prices. If it stays, the closure
    #     acted through something else that also scales with nitrogen need (e.g. expected
    #     shortages of fertilizer QUANTITIES that the price index does not capture).
    "(4) Hormuz x N + fert x N": f"dp ~ hxN + {FERT_TERMS} + dp_l1 + {ENERGY} + {FE}",
}
results = {name: fit(formula) for name, formula in specs.items()}

# Print and collect the key numbers.
summary = []
for name, res in results.items():
    row = {"spec": name, "n": int(res.nobs), "r2": res.rsquared, "r": res.params["dp_l1"]}
    if "fxN_0" in res.params.index:
        row.update(fert_effects(res))
    if "hxN" in res.params.index:
        row.update({"theta": res.params["hxN"], "theta_se": res.bse["hxN"], "theta_p": res.pvalues["hxN"]})
    summary.append(row)
summary = pd.DataFrame(summary)
summary.to_csv(TABLE_DIR / f"crop_panel_{FERT}.csv", index=False)

print(f"Fertilizer price: {FERT}, lags 0-{LAGS}; Driscoll-Kraay SE ({DK_LAGS} lags)")
for _, row in summary.iterrows():
    line = f"{row['spec']:30s}"
    if pd.notna(row.get("S")):
        line += f" sum beta = {row['S']:6.3f}{stars(row['S_p']):3s} (p={row['S_p']:.3f})"
        line += f"  long-run = {row['LR']:6.3f}{stars(row['LR_p']):3s} (p={row['LR_p']:.3f})"
    if pd.notna(row.get("theta")):
        line += f"  theta = {row['theta']:6.3f}{stars(row['theta_p']):3s} (p={row['theta_p']:.3f})"
    print(line)

# The full output of the main model (individual lags, energy slopes) for the appendix.
# summary() also runs an F-test on ALL coefficients, including the ~300 month dummies. With
# Driscoll-Kraay errors the month-dummy part of the covariance matrix is degenerate (residuals sum
# to zero within each month), so statsmodels warns that the matrix is not full rank. That overall
# F-test is not used; the coefficients and tests above are unaffected. We silence the warning.
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    (TABLE_DIR / f"crop_panel_main_{FERT}.txt").write_text(results["(1) Fertilizer x N"].summary().as_text())


# =============================================================================
# %% STEP 7 — HOW TO READ THE NUMBERS
# =============================================================================
# sum beta (1)   +1% fertilizer price -> maize price grows sum-beta % MORE than soybeans over
#                12 months. Wheat (N = 0.80) by 0.80 x that, rice (N = 0.95) by 0.95 x that.
#                Positive and significant = evidence of a fertilizer cost channel into food prices
#                that oil, gas, the dollar and global shocks cannot explain.
# (1) vs (2)     if the effect only appears without energy slopes, it was a biofuel/energy story.
# theta (3)      extra monthly price growth of maize vs soybeans during the closure (in %/month);
#                over the 7 closure months, the cumulative gap is roughly 7 x theta.
# (3) vs (4)     the part of theta explained by fertilizer prices = the fertilizer channel of Hormuz.
# Caveats: 4 crops only; each world price also has its own shocks (e.g. India's rice export ban
# in 2023, Brazilian soybean harvests); N is a global average applied to a US or Thai price.


# =============================================================================
# %% STEP 8 — EVENT STUDY AROUND THE CLOSURE
# =============================================================================
# One coefficient per month (N x month dummy). Each is the extra % price change of maize vs
# soybeans IN that month, relative to the average outside the window. Before March 2026 the
# coefficients should be close to 0 (no pre-trend); afterwards they show when the gap opened.
res_ev = fit(f"dp ~ {' + '.join(event_cols)} + dp_l1 + {ENERGY} + {FE}")

# Cumulative path = relative log price of maize vs soybeans, normalised to 0 in February 2026
# (the last month before the closure). The coefficients are monthly CHANGES, so the level gap is
# their running sum:
#   after Feb 2026:  sum of the monthly coefficients from March 2026 up to month m
#   before Feb 2026: minus the sum of the coefficients from month m+1 up to Feb 2026 (included)
# Each sum is a linear combination of coefficients -> t_test gives its standard error.
path = []
for m in event_months:
    if m == pd.Timestamp(EVENT_REF):
        path.append((m, 0.0, 0.0))
        continue
    if m > pd.Timestamp(EVENT_REF):
        terms = [f"ev_{x:%Y_%m}" for x in event_months if pd.Timestamp(EVENT_REF) < x <= m]
        sign = ""
    else:
        terms = [f"ev_{x:%Y_%m}" for x in event_months if m < x <= pd.Timestamp(EVENT_REF)]
        sign = "-"
    t = res_ev.t_test(f"{sign}({' + '.join(terms)}) = 0")
    path.append((m, t.effect.item(), t.sd.item()))
path = pd.DataFrame(path, columns=["month", "cum", "se"]).set_index("month")
path.to_csv(TABLE_DIR / f"crop_panel_event_{FERT}.csv")

# --- Figure: (a) raw prices, (b) monthly coefficients, (c) cumulative gap -----------------------
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix"})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# (a) Raw prices, Feb 2026 = 100: the descriptive picture before any model.
ax = axes[0]
for sid, (name, _) in CROPS.items():
    s = df[sid].loc[EVENT_START:END]
    ax.plot(s.index, 100 * s / df.loc[EVENT_REF, sid], label=f"{name} (N = {N[sid]:.2f})",
            lw=2 if sid == "wb_soybeans" else 1.3, ls="--" if sid == "wb_soybeans" else "-")
ax.set_title("(a) World prices, Feb 2026 = 100")

# (b) Monthly event-study coefficients with 95% intervals.
# Caveat: each monthly coefficient is identified by ONE month, so its Driscoll-Kraay standard
# error rests on very little information and is probably too small. Read panels (b)-(c) as a
# description of the timing; the formal test is theta in regressions (3)-(4).
ax = axes[1]
coefs = pd.Series({pd.Timestamp(c[3:].replace("_", "-") + "-01"): res_ev.params[c] for c in event_cols})
ses = pd.Series({pd.Timestamp(c[3:].replace("_", "-") + "-01"): res_ev.bse[c] for c in event_cols})
ax.errorbar(coefs.index, coefs, yerr=1.96 * ses, fmt="o", ms=4, capsize=2, color="black")
ax.axhline(0, color="grey", lw=0.8)
ax.set_title("(b) Extra monthly % change, maize vs soybeans")

# (c) Cumulative gap with 95% band.
ax = axes[2]
ax.plot(path.index, path["cum"], color="black")
ax.fill_between(path.index, path["cum"] - 1.96 * path["se"], path["cum"] + 1.96 * path["se"],
                color="grey", alpha=0.3)
ax.axhline(0, color="grey", lw=0.8)
ax.set_title("(c) Cumulative price gap (%), Feb 2026 = 0")

for ax in axes:
    ax.axvline(pd.Timestamp(config.HORMUZ_START), color="firebrick", lw=1, ls=":")   # closure
    ax.tick_params(axis="x", rotation=45)
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("Hormuz closure: nitrogen-intensive crops vs soybeans (red line = March 2026). "
             "Panels (b)-(c): N x month coefficients, Driscoll-Kraay 95% intervals.", fontsize=10)
plt.tight_layout()
fig.savefig(FIG_DIR / f"crop_panel_event_{FERT}.png", dpi=200, bbox_inches="tight")


# =============================================================================
# %% STEP 9 — ACADEMIC-STYLE TABLE
# =============================================================================
cols = list(results)
table_rows = []                                             # (label, {spec: (value, se)})


def two_lines(value, se, p):
    return (f"{value:.3f}{stars(p)}".replace("-", "−"), f"({se:.3f})") if pd.notna(value) else ("", "")


get = lambda name, key: summary.set_index("spec").loc[name].get(key, np.nan)
table_rows.append((f"Fertilizer × N, short-run $\\Sigma_{{k=0}}^{{{LAGS}}}\\beta_k$",
                   {c: two_lines(get(c, "S"), get(c, "S_se"), get(c, "S_p")) for c in cols}))
table_rows.append(("Fertilizer × N, long-run $\\Sigma\\beta_k/(1-r)$",
                   {c: two_lines(get(c, "LR"), get(c, "LR_se"), get(c, "LR_p")) for c in cols}))
table_rows.append(("Hormuz × N ($\\theta$)",
                   {c: two_lines(get(c, "theta"), get(c, "theta_se"), get(c, "theta_p")) for c in cols}))
table_rows.append(("Own price change$_{t-1}$ ($r$)",
                   {c: two_lines(results[c].params["dp_l1"], results[c].bse["dp_l1"], results[c].pvalues["dp_l1"])
                    for c in cols}))
first_stat = len(table_rows)
table_rows.append(("Crop & month fixed effects", {c: ("Yes", "") for c in cols}))
table_rows.append(("Crop-specific oil & gas slopes",
                   {c: ("No" if "No energy" in c else "Yes", "") for c in cols}))
table_rows.append(("Observations", {c: (f"{int(results[c].nobs)}", "") for c in cols}))
table_rows.append(("$R^2$", {c: (f"{results[c].rsquared:.3f}", "") for c in cols}))

heights = [2 if i < first_stat else 1 for i in range(len(table_rows))]
total_h = sum(heights) + 8
fig, ax = plt.subplots(figsize=(9, total_h * 0.27))
ax.set_xlim(0, 1)
ax.set_ylim(total_h, 0)
ax.axis("off")
xs = {c: 0.46 + i * 0.145 for i, c in enumerate(cols)}
y = 0.5
ax.text(0.5, y, "Table: Crop panel — nitrogen-intensive crops vs soybeans", ha="center", va="center", fontsize=10.5)
y += 1.5
ax.axhline(y - 0.6, color="black", lw=1.4)
for c in cols:
    ax.text(xs[c], y, c.split(")")[0] + ")", ha="center", va="center", fontsize=10)
y += 1
ax.axhline(y - 0.5, color="black", lw=0.7)
for i, ((label, vals), h) in enumerate(zip(table_rows, heights)):
    if i == first_stat:
        ax.axhline(y - 0.5, color="black", lw=0.7)
    ax.text(0.02, y, label, ha="left", va="center", fontsize=9.5)
    for c in cols:
        v, s = vals[c]
        ax.text(xs[c], y, v, ha="center", va="center", fontsize=9.5)
        if s:
            ax.text(xs[c], y + 0.9, s, ha="center", va="center", fontsize=8.5)
    y += h
ax.axhline(y - 0.4, color="black", lw=1.4)
ax.text(0.02, y + 0.3,
        f"Notes: dependent variable = monthly % change (100 × Δln) of the world price of maize, rice, wheat and soybeans "
        f"(World Bank), {panel['date'].min():%Y-%m} to {panel['date'].max():%Y-%m}.\n"
        f"N = nitrogen intensity, 0 (soybeans, 16 kg N/ha) to 1 (maize, 118.8 kg N/ha). Fertilizer price: {FERT}, "
        f"lags 0–{LAGS}. Hormuz = 1 from March 2026.\n"
        "(1) main model; (2) without crop-specific energy slopes; (3) Hormuz difference-in-differences; "
        "(4) Hormuz controlling for the fertilizer channel.\n"
        f"Driscoll-Kraay standard errors ({DK_LAGS} lags) in parentheses; long-run SE by the delta method. "
        "***, **, *: 1%, 5%, 10%.",
        ha="left", va="top", fontsize=7.5, style="italic")
plt.tight_layout()
fig.savefig(FIG_DIR / f"crop_panel_table_{FERT}.png", dpi=200, bbox_inches="tight")

if args.no_show:
    plt.close("all")
else:
    plt.show()

print(f"\nSaved: {TABLE_DIR / f'crop_panel_{FERT}.csv'}, crop_panel_main_{FERT}.txt, "
      f"crop_panel_event_{FERT}.csv and two figures in {FIG_DIR}")
