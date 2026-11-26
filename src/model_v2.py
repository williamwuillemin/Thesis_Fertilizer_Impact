"""
model.py — distributed-lag OLS of food-price inflation on fertilizer-price inflation,
with ONE episode dummy (Hormuz, Ukraine or COVID) and a step-by-step decomposition
of that episode's effect. Replaces the old model.py, model_covid.py and model_ukraine.py.

Main model (every price in monthly % changes, i.e. 100 * log-differences):

    food_t = a + r*food_t-1 + b0*fert_t + b1*fert_t-1 + ... + b12*fert_t-12
               + c*gas_t + d*brent_t + e*D_t + u_t

    a          intercept (average monthly food inflation not explained by the regressors)
    r          persistence: share of last month's food inflation that carries over this month
    b0..b12    distributed lags of the fertilizer price (default: US PPI nitrogen fertilizer)
    b0+...+b12 SHORT-RUN cumulative effect after 12 months, holding food persistence aside
    (b0+...+b12)/(1-r)   LONG-RUN effect once persistence has played out  -> headline number
    c, d       energy controls: European gas and Brent oil
    e          episode dummy D_t: extra monthly food inflation during the episode not explained above
    u_t        error term

Decomposition of the episode effect (same sample for every column, so they are comparable):
    (A) food ~ D                         total gap in food inflation during the episode
    (B) food ~ fertilizer lags + D       ... left after the fertilizer channel
    (C) + gas + brent                    ... left after the energy channel
    (D) + food_t-1                       ... left after food-price persistence  (= main model)
If e falls from (A) to (C), the episode raised food prices THROUGH the variables added.

Usage (from the repository root, after  python src/extract.py fao worldbank fred):
    python src/model.py                                  # Hormuz, FFPI + cereals + veg oils, PPI nitrogen, 12 lags
    python src/model.py --dummy ukraine                  # or covid
    python src/model.py --food fao_ffpi fao_dairy        # any FAO indices
    python src/model.py --fert wb_urea                   # robustness: World Bank urea (FOB origin) price
    python src/model.py --fert wb_urea --lags 6 --no-ar --food fao_ffpi   # reproduces the old model.py
    python src/model.py --no-show                        # save the outputs without opening plot windows
"""

# =============================================================================
# %% IMPORTS
# =============================================================================

# argparse reads the options typed on the command line (--dummy, --food, ...).
import argparse

# numpy: numerical functions; we use np.log (natural logarithm) and np.sqrt.
import numpy as np

# pandas: the DataFrame, a table with one row per month and one column per variable.
import pandas as pd

# statsmodels formula API: regression written as a formula, R-style "y ~ x1 + x2".
import statsmodels.formula.api as smf

# Durbin-Watson statistic (residual autocorrelation) and the normal distribution (p-values).
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import norm

# matplotlib: draws the academic-style tables saved as images.
import matplotlib.pyplot as plt

# config.py (this repository): all settings, incl. the episode dates.
import config

# database.py (this repository): monthly_panel() reads series from the DuckDB database.
from database import monthly_panel


# =============================================================================
# %% STEP 1 — SETTINGS AND COMMAND-LINE OPTIONS
# =============================================================================

# Episodes: name -> (first month = 1, last month = 1). None as last month = until the end
# of the sample. Every episode is a WINDOW: in a model of monthly CHANGES, a dummy that
# stays at 1 forever measures a permanent change in the trend of food inflation, not the
# shock itself. Hormuz is still closed at the end of the sample (Sep 2026), so its window
# simply runs to the last month.
EPISODES = {
    "hormuz":  (config.HORMUZ_START, None),
    "ukraine": (config.UKRAINE_START, config.UKRAINE_END),
    "covid":   (config.COVID_START, config.COVID_END),
}

# Readable names for the tables.
TITLES = {"fao_ffpi": "Food Price Index", "fao_cereals": "Cereals", "fao_vegoils": "Vegetable oils",
          "fao_dairy": "Dairy", "fao_meat": "Meat", "fao_sugar": "Sugar"}
FERT_NAMES = {"fred_ppi_nitrogen": "US PPI nitrogen fertilizer", "wb_urea": "Urea (World Bank, FOB origin)",
              "fred_ppi_phosphate": "US PPI phosphate fertilizer", "wb_dap": "DAP (World Bank)"}

parser = argparse.ArgumentParser(description="Distributed-lag OLS of food prices on fertilizer prices.",
                                 allow_abbrev=False)
parser.add_argument("--dummy", choices=list(EPISODES), default="hormuz",
                    help="which episode dummy to include (default: hormuz)")
parser.add_argument("--food", nargs="+", default=["fao_ffpi", "fao_cereals", "fao_vegoils"],
                    help="FAO indices to explain, one regression set per index")
# Default fertilizer price: the US producer price index for nitrogen fertilizer (BLS, via FRED).
# Why not the World Bank urea price: since March 2022 it is quoted FOB Middle East, i.e. at the
# blocked origin. During the closure it COLLAPSED (-53% in June 2026) because cargoes were
# stranded, while buyers elsewhere paid more. The PPI is a market price in a large importing
# and producing country, available for the whole 2000-2026 sample without a series break.
# Caveat for the thesis: it is the price received by US plants (mostly gas-based), not an
# import price, and it is not seasonally adjusted.
parser.add_argument("--fert", default="fred_ppi_nitrogen",
                    help="fertilizer price series (default: fred_ppi_nitrogen; robustness: wb_urea)")
# 12 lags: fertilizer is bought before planting and affects prices around the NEXT harvest.
# Morão (2025) finds the peak around 10 months, Hernandez (2026) a build-up from month 4.
# 12 months covers a full crop year.
parser.add_argument("--lags", type=int, default=12, help="number of fertilizer lags (default 12)")
# Lagged food inflation: the old model's Durbin-Watson was 1.16 -> food inflation has momentum
# (persistence) that the regressors did not capture.
parser.add_argument("--no-ar", action="store_true", help="drop lagged food inflation from the model")
parser.add_argument("--no-show", action="store_true", help="do not open the table windows")
# parse_known_args ignores the extra arguments added by Jupyter / the VS Code interactive window.
# allow_abbrev=False (above) is needed because VS Code passes "--f=kernel.json", which argparse
# would otherwise read as an abbreviation of --fert / --food.
args = parser.parse_known_args()[0]

DUMMY, FOODS, FERT, LAGS, AR = args.dummy, args.food, args.fert, args.lags, not args.no_ar

# Newey-West correction: errors may be correlated up to 6 months apart.
# (Rule of thumb 0.75 * T^(1/3) gives about 5 for T = 300 months.)
HAC = {"cov_type": "HAC", "cov_kwds": {"maxlags": 6}}

# Last month where all core variables are available.
END = "2026-09-01"

# Folders for the outputs.
TABLE_DIR = config.PROJECT_DIR / "outputs" / "tables"
FIG_DIR = config.PROJECT_DIR / "outputs" / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# %% STEP 2 — LOAD THE DATA FROM THE DATABASE
# =============================================================================

# One call for all food indices plus the three prices.
#   FOODS        the FAO indices to explain (2014-16 = 100)
#   FERT         fertilizer price (PPI index or $/tonne; units do not matter in % changes)
#   wb_ngas_eu   natural gas in Europe, $/MMBtu: the main input of nitrogen fertilizer
#                (about 70% of the cost of ammonia), so it controls for the cost-push from gas.
#   wb_brent     Brent crude oil, $/barrel: transport, machinery fuel and the biofuel link
#                (vegetable oils and maize are used to make biodiesel and ethanol), so it
#                controls for the general energy channel into food prices.
df = monthly_panel(FOODS + [FERT, "wb_ngas_eu", "wb_brent"], start=config.START_DATE, end=END)


# =============================================================================
# %% STEP 3 — TRANSFORM PRICES INTO MONTHLY % CHANGES AND BUILD THE DUMMY
# =============================================================================
# Price LEVELS trend upward (non-stationary): regressing one trending series on another gives a
# spurious regression. Monthly CHANGES are stationary, so OLS is valid.
# 100 * [ln(P_t) - ln(P_t-1)] ~ % change from last month; coefficients are elasticities.

def pct_change(series):
    """Monthly % change as 100 * log-difference. The first month becomes NaN."""
    return 100 * np.log(series).diff()


# Variables shared by every food index.
base = pd.DataFrame({
    "gas": pct_change(df["wb_ngas_eu"]),
    "brent": pct_change(df["wb_brent"]),
})

# Episode dummy: 1 inside the window, 0 outside.
#   start, end = the episode's first and last month (end None -> last month of the sample)
#   (df.index >= start) & (df.index <= end) -> True inside the window
#   .astype(int) -> True/False becomes 1/0
start, end = EPISODES[DUMMY]
end = end or df.index[-1]
base[DUMMY] = ((df.index >= start) & (df.index <= pd.Timestamp(end))).astype(int)

# Fertilizer lags: column fert_lag{k} holds the fertilizer % change of month t-k.
#   .shift(k) moves the column down by k rows.
fert = pct_change(df[FERT])
for k in range(LAGS + 1):
    base[f"fert_lag{k}"] = fert.shift(k)

# Text "fert_lag0 + fert_lag1 + ... + fert_lag12", used in the formulas and the tests.
FERT_TERMS = " + ".join(f"fert_lag{k}" for k in range(LAGS + 1))


# =============================================================================
# %% STEP 4 — HELPER: CUMULATIVE AND LONG-RUN FERTILIZER EFFECTS
# =============================================================================

def fert_effects(result):
    """
    Short-run cumulative effect S = b0 + ... + bL, and, if food_l1 is in the model,
    the long-run effect LR = S / (1 - r).

    Why LR: with lagged food inflation, a fertilizer shock raises food inflation today,
    which raises it again next month through r, and so on. Summing that geometric series
    gives S * (1 + r + r^2 + ...) = S / (1 - r).

    Standard error of LR by the delta method: LR is a nonlinear function of the estimated
    coefficients, so its variance is g' V g, where V is the (HAC) covariance matrix of the
    coefficients and g the vector of derivatives of LR with respect to each coefficient:
        dLR/db_k = 1 / (1 - r)        for every lag k
        dLR/dr   = S / (1 - r)^2
    """
    # Short run: statsmodels tests the linear combination b0 + ... + bL = 0 directly.
    t = result.t_test(f"{FERT_TERMS} = 0")
    out = {"S": t.effect.item(), "S_se": t.sd.item(), "S_p": t.pvalue.item()}

    if "food_l1" in result.params.index:
        r = result.params["food_l1"]
        S = out["S"]
        # g: one derivative per coefficient, in the same order as result.params; 0 for the others.
        g = pd.Series(0.0, index=result.params.index)
        g[[f"fert_lag{k}" for k in range(LAGS + 1)]] = 1 / (1 - r)
        g["food_l1"] = S / (1 - r) ** 2
        V = result.cov_params()                          # HAC covariance matrix of the coefficients
        lr_se = float(np.sqrt(g @ V @ g))                # sqrt(g' V g)
        lr = S / (1 - r)
        out.update({"LR": lr, "LR_se": lr_se, "LR_p": 2 * (1 - norm.cdf(abs(lr / lr_se)))})
    return out


def stars(p):
    """*** 1%, ** 5%, * 10% significance."""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# =============================================================================
# %% STEP 5 — ESTIMATE THE DECOMPOSITION FOR EACH FOOD INDEX
# =============================================================================
# For each food index, four nested regressions (A) -> (D). Each adds one channel, and we
# watch how the episode coefficient e changes. What disappears between two columns is the
# part of the episode effect that went THROUGH the variables just added.

summary_rows = []          # one row per food index and specification, saved as a CSV at the end
all_results = {}           # food -> {spec: fitted result}, used to draw the tables

for FOOD in FOODS:

    # ---- 5.1 Build the data for this food index --------------------------------------
    # .copy() so that adding columns does not change the shared table "base".
    data = base.copy()
    # Dependent variable: monthly % change of this food index.
    data["food"] = pct_change(df[FOOD])
    # Last month's food inflation (the persistence term r).
    data["food_l1"] = data["food"].shift(1)

    # ---- 5.2 One common sample for all four columns ----------------------------------
    # The 12 fertilizer lags make the first 13 months unusable. If column (A) used those
    # extra months and (D) did not, a change in e could come from the different months,
    # not from the added variables. Dropping incomplete rows ONCE, with every variable
    # present, guarantees that all columns are estimated on exactly the same months.
    needed = ["food", "food_l1", "gas", "brent", DUMMY] + [f"fert_lag{k}" for k in range(LAGS + 1)]
    data = data[needed].dropna()

    # ---- 5.3 The four specifications -------------------------------------------------
    # Each formula adds terms to the previous one. A dict keeps them in order.
    specs = {
        "A": f"food ~ {DUMMY}",                                   # episode only: total gap
        "B": f"food ~ {FERT_TERMS} + {DUMMY}",                    # + fertilizer channel
        "C": f"food ~ {FERT_TERMS} + gas + brent + {DUMMY}",      # + energy channel
    }
    if AR:
        specs["D"] = f"food ~ food_l1 + {FERT_TERMS} + gas + brent + {DUMMY}"   # + persistence

    # ---- 5.4 Estimate each one with Newey-West standard errors -----------------------
    # smf.ols(...) defines the model, .fit(**HAC) estimates it; coefficients are plain OLS,
    # only the standard errors and p-values are corrected for autocorrelation.
    results = {name: smf.ols(formula, data=data).fit(**HAC) for name, formula in specs.items()}
    all_results[FOOD] = (results, data)

    # ---- 5.5 Collect the numbers to compare across columns ---------------------------
    for name, res in results.items():
        row = {"food": FOOD, "spec": name, "dummy": DUMMY, "fert": FERT, "lags": LAGS,
               "e_dummy": res.params[DUMMY], "e_se": res.bse[DUMMY], "e_p": res.pvalues[DUMMY],
               "n": int(res.nobs), "r2": res.rsquared, "dw": durbin_watson(res.resid)}
        if "fert_lag0" in res.params.index:          # columns B, C, D have the fertilizer lags
            row.update(fert_effects(res))
        for v in ("gas", "brent", "food_l1"):        # energy and persistence, when present
            if v in res.params.index:
                row[v], row[v + "_p"] = res.params[v], res.pvalues[v]
        summary_rows.append(row)

    # ---- 5.6 Save the full regression output of the main (last) specification --------
    main = list(results)[-1]                          # "D" with persistence, otherwise "C"
    (TABLE_DIR / f"ols_{DUMMY}_{FOOD}_{FERT}.txt").write_text(results[main].summary().as_text())

    # ---- 5.7 Print the decomposition for this food index -----------------------------
    print(f"\n=== {TITLES.get(FOOD, FOOD)} | episode: {DUMMY} | fertilizer: {FERT} | "
          f"{LAGS} lags | sample {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} months)")
    for name, res in results.items():
        print(f"  ({name}) e = {res.params[DUMMY]:7.3f}{stars(res.pvalues[DUMMY]):3s} "
              f"(se {res.bse[DUMMY]:.3f}, p = {res.pvalues[DUMMY]:.3f})   R2 = {res.rsquared:.3f}")
    fx = fert_effects(results[main])
    print(f"  Fertilizer, main model: short-run sum = {fx['S']:.3f} (p = {fx['S_p']:.3f})"
          + (f", long-run = {fx['LR']:.3f} (p = {fx['LR_p']:.3f})" if "LR" in fx else ""))


# Save every number of every column in one CSV (easy to turn into a LaTeX table later).
summary = pd.DataFrame(summary_rows)
summary.to_csv(TABLE_DIR / f"decomposition_{DUMMY}_{FERT}.csv", index=False)


# =============================================================================
# %% STEP 6 — HOW TO READ THE DECOMPOSITION
# =============================================================================
# e(A)          total difference in monthly food inflation during the episode vs other months.
# e(A) - e(B)   part that runs through fertilizer prices.
# e(B) - e(C)   part that runs through energy prices. Oil and gas are partly CAUSED by the
#               episode (Hormuz raised Brent by 38% in March 2026), so they are mediators,
#               not neutral controls: e(C) is the DIRECT effect, net of these channels.
# e(C) - e(D)   part absorbed by food-price momentum.
# The differences are descriptive (each step is a separate regression), not formal tests;
# the order in which channels are added can change how the total is split.


# =============================================================================
# %% STEP 7 — ACADEMIC-STYLE TABLE (one image per food index)
# =============================================================================
# Columns = specifications (A)-(D). Rows = episode dummy, fertilizer effects, controls,
# statistics. The individual lag coefficients are in the .txt files of step 5.6.

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],   # Times-like font
                     "mathtext.fontset": "stix"})

DUMMY_LABEL = {"hormuz": "Hormuz closure", "ukraine": "Ukraine war", "covid": "COVID-19"}[DUMMY]


def cell(value, se=None, p=None):
    """Format a coefficient with stars, and its standard error in brackets (two lines)."""
    if value is None or pd.isna(value):
        return "", ""
    txt = f"{value:.3f}{stars(p) if p is not None else ''}".replace("-", "−")
    return txt, (f"({se:.3f})" if se is not None else "")


for FOOD, (results, data) in all_results.items():
    names = list(results)                                     # ["A", "B", "C", "D"]
    effects = {n: (fert_effects(r) if "fert_lag0" in r.params.index else {}) for n, r in results.items()}

    # Each table row: (label, {column: (line 1, line 2)})
    rows = []

    def coef_row(label, var):
        """Coefficient of variable `var` in each column ('' where the variable is absent)."""
        rows.append((label, {n: cell(r.params.get(var), r.bse.get(var), r.pvalues.get(var))
                             for n, r in results.items()}))

    coef_row(f"{DUMMY_LABEL} dummy$_t$", DUMMY)
    rows.append((f"Fertilizer, short-run $\\Sigma_{{k=0}}^{{{LAGS}}} b_k$",
                 {n: cell(e.get("S"), e.get("S_se"), e.get("S_p")) for n, e in effects.items()}))
    if AR:
        rows.append(("Fertilizer, long-run $\\Sigma b_k/(1-r)$",
                     {n: cell(e.get("LR"), e.get("LR_se"), e.get("LR_p")) for n, e in effects.items()}))
    coef_row("$\\Delta\\ln$ Gas (Europe)$_t$", "gas")
    coef_row("$\\Delta\\ln$ Brent$_t$", "brent")
    if AR:
        coef_row("Food inflation$_{t-1}$ ($r$)", "food_l1")
    coef_row("Constant", "Intercept")
    first_stat = len(rows)
    for label, fn in [("Observations", lambda r: f"{int(r.nobs)}"),
                      ("$R^2$", lambda r: f"{r.rsquared:.3f}"),
                      ("Durbin-Watson", lambda r: f"{durbin_watson(r.resid):.2f}")]:
        rows.append((label, {n: (fn(r), "") for n, r in results.items()}))

    # ---- draw ---------------------------------------------------------------------
    # Each coefficient row uses two text lines (value, standard error); statistics use one.
    heights = [2 if i < first_stat else 1 for i in range(len(rows))]
    total_h = sum(heights) + 8
    fig, ax = plt.subplots(figsize=(8, total_h * 0.26))
    ax.set_xlim(0, 1)
    ax.set_ylim(total_h, 0)                                     # line 0 at the top
    ax.axis("off")
    xs = {n: 0.50 + i * 0.15 for i, n in enumerate(names)}      # x position of each column

    y = 0.5
    ax.text(0.5, y, f"Table: Decomposition of the {DUMMY_LABEL} effect on food-price inflation — "
                    f"{TITLES.get(FOOD, FOOD)}", ha="center", va="center", fontsize=10)
    y += 1.5
    ax.axhline(y - 0.6, color="black", lw=1.4)                  # top rule
    for n in names:
        ax.text(xs[n], y, f"({n})", ha="center", va="center", fontsize=10)
    y += 1
    ax.axhline(y - 0.5, color="black", lw=0.7)                  # rule under the header

    for i, ((label, cols), h) in enumerate(zip(rows, heights)):
        if i == first_stat:
            ax.axhline(y - 0.5, color="black", lw=0.7)          # rule before the statistics
        ax.text(0.02, y, label, ha="left", va="center", fontsize=9.5)
        for n in names:
            line1, line2 = cols.get(n, ("", ""))
            ax.text(xs[n], y, line1, ha="center", va="center", fontsize=9.5)
            if line2:
                ax.text(xs[n], y + 0.9, line2, ha="center", va="center", fontsize=8.5)
        y += h

    ax.axhline(y - 0.4, color="black", lw=1.4)                  # bottom rule
    ax.text(0.02, y + 0.3,
            f"Notes: dependent variable = monthly % change (100 × Δln) of the FAO {TITLES.get(FOOD, FOOD)} index, "
            f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}.\n"
            f"Fertilizer price: {FERT_NAMES.get(FERT, FERT)}, lags 0–{LAGS}. (A) episode dummy only; "
            f"(B) + fertilizer; (C) + energy" + ("; (D) + lagged food inflation." if AR else ".") + "\n"
            "Newey-West (HAC, 6 lags) standard errors in parentheses; long-run SE by the delta method.\n"
            "***, **, *: significant at the 1%, 5%, 10% level. Sources: FAO, World Bank Pink Sheet, FRED (BLS).",
            ha="left", va="top", fontsize=7.5, style="italic")

    plt.tight_layout()
    fig.savefig(FIG_DIR / f"ols_table_{DUMMY}_{FOOD}_{FERT}.png", dpi=200, bbox_inches="tight")
    if args.no_show:
        plt.close(fig)

if not args.no_show:
    plt.show()


# =============================================================================
# %% STEP 8 — COMPACT SUMMARY ACROSS FOOD INDICES
# =============================================================================
# One line per food index, main specification only: the numbers to quote in the text.
main_spec = "D" if AR else "C"
print(f"\nSummary, main model ({main_spec}), episode = {DUMMY}, fertilizer = {FERT}:")
cols = ["food", "e_dummy", "e_p", "S", "S_p"] + (["LR", "LR_p", "food_l1"] if AR else []) + ["brent", "brent_p", "r2"]
print(summary.loc[summary["spec"] == main_spec, cols].round(3).to_string(index=False))
print(f"\nSaved: {TABLE_DIR / f'decomposition_{DUMMY}_{FERT}.csv'}, the .txt regression outputs "
      f"and the table images in {FIG_DIR}")


# NOTES

# the fertilizer price is now fred_ppi_nitrogen, with 12 lags and last month's food inflation added.
#  It runs on three indices: overall (FFPI), cereals and vegetable oils. 
# --fert wb_urea keeps the old urea price as a robustness check.