"""
model_covid.py — same OLS as model.py, with a covid dummy instead of the Hormuz dummy.

Model estimated (all variables in monthly % changes, i.e. 100 * log-differences):

    dlog(food)_t = a + b0*dlog(urea)_t + b1*dlog(urea)_t-1 + ... + b6*dlog(urea)_t-6
                     + c*dlog(gas)_t + d*dlog(brent)_t + e*covid_t + u_t

    a          intercept (average monthly food inflation not explained by the regressors)
    b0         impact effect: % change in food this month for a +1% urea change this month
    b1..b6     effect of urea changes 1..6 months ago (the "distributed lags")
    b0+...+b6  total (cumulative) effect after 6 months  -> the headline number
    c, d       controls: European gas and Brent oil, so urea does not pick up the energy shock
    e          COVID dummy: extra monthly food inflation during March 2020 - December 2021 not explained above
    u_t        error term

Usage (from the repository root, after the database is filled with  python src/extract.py fao worldbank):
    python src/model_covid.py               # FAO Food Price Index
    python src/model_covid.py fao_cereals   # or any sub-index: fao_vegoils, fao_dairy, fao_meat, fao_sugar
"""

# =============================================================================
# %% IMPORTS
# =============================================================================

# sys gives access to the words typed on the command line (sys.argv),
# so the food index can be chosen when running the script.
import sys

# numpy: numerical functions; we use np.log (natural logarithm).
import numpy as np

# pandas: the DataFrame, a table with one row per month and one column per variable.
import pandas as pd

# statsmodels formula API: lets us write the regression as a formula, R-style:
# "y ~ x1 + x2". smf.ols = ordinary least squares.
import statsmodels.formula.api as smf

# config.py (this repository): all settings, incl. COVID_START / COVID_END = the pandemic window.
import config

# database.py (this repository): monthly_panel() reads series from the DuckDB database
# (data/processed/thesis.duckdb) and returns them as a wide monthly table.
from database import monthly_panel


# =============================================================================
# %% SETTINGS
# =============================================================================

# Which food index to explain (the dependent variable).
# sys.argv is the list of words typed after "python":
#   sys.argv[0] = "src/model_covid.py" (the script itself)
#   sys.argv[1] = the first word after it, e.g. "fao_cereals", if one was typed
# If nothing was typed (len(sys.argv) == 1), we default to the overall FAO Food Price Index.

# fao_ffpi (the overall index) with a sub-index.
FOOD = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "fao_ffpi"

# Number of past months of urea to include. LAGS = 6 -> lags 0,1,...,6 = 7 urea variables.
# Fertilizer affects food with a delay (purchase -> planting -> harvest), so we let
# the data show the timing instead of forcing the whole effect into the same month.


#Fertilizer is bought before planting and the harvest comes months later, 
#so the effect should be spread over several months, not all in the same month. 
#Six months roughly covers one planting-to-harvest cycle.

# Hernandez finds the transmission is weak in months 1–3 and strengthens from month 4. 
# Six lags capture that build-up.
LAGS = 6


# =============================================================================
#%%STEP 1 — LOAD THE DATA FROM THE DATABASE
# =============================================================================

# Ask the database for 4 series, from January 2000 to September 2026
# (September 2026 = last month where all core variables are available):
#   FOOD         the chosen FAO index (2014-16 = 100)
#   wb_urea      urea price, $/tonne (World Bank Pink Sheet, FOB Middle East since 2022)
#   wb_ngas_eu   natural gas price in Europe, $/MMBtu (main input to make urea)
#   wb_brent     Brent crude oil price, $/barrel (transport, biofuels, general energy costs)
# Result: a table with the month as row index and one column per series.
df = monthly_panel([FOOD, "wb_urea", "wb_ngas_eu", "wb_brent"], start="2000-01-01", end="2026-09-01")


# Explanation of the variables:

#  FOOD, the FAO food index (the outcome). This is what the thesis wants to explain. 
# The FAO index is the standard measure of world food prices: it tracks international 
# export prices of the main traded foods.

# wb_urea, the urea price (the cause being tested). Urea is the most traded nitrogen fertilizer, 
# and the Gulf supplies about 35% of world exports, almost all through Hormuz.

# wb_ngas_eu, the European natural gas price (a control). Gas is the main raw material for urea, 
# about 70% of its cost. When gas rises, urea rises.

# wb_ngas_eu, the European natural gas price (a control). Gas is the main raw material for urea, about 70% of its cost.
#  When gas rises, urea rises.


# =============================================================================
#%%STEP 2 — TRANSFORM PRICES INTO MONTHLY % CHANGES
# =============================================================================
# Why: price LEVELS trend upward over time (non-stationary). Regressing one trending
# series on another gives a "spurious regression" (high R², significant coefficients
# even if the series are unrelated). Monthly CHANGES are stationary, so OLS is valid.
#
# How: 100 * [log(P_t) - log(P_t-1)]  ≈  % change from last month.
#   np.log(...)  -> natural log of every value
#   .diff()      -> this month's value minus last month's value
#   100 *        -> expressed in percent
# The first month becomes empty (NaN): it has no previous month.

# Create a new, empty-indexed table that will hold the regression variables.


# Use logs -> everything is in proportional terms, so variables with different units can be compared.
# Coefficients become elasticities.

# difference -> The question is about shocks, not levels. The thesis asks: when fertilizer prices jump, 
# do food prices follow? That's a relationship between changes.

# the log difference is approximate for a % change.
# for small changes, 100 × [ln(Pₜ) − ln(Pₜ₋₁)] = 100 × ln(Pₜ / Pₜ₋₁). For small changes, ln(1 + x) ≈ x


# COVID => dummy: It's 1 from March 2020 to December 2021, and 0 in all other months
# (WHO declares the pandemic on 11 March 2020).
# Unlike Hormuz, it is a WINDOW dummy (it switches back to 0): the pandemic shock is over,
# and ending in December 2021 keeps it separate from the Ukraine war (March 2022).

# The regression compares the pandemic months (1) with all the other months (0).
# Its coefficient shows whether food inflation was different during COVID
# (lockdowns, supply-chain and shipping disruptions, demand swings),
# beyond what urea, gas and oil explain.

data = pd.DataFrame({
    # Dependent variable: monthly % change of the food index.
    "food": 100 * np.log(df[FOOD]).diff(),

    # Control 1: monthly % change of the European gas price.
    "gas": 100 * np.log(df["wb_ngas_eu"]).diff(),

    # Control 2: monthly % change of the Brent oil price.
    "brent": 100 * np.log(df["wb_brent"]).diff(),

    # COVID dummy: 1 between the first and last month of the pandemic window, 0 otherwise.
    #   df.index                          -> the list of months
    #   >= config.COVID_START             -> True from March 2020
    #   <= config.COVID_END               -> True up to December 2021
    #   &                                 -> both must be True (inside the window)
    #   .astype(int)                      -> True/False becomes 1/0
    "covid": ((df.index >= config.COVID_START) & (df.index <= config.COVID_END)).astype(int),
})

print(data)

# =============================================================================
# %% STEP 3 — CREATE THE UREA LAGS
# =============================================================================
# Loop over k = 0, 1, 2, ..., 6  (range(7) stops before 7).

# urea_lag6: the change six months ago.

for k in range(LAGS + 1):
    # 1) 100 * np.log(urea).diff()  -> monthly % change of urea (as in step 2)
    # 2) .shift(k)                  -> moves the column DOWN by k rows, so the row of
    #                                  month t contains urea's change from month t-k
    # 3) saved as a new column named urea_lag0 (this month), urea_lag1 (last month), ...
    #    (f"urea_lag{k}" is an f-string: {k} is replaced by the number)
    data[f"urea_lag{k}"] = 100 * np.log(df["wb_urea"]).diff().shift(k)

# Remove every row with at least one missing value.
# The first months have no previous values (no difference, no lags), so they are dropped
# and the sample starts in August 2000. OLS cannot use incomplete rows.
data = data.dropna()


# =============================================================================
# %% STEP 4 — ESTIMATE THE REGRESSION
# =============================================================================

# OLS measures how much each driver moves food prices, holding the others fixed.
# Newey-West errors correct for autocorrelation in the errors

# this month's food inflation depends on

# fertilizer price changes this month and in the past 6 months (the channel being tested);
# this month's gas and oil price changes (the energy channels, as controls);
# whether we are in the COVID period (anything else specific to the pandemic);
# a constant (the intercept);
# and everything else, which ends up in the error term


# Build the text "urea_lag0 + urea_lag1 + ... + urea_lag6".
# Written once here because it is used twice (in the model and in the test below).
urea = " + ".join(f"urea_lag{k}" for k in range(LAGS + 1))

# Define the model (nothing is estimated yet):
#   "food ~ ..."  -> food is explained by ("~") the variables on the right
#   the intercept a is added automatically by statsmodels
#   data=data     -> the table where these column names are found
model = smf.ols(f"food ~ {urea} + gas + brent + covid", data=data)

# Estimate by least squares.
#   cov_type="HAC" -> Newey-West standard errors, robust to Heteroskedasticity and
#                     AutoCorrelation. Monthly commodity shocks persist, so errors are
#                     correlated over time; plain OLS standard errors would be too small.
#                     The COEFFICIENTS are the same as plain OLS; only standard errors
#                     and p-values change.
#   maxlags=6      -> the correction allows errors to be correlated up to 6 months apart.
result = model.fit(cov_type="HAC", cov_kwds={"maxlags": 6})

# Print the full regression table: coefficients, standard errors, p-values,
# R², and the Durbin-Watson statistic (close to 2 = no autocorrelation; below 2 = positive
# autocorrelation in the errors).
print(result.summary())

# Save the same table as a text file in outputs/tables/, one file per food index
# (e.g. outputs/tables/ols_covid_fao_ffpi.txt), so it is overwritten each run, not duplicated.
table_dir = config.PROJECT_DIR / "outputs" / "tables"
table_dir.mkdir(parents=True, exist_ok=True)
(table_dir / f"ols_covid_{FOOD}.txt").write_text(result.summary().as_text())


# =============================================================================
#%% STEP 5 — TOTAL EFFECT OF UREA AFTER 6 MONTHS
# =============================================================================

# Test the hypothesis  b0 + b1 + ... + b6 = 0  ("urea has no effect in total").
# Returns the estimated sum (cumulative pass-through), its standard error and p-value.
# The individual lag coefficients are imprecise because the lags are correlated with each
# other (multicollinearity), but their SUM is estimated precisely: this is the number to report.
total = result.t_test(f"{urea} = 0")

# Print the headline result in one sentence.
#   \n          -> blank line before
#   .item()     -> extracts the plain number from a 1x1 array
#   :.3f        -> rounds to 3 decimals
print(f"\n{FOOD}: a +1% urea price change moves food prices by "
      f"{total.effect.item():.3f}% after {LAGS} months (p = {total.pvalue.item():.3f})")

# Print the sample actually used.
#   data.index[0]  -> first month, data.index[-1] -> last month
#   :%Y-%m         -> date formatted as 2000-08
#   len(data)      -> number of rows = number of observations
print(f"Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}, {len(data)} months")

# %% STEP 7 — SHOW AN ACADEMIC-STYLE TABLE IN PYTHON
# =============================================================================
# Builds the same table as the LaTeX step (coefficients, stars, standard errors,
# total urea effect, statistics) and
#   (a) displays it in the interactive window / notebook, styled like a paper table
#   (b) saves it as an image: outputs/figures/ols_table_covid_<FOOD>.png (for slides or a quick look)

import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson

# Readable row names (plain text, matplotlib's math mode gives the LaTeX look)
NAMES = {f"urea_lag{k}": (f"$\\Delta\\ln$ Urea$_{{t-{k}}}$" if k else "$\\Delta\\ln$ Urea$_{t}$")
         for k in range(LAGS + 1)}
NAMES.update({"gas": "$\\Delta\\ln$ Gas (Europe)$_t$", "brent": "$\\Delta\\ln$ Brent$_t$",
              "covid": "COVID dummy$_t$", "Intercept": "Constant"})
TITLES = {"fao_ffpi": "Food Price Index", "fao_cereals": "Cereals", "fao_vegoils": "Vegetable oils",
          "fao_dairy": "Dairy", "fao_meat": "Meat", "fao_sugar": "Sugar"}


def stars(p):
    """*** 1%, ** 5%, * 10%"""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ---- 1. Build the table as a list of rows: (label, value) -----------------
# Each regressor takes two rows: coefficient with stars, then (standard error).
rows = []
for name in result.params.index:
    rows.append((NAMES.get(name, name), f"{result.params[name]:.3f}{stars(result.pvalues[name])}"))
    rows.append(("", f"({result.bse[name]:.3f})"))

# Total urea effect = sum of the lag coefficients, with its own standard error
total = result.t_test(f"{urea} = 0")
first_total = len(rows)
rows.append((f"Total urea effect ($\\Sigma_{{k=0}}^{{{LAGS}}}\\, b_k$)",
             f"{total.effect.item():.3f}{stars(total.pvalue.item())}"))
rows.append(("", f"({total.sd.item():.3f})"))

# Model statistics
first_stat = len(rows)
rows += [("Observations", f"{int(result.nobs)}"),
         ("Sample", f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"),
         ("$R^2$", f"{result.rsquared:.3f}"),
         ("Adjusted $R^2$", f"{result.rsquared_adj:.3f}"),
         ("Durbin-Watson", f"{durbin_watson(result.resid):.2f}")]

table = pd.DataFrame(rows, columns=["", TITLES.get(FOOD, FOOD)])

# ---- 2. Draw it like a LaTeX booktabs table --------------------------------
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],  # Times-like font of academic papers
                     "mathtext.fontset": "stix"})                            # matching math symbols

n = len(rows)
line_h = 0.28                               # height of one row (inches)
fig, ax = plt.subplots(figsize=(6, (n + 6) * line_h))
ax.set_xlim(0, 1)
ax.set_ylim(n + 5, 0)                       # row 0 at the top
ax.axis("off")

x_label, x_value = 0.02, 0.75               # column positions
y = 0

# Title
ax.text(0.5, y, "Table: Distributed-lag OLS estimates — COVID-19",
        ha="center", va="center", fontsize=11)
y += 1.2

# Top rule (thick), column header, rule under the header (thin)
ax.axhline(y - 0.5, xmin=0, xmax=1, color="black", lw=1.4)
ax.text(x_value, y, TITLES.get(FOOD, FOOD), ha="center", va="center", fontsize=10)
y += 1
ax.axhline(y - 0.5, color="black", lw=0.7)

# Body rows, with a thin rule before the total effect and before the statistics
for i, (label, value) in enumerate(rows):
    if i in (first_total, first_stat):
        ax.axhline(y - 0.5, color="black", lw=0.7)
    ax.text(x_label, y, label, ha="left", va="center", fontsize=10)
    # replace "-" by a real minus sign "−" (typographic, like LaTeX)
    ax.text(x_value, y, value.replace("-", "\u2212") if value[:1] == "-" else value,
            ha="center", va="center", fontsize=10)
    y += 1

# Bottom rule (thick) and notes
ax.axhline(y - 0.5, color="black", lw=1.4)
ax.text(x_label, y + 0.5,
        "Notes: prices enter as 100 × Δln (monthly % change). Newey-West (HAC) standard errors\n"
        "with 6 lags in parentheses. ***, **, *: significant at the 1%, 5%, 10% level.\n"
        "Sources: FAO, World Bank Pink Sheet.",
        ha="left", va="top", fontsize=8, style="italic")

plt.tight_layout()

# Save the image and show it in the interactive window
fig_dir = config.PROJECT_DIR / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / f"ols_table_covid_{FOOD}.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"Table image saved to {fig_dir / f'ols_table_covid_{FOOD}.png'}")


# %%
