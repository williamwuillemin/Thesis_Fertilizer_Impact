"""
model_panel.py — crop panel with a soybean placebo: does a nitrogen fertilizer price shock
raise the prices of nitrogen-intensive crops MORE than the price of soybeans?

THE IDEA IN ONE PARAGRAPH
    The problem of every time-series model so far: oil, gas, the dollar and fertilizer prices
    move together, so a regression of the food index on fertilizer prices cannot tell the
    fertilizer channel apart from the energy channel.
    Here we use crops instead of time to identify the effect. Maize, wheat and rice need a lot
    of nitrogen fertilizer; soybeans are a legume and fix their own nitrogen from the air, so
    they need almost none. A nitrogen price shock should therefore raise the cost of producing
    maize, wheat and rice, but hardly that of soybeans. All four crops face the SAME oil price,
    the SAME dollar, the SAME world demand and the SAME Hormuz closure in a given month. Comparing
    them month by month removes all those common shocks: what remains is the extra response of
    nitrogen-intensive crops, i.e. the fertilizer channel. This is a difference-in-differences
    logic (treated = N-intensive crops, control = soybeans, "treatment intensity" = the size of
    the nitrogen price change that month).

MODEL (local projections on a panel, Jorda 2005; one regression per horizon h = 0, 1, ..., 12)

    100*[ln P_{c,t+h} - ln P_{c,t-1}] = theta_h * (dlnN_t  x  Nint_c)
                                      + phi_h  * (dlnN_{t-1} x Nint_c)
                                      + rho_h  * dlnP_{c,t-1}
                                      + mu_c   (crop fixed effect)
                                      + tau_t  (month fixed effect)
                                      + e_{c,t+h}

    P_{c,t}   World Bank price of crop c in month t ($/tonne)
    dlnN_t    100 * monthly log change of the US PPI for nitrogen fertilizer (FRED)
    Nint_c    1 for maize, wheat, rice; 0 for soybeans
    theta_h   THE NUMBER OF INTEREST: extra cumulative % change of an N-intensive crop price,
              relative to soybeans, h months after a +1% nitrogen price shock

Usage (from the repository root, after the database is filled):
    python src/model_panel.py
Outputs:
    outputs/tables/panel_results.txt     table: baseline + robustness, all horizons
    outputs/tables/panel_results.csv     same numbers, machine-readable
    outputs/figures/panel_irf.png        impulse-response figure
"""

# =============================================================================
# %% IMPORTS
# =============================================================================
import numpy as np                      # logs
import pandas as pd                     # tables
import matplotlib.pyplot as plt         # figure

# linearmodels: panel econometrics for Python (pip install linearmodels).
# Why not statsmodels as in model.py? statsmodels has no Driscoll-Kraay standard errors
# and no efficient two-way fixed effects; PanelOLS does both (see STEP 5).
from linearmodels.panel import PanelOLS

import config                           # paths, HORMUZ_START
from database import monthly_panel      # reads the DuckDB database


# =============================================================================
# %% SETTINGS — every choice is explained here
# =============================================================================

# ---- The crops and their nitrogen intensity ---------------------------------
# Why World Bank prices and not the FAO sub-indices?
#   The FAO Cereals index mixes maize, wheat and rice into one number and there is no FAO
#   "soybean" index (soybeans sit inside the vegetable-oils and cereals-adjacent indices).
#   A placebo design needs SEPARATE prices for treated and control crops, measured the same
#   way. The Pink Sheet gives all four in $/tonne, monthly, from one source, since 1960.
#
# Why these four?
#   maize, wheat, rice: the three staple cereals; nitrogen is their main fertilizer
#   (roughly 150-200 kg N/ha for US maize, 100+ for wheat and rice; FAO/IFA statistics).
#   soybeans: the textbook low-nitrogen crop (biological fixation by root bacteria),
#   AND it trades on the same exchanges, with the same traders, shipped on the same
#   dry-bulk ships as maize and wheat. A good control must be exposed to everything
#   EXCEPT the treatment: soybeans are the best available match.
#
# Why a 0/1 dummy rather than kg N per hectare?
#   A continuous intensity would need external numbers (which country? which year?) that a
#   referee could dispute, and would assume the effect is proportional to kg N/ha. The dummy
#   only assumes "cereals need N, soybeans do not", which is uncontroversial.
#   (A continuous version is a natural robustness check later.)
CROPS = {
    "wb_maize": 1,       # maize (US Gulf)
    "wb_wheat_hrw": 1,   # wheat, US hard red winter
    "wb_rice": 1,        # rice, Thai 5% broken
    "wb_soybeans": 0,    # soybeans (US) = placebo / control crop
}
CROP_NAMES = {"wb_maize": "Maize", "wb_wheat_hrw": "Wheat", "wb_rice": "Rice", "wb_soybeans": "Soybeans"}

# ---- The fertilizer shock ---------------------------------------------------
# US Producer Price Index, nitrogenous fertilizer manufacturing (BLS, via FRED).
# Why not the World Bank urea price used in model.py?
#   1) The Pink Sheet urea is an ORIGIN (FOB) price: Black Sea until 2022, Middle East after
#      -> series break in 03/2022.
#   2) After June 2026 it FELL back to ~$420/t while the strait was still closed: Gulf
#      producers who could not ship had to sell cheap locally. The origin price then tells us
#      about the seller's problem, not the buyer's cost. Farmers pay the DESTINATION price.
#   3) The PPI covers all nitrogen products (ammonia, urea, UAN), monthly, 2000-2026, no break.
SHOCK = "fred_ppi_nitrogen"

# ---- Horizons ----------------------------------------------------------------
# 0 to 12 months. Fertilizer is bought before planting and the crop is sold after harvest:
# one cycle is ~6-9 months. Morao (2025) finds the peak ~10 months after the shock and
# Hernandez et al. (2026) find the effect builds from month 4, so 12 months covers both.
H = 12

# ---- Sample ------------------------------------------------------------------
# Same window as the other models: Jan 2000 (config.START_DATE) to Sep 2026, the last month
# where all core series are available. The Hormuz months are INCLUDED: they are just more
# observations of nitrogen shocks; the month fixed effects absorb their common component.
START, END = "2000-01-01", "2026-09-01"


# =============================================================================
# %% STEP 1 — LOAD THE DATA
# =============================================================================
# Ask the database for the four crop prices, the nitrogen PPI, and Brent + the dollar
# (the last two are used only in the robustness check of STEP 6).
df = monthly_panel([*CROPS, SHOCK, "wb_brent", "fred_usd_reer"], start=START, end=END)

# Monthly % change of the nitrogen price: 100 * [ln N_t - ln N_{t-1}].
# Logs -> coefficients are elasticities (% response to a 1% change), units disappear.
# Differences -> the shock is a CHANGE, which is stationary (no spurious regression).
shock = 100 * np.log(df[SHOCK]).diff()
brent = 100 * np.log(df["wb_brent"]).diff()
reer = 100 * np.log(df["fred_usd_reer"]).diff()


# =============================================================================
# %% STEP 2 — BUILD THE PANEL FOR ONE HORIZON h
# =============================================================================
# A panel = one row per (crop, month): 4 crops x ~320 months ~ 1,280 rows.
# We rebuild it for each horizon because the dependent variable changes with h.

def build_panel(h: int) -> pd.DataFrame:
    frames = []
    for crop, nint in CROPS.items():
        lp = 100 * np.log(df[crop])            # 100 * log price of this crop

        frames.append(pd.DataFrame({
            "crop": crop,
            "date": df.index,

            # DEPENDENT VARIABLE: cumulative % change from the month BEFORE the shock (t-1)
            # to h months after (t+h).
            #   lp.shift(-h) -> value of month t+h placed on row t
            #   lp.shift(1)  -> value of month t-1 placed on row t
            # Why from t-1 and not from t? So that h = 0 already measures the impact in the
            # shock month itself (t-1 -> t). Why cumulative? theta_h is then directly
            # "how much higher is the price h months later", the impulse response, with no
            # need to add up lag coefficients as in model.py.
            "y": (lp.shift(-h) - lp.shift(1)).values,

            # THE REGRESSOR OF INTEREST: nitrogen shock x N-intensity dummy.
            # For soybeans this is always 0; for cereals it equals the shock.
            # Why only the interaction and not the shock itself? The shock is the same for
            # all crops in a month, so it is perfectly absorbed by the month fixed effect
            # (STEP 3): it cannot be estimated, and it does not need to be.
            "x": (shock * nint).values,

            # CONTROL 1: last month's interacted shock.
            # Nitrogen price changes are persistent (a rise this month often follows a rise
            # last month). Without this control, theta_h would partly pick up the response
            # to LAST month's shock. Standard in local projections (Jorda 2005; Ramey 2016).
            "x_l1": (shock.shift(1) * nint).values,

            # CONTROL 2: last month's own price change of the crop.
            # Crop prices have momentum; controlling for the pre-shock trend means theta_h
            # measures a change relative to where the price was already heading.
            "dy_l1": lp.diff().shift(1).values,

            # Used only in the robustness check: energy and dollar x crop (see STEP 6).
            "brent": brent.values, "reer": reer.values,
        }))
    panel = pd.concat(frames, ignore_index=True)

    # Rows with a missing value cannot be used: the first months (no lag) and the last h
    # months (no t+h yet).
    panel = panel.dropna(subset=["y", "x", "x_l1", "dy_l1"])

    # linearmodels expects a two-level index (entity, time) = (crop, month).
    return panel.set_index(["crop", "date"])


# =============================================================================
# %% STEP 3 — FIXED EFFECTS: WHY BOTH
# =============================================================================
# entity_effects=True -> mu_c, one constant per crop.
#     Crops have different average price growth (rice is more managed by governments, maize
#     has an ethanol demand trend...). mu_c removes those permanent differences, so they are
#     not confused with the fertilizer effect.
#
# time_effects=True -> tau_t, one constant per MONTH (~320 of them).
#     THIS IS THE KEY OF THE DESIGN. tau_t absorbs everything that hits all four crops in the
#     same month in the same way: oil and gas prices, the dollar, world demand, freight costs,
#     the 2008 and 2022 commodity booms, and the Hormuz closure itself. That is why there is no
#     Brent, gas or dollar control in the baseline: they are already in tau_t.
#     What tau_t does NOT absorb: shocks that hit cereals and soybeans DIFFERENTLY. Those are
#     the threats to the design, handled in STEP 6 and discussed in the notes at the end.


# =============================================================================
# %% STEP 4 — STANDARD ERRORS: WHY DRISCOLL-KRAAY
# =============================================================================
# Two problems with ordinary standard errors here:
#   1) Cross-crop correlation: in a given month the four crop errors are correlated (they
#      share weather, policy news...). Rows are not independent.
#   2) Serial correlation: the dependent variable is a change over h+1 months, so
#      consecutive months OVERLAP (Jan->Jul and Feb->Aug share 5 months). Errors are
#      mechanically autocorrelated up to h months apart.
# Clustering by crop would fix (2) but needs many clusters; we only have 4 -> invalid.
# Driscoll & Kraay (1998) standard errors handle BOTH problems and are valid with few
# entities and a long time dimension (T ~ 320), exactly our case. They are a Newey-West
# correction applied to the cross-sectional averages.
#   kernel="bartlett"  -> Newey-West weights
#   bandwidth=h+1      -> allow correlation up to h+1 months, the length of the overlap

def estimate(h: int, robust: bool = False):
    p = build_panel(h)
    regressors = ["x", "x_l1", "dy_l1"]
    if robust:
        # STEP 6 robustness: each crop gets its OWN response to oil and to the dollar.
        # Built as (Brent change x crop dummy) and (dollar change x crop dummy), soybeans
        # being the reference (their slope is absorbed with the month effects).
        crop = p.index.get_level_values("crop")
        for c in CROPS:
            if c == "wb_soybeans":
                continue
            p[f"brent_{c}"] = p["brent"] * (crop == c)
            p[f"reer_{c}"] = p["reer"] * (crop == c)
            regressors += [f"brent_{c}", f"reer_{c}"]
    model = PanelOLS(p["y"], p[regressors], entity_effects=True, time_effects=True)
    return model.fit(cov_type="kernel", kernel="bartlett", bandwidth=h + 1)


# =============================================================================
# %% STEP 5 — RUN ALL HORIZONS
# =============================================================================
def stars(p):
    """*** 1%, ** 5%, * 10%"""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


rows = []
for spec, robust in {"baseline": False, "crop_energy_controls": True}.items():
    for h in range(H + 1):
        r = estimate(h, robust)
        rows.append({"spec": spec, "h": h,
                     "theta": r.params["x"], "se": r.std_errors["x"], "p": r.pvalues["x"],
                     "nobs": int(r.nobs),
                     "first": r.model.dependent.index.get_level_values("date").min(),
                     "last": r.model.dependent.index.get_level_values("date").max()})
res = pd.DataFrame(rows)


# =============================================================================
# %% STEP 6 — ROBUSTNESS: CROP-SPECIFIC ENERGY AND DOLLAR EXPOSURE (why this one)
# =============================================================================
# The month fixed effects assume oil and the dollar move all four crops EQUALLY. That is
# not quite true: maize is ethanol feedstock, soybeans (via soy oil) biodiesel feedstock,
# so an oil shock can move them differently. And nitrogen prices rise WITH gas and oil.
# If cereals react more to oil than soybeans do, theta_h could pick up oil instead of
# fertilizer. The "crop_energy_controls" specification lets each crop have its own slope
# on Brent and on the dollar. If theta_h survives, the energy channel is not driving it.
# (Already estimated in STEP 5 with robust=True.)


# =============================================================================
# %% STEP 7 — TABLE
# =============================================================================
lines = ["Crop panel with soybean placebo: extra cumulative % response of maize, wheat and rice",
         "relative to soybeans, h months after a +1% change in the US PPI for nitrogen fertilizer.",
         "Crop and month fixed effects; Driscoll-Kraay s.e. (Bartlett, bandwidth h+1) in parentheses.",
         "*** 1%, ** 5%, * 10%.", "",
         f"{'h':>3}   {'Baseline':<24}{'+ crop x (Brent, USD)':<24}{'obs':>6}",
         "-" * 60]
for h in range(H + 1):
    b = res[(res.spec == "baseline") & (res.h == h)].iloc[0]
    r = res[(res.spec == "crop_energy_controls") & (res.h == h)].iloc[0]
    lines.append(f"{h:>3}   {b.theta:>7.3f}{stars(b.p):<3} ({b.se:.3f})     "
                 f"{r.theta:>7.3f}{stars(r.p):<3} ({r.se:.3f})     {b.nobs:>6}")
b0 = res[(res.spec == "baseline") & (res.h == 0)].iloc[0]
lines += ["-" * 60,
          f"Crops: {', '.join(CROP_NAMES[c] for c, n in CROPS.items() if n)} (N-intensive) vs "
          f"{', '.join(CROP_NAMES[c] for c, n in CROPS.items() if not n)} (placebo).",
          f"Sample at h=0: {b0['first']:%Y-%m} to {b0['last']:%Y-%m}; the last h months are lost at horizon h."]
table = "\n".join(lines)
print(table)

out = config.PROJECT_DIR / "outputs"
(out / "tables").mkdir(parents=True, exist_ok=True)
(out / "tables" / "panel_results.txt").write_text(table)
res.to_csv(out / "tables" / "panel_results.csv", index=False)


# =============================================================================
# %% STEP 8 — LATEX TABLE FOR THE PAPER
# =============================================================================
# Reads panel_results.csv back (so this cell also runs on its own once the CSV exists) and
# writes a booktabs table: one row per horizon, coefficient with stars, standard error in
# parentheses on the line below. Needs in the LaTeX preamble:
#   \usepackage{booktabs}  \usepackage{threeparttable}
tab = pd.read_csv(config.PROJECT_DIR / "outputs" / "tables" / "panel_results.csv",
                  parse_dates=["first", "last"])
base = tab[tab.spec == "baseline"].set_index("h")
rob = tab[tab.spec == "crop_energy_controls"].set_index("h")


def coef(row):
    """0.443^{**} -> stars as superscripts, in math mode so they align nicely."""
    s = stars(row.p)
    return f"${row.theta:.3f}" + (f"^{{{s}}}$" if s else "$")


tex = [r"\begin{table}[htbp]",
       r"\centering",
       r"\begin{threeparttable}",
       r"\caption{Response of nitrogen-intensive crop prices relative to soybeans after a nitrogen fertilizer price shock}",
       r"\label{tab:panel_lp}",
       r"\begin{tabular}{l c c c}",
       r"\toprule",
       r" & (1) & (2) & \\",
       r"Horizon $h$ & Baseline & + Crop $\times$ & Obs. \\",
       r"(months) & & (Brent, USD) & \\",
       r"\midrule"]
for h in base.index:
    b, r = base.loc[h], rob.loc[h]
    tex.append(f"{h} & {coef(b)} & {coef(r)} & {int(b.nobs):,} \\\\")
    tex.append(f" & ({b.se:.3f}) & ({r.se:.3f}) & \\\\")
    if h < base.index.max():
        tex.append(r"\addlinespace[2pt]")
tex += [r"\midrule",
        r"Crop fixed effects & Yes & Yes & \\",
        r"Month fixed effects & Yes & Yes & \\",
        r"Crop $\times$ (Brent, USD) & No & Yes & \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item \textit{Notes:} Panel local projections (Jord\`a, 2005). Each cell reports $\theta_h$, "
        r"the extra cumulative percentage change in the price of maize, wheat and rice relative to "
        r"soybeans between $t-1$ and $t+h$, following a 1\% increase in the US producer price index "
        r"for nitrogen fertilizer. All regressions control for the lagged interacted shock and the "
        r"lagged own price change. Driscoll--Kraay standard errors (Bartlett kernel, bandwidth $h+1$) "
        f"in parentheses. Sample: {base.loc[0, 'first']:%Y-%m} to {base.loc[0, 'last']:%Y-%m} at $h=0$; "
        r"the last $h$ months are lost at horizon $h$. "
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}"]
latex = "\n".join(tex)
print(latex)
(config.PROJECT_DIR / "outputs" / "tables" / "panel_results.tex").write_text(latex + "\n")


# =============================================================================
# %% STEP 9 — FIGURE
# =============================================================================
# Impulse response with a 90% confidence band (+-1.645 s.e.): the usual band in the
# local-projection literature, because individual horizons are noisy.
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for spec, color, label in [("baseline", "black", "Baseline"),
                           ("crop_energy_controls", "#c0392b", "+ crop-specific Brent and USD")]:
    s = res[res.spec == spec]
    ax.plot(s.h, s.theta, color=color, lw=1.8, label=label, marker="o", ms=3)
    if spec == "baseline":
        ax.fill_between(s.h, s.theta - 1.645 * s.se, s.theta + 1.645 * s.se, color="grey", alpha=.25,
                        label="90% CI (baseline)")
ax.axhline(0, color="black", lw=.6)
ax.set_xlabel("Months after the nitrogen price shock")
ax.set_ylabel("Extra % response vs soybeans\n(to +1% nitrogen PPI)")
ax.set_title("N-intensive crops vs soybeans: response to a nitrogen price shock", fontsize=10)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
(out / "figures").mkdir(parents=True, exist_ok=True)
fig.savefig(out / "figures" / "panel_irf.png", dpi=200)
print(f"\nSaved: {out / 'tables' / 'panel_results.txt'} and {out / 'figures' / 'panel_irf.png'}")


# =============================================================================
# NOTES FOR THE THESIS — what this design can and cannot claim
# =============================================================================
# * theta_h is a RELATIVE effect (cereals minus soybeans). If fertilizer also raised soybean
#   prices a little (soybeans use some P and K, and compete with maize for land), the total
#   effect on cereals is LARGER than theta_h: theta_h is a lower bound.
# * Land competition: dearer nitrogen pushes US farmers from maize to soybeans, which raises
#   the soybean supply and lowers its price. That widens the gap, so part of theta_h is this
#   substitution. It is still a fertilizer effect, but through land allocation, not only cost.
# * Remaining threats: shocks that hit cereals and soybeans differently AND coincide with
#   nitrogen shocks, e.g. China's soybean import demand, the 2022 Ukraine war (wheat and maize,
#   not soybeans, and gas at the same time). Robustness to add: drop 2022, leave one crop out,
#   use palm oil as a second low-N comparison.
# * Hormuz: the closure months are inside the sample but are 7 of ~320 months. The design
#   estimates the general pass-through; the Hormuz question is then whether the closure months
#   fit that pass-through (compare predicted vs actual cereal-soybean gap since 03/2026).

# %%
