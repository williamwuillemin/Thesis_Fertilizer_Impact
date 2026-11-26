"""
config.py — every setting of the data pipeline lives here.

Rule of thumb: if you ever need to change a URL, a date or add a series,
you should only have to touch THIS file, never the extraction code.
"""
from pathlib import Path
import os

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# config.py lives in src/, the repository root is one level up
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                          # every downloaded file is archived here (vintages)
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = PROCESSED_DIR / "thesis.duckdb"           # the local database (rebuildable -> not in Git)

# API keys are read from a .env file (never commit that file to Git!)
load_dotenv(PROJECT_DIR / ".env")
def _key(name):
    """Read a key from .env; empty values and template placeholders ('your_..._here') count as missing."""
    value = (os.getenv(name) or "").strip().strip('"').strip("'")
    return None if not value or value.lower().startswith("your_") else value


FRED_API_KEY = _key("FRED_API_KEY")          # free: https://fred.stlouisfed.org/docs/api/api_key.html
COMTRADE_API_KEY = _key("COMTRADE_API_KEY")  # free "comtrade - v1" product: https://comtradedeveloper.un.org
AMS_API_KEY = _key("AMS_API_KEY")            # free USDA MyMarketNews account: https://mymarketnews.ams.usda.gov

# ---------------------------------------------------------------------------
# Sample period (proposal: monthly, Jan 2000 → latest available)
# ---------------------------------------------------------------------------
START_DATE = "2000-01-01"

# Hormuz closure: strikes + IRGC warning on 28 Feb 2026 → first FULL month closed = March 2026.
# This is a modelling choice you should justify in the thesis (Feb vs Mar).
HORMUZ_START = "2026-03-01"

# War in Ukraine: Russian invasion on 24 Feb 2022 → first FULL month of war = March 2022.
# Used as a step dummy (1 from March 2022 onward), same logic as HORMUZ_START.
UKRAINE_START = "2022-03-01"

# COVID-19: WHO declares a pandemic on 11 Mar 2020 → first month = March 2020.
# Window dummy (1 only inside the period): ends December 2021, before the Ukraine war,
# so the two shocks are not mixed. Justify the end date in the thesis (alternatives:
# Feb 2022, or May 2023 = end of the WHO emergency).
COVID_START = "2020-03-01"
COVID_END = "2021-12-01"

# ---------------------------------------------------------------------------
# FAO Food Price Index (dependent variables)
# The page https://www.fao.org/worldfoodsituation/foodpricesindex/en/ links the files.
# The "?sfvrsn=..." part of the link changes every month, so we use the base path.
# If a download fails: open the page, right-click "CSV" and paste the new link here.
# ---------------------------------------------------------------------------
FAO_CSV_URL = (
    "https://www.fao.org/media/docs/worldfoodsituationlibraries/"
    "wfs-library/food_price_indices_data.csv"
)
# column name in the FAO file  ->  our series_id
FAO_SERIES = {
    "Food Price Index": "fao_ffpi",
    "Cereals": "fao_cereals",
    "Oils": "fao_vegoils",          # called "Oils" / "Vegetable Oils" depending on the vintage
    "Vegetable Oils": "fao_vegoils",
    "Dairy": "fao_dairy",
    "Meat": "fao_meat",
    "Sugar": "fao_sugar",
}

# ---------------------------------------------------------------------------
# World Bank Pink Sheet (fertilizers, energy, grains)
# The document id (74e8be...-0050012026) changes every YEAR. In 2027 look it up on
# https://www.worldbank.org/en/research/commodity-markets and update the URL.
# ---------------------------------------------------------------------------
WB_MONTHLY_URL = (
    "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/"
    "related/CMO-Historical-Data-Monthly.xlsx"
)
# Header text in the "Monthly Prices" sheet (lower-case, without '*' and spaces at the end)
#   -> (series_id, unit)
WB_SERIES = {
    # fertilizers (explanatory variables)
    "urea": ("wb_urea", "$/mt"),
    "dap": ("wb_dap", "$/mt"),
    "tsp": ("wb_tsp", "$/mt"),
    "phosphate rock": ("wb_phosrock", "$/mt"),
    "potassium chloride": ("wb_potash", "$/mt"),
    # energy (controls)
    "crude oil, brent": ("wb_brent", "$/bbl"),
    "crude oil, wti": ("wb_wti", "$/bbl"),
    "natural gas, us": ("wb_ngas_us", "$/mmbtu"),
    "natural gas, europe": ("wb_ngas_eu", "$/mmbtu"),
    "liquefied natural gas, japan": ("wb_lng_japan", "$/mmbtu"),
    # individual crops (useful for the "which commodity reacts most" question)
    "maize": ("wb_maize", "$/mt"),
    "wheat, us hrw": ("wb_wheat_hrw", "$/mt"),
    "rice, thai 5%": ("wb_rice", "$/mt"),
    "soybeans": ("wb_soybeans", "$/mt"),   # legume: needs little N fertilizer -> placebo!
    "soybean oil": ("wb_soyoil", "$/mt"),
    "palm oil": ("wb_palmoil", "$/mt"),
    "sugar, world": ("wb_sugar", "$/kg"),
}

# Nitrogen fertilizer application rate by crop (kg N per hectare), used by crop_panel.py
# to build the nitrogen intensity N_c (0 = soybeans, 1 = maize).
#   series_id: (crop name, kg N/ha)
# TODO: add the exact source (publication, year, table) of these rates for the thesis.
CROP_N_RATE = {
    "wb_maize": ("Maize", 118.8),
    "wb_rice": ("Rice", 113.8),
    "wb_wheat_hrw": ("Wheat", 97.8),
    "wb_soybeans": ("Soybeans", 16.0),   # legume: fixes its own nitrogen -> placebo crop
}

# ---------------------------------------------------------------------------
# FRED (St. Louis Fed) — needs FRED_API_KEY
# ---------------------------------------------------------------------------
FRED_SERIES = {
    # series_id_in_FRED: (our_series_id, description, unit)
    "IGREA": ("fred_igrea", "Kilian index of global real economic activity (Dallas Fed)", "% dev. from trend"),
    "CPIAUCSL": ("fred_us_cpi", "US CPI all items (to deflate to real prices)", "index 1982-84=100"),
    "RBUSBIS": ("fred_usd_reer", "Real broad effective exchange rate USA (BIS)", "index 2020=100"),
    # Long monthly fertilizer price series (US producer prices, BLS): cover the whole 2000-2026 sample.
    # US PRODUCER prices (mostly gas-based US plants), not import prices -> state this in the thesis.
    "PCU325311325311": ("fred_ppi_nitrogen", "US PPI nitrogenous fertilizer manufacturing (BLS)", "index Dec 1979=100"),
    "PCU325312325312P": ("fred_ppi_phosphate", "US PPI phosphatic fertilizer manufacturing, primary products (BLS)",
                         "index Dec 1979=100"),
}

# ---------------------------------------------------------------------------
# Weather / climate
# ---------------------------------------------------------------------------
NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
NOAA_RONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt"
# US drought (PDSI, contiguous US = location 110) from NOAA "Climate at a Glance".
# URL pattern documented by NCEI: .../{scope}/time-series/{location}/{parameter}/{surface}/{timescale}/{month}/{beg}-{end}/data.csv
# !! I could not test this exact URL from my side. If it returns 404, open
# https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series/110/pdsi
# choose the options, and copy the CSV download link here.
NOAA_PDSI_URL = (
    "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/"
    "national/time-series/110/pdsi/1/0/2000-2026/data.csv"
)

# ---------------------------------------------------------------------------
# Risk / supply-chain indices
# ---------------------------------------------------------------------------
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
GPR_COLUMNS = {"GPR": "gpr_global", "GPRT": "gpr_threats", "GPRA": "gpr_acts"}

GSCPI_URL = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"

# ---------------------------------------------------------------------------
# IMF PortWatch — daily transits through chokepoints (ArcGIS REST API, no key)
# ---------------------------------------------------------------------------
PORTWATCH_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Chokepoints_Data/FeatureServer/0/query"
)
# keyword used in a  portname LIKE '%keyword%'  filter  ->  short slug for our series ids
PORTWATCH_CHOKEPOINTS = {"Hormuz": "hormuz", "Bab el-Mandeb": "babelmandeb", "Suez": "suez"}

# ---------------------------------------------------------------------------
# UN Comtrade — monthly physical fertilizer trade (needs COMTRADE_API_KEY)
# Strategy: Gulf exporters report late or not at all, so we ALSO pull "mirror"
# data: what big importers declare as imports FROM the Gulf.
# ---------------------------------------------------------------------------
COMTRADE_URL = "https://comtradeapi.un.org/data/v1/get/C/M/HS"
COMTRADE_HS = {
    "310210": "Urea",
    "281410": "Anhydrous ammonia",
    "310530": "Diammonium phosphate (DAP)",
    "310540": "Monoammonium phosphate (MAP)",
}
# Comtrade numeric (M49-based) codes
GULF = {682: "Saudi Arabia", 634: "Qatar", 784: "United Arab Emirates", 512: "Oman",
        48: "Bahrain", 414: "Kuwait", 364: "Iran"}
BIG_IMPORTERS = {699: "India", 76: "Brazil", 842: "USA", 36: "Australia", 764: "Thailand"}
COMTRADE_START_YEAR = 2015   # monthly Comtrade data before ~2010 is patchy anyway

# ---------------------------------------------------------------------------
# Manual files (data without a free API, exported from a terminal: Bloomberg / LSEG)
# CSV with two columns  date,value  (daily or monthly) in data/manual/.
# Daily values are kept in the table market_daily AND averaged to monthly.
# data/manual/ is git-ignored: terminal data may not be redistributed.
# ---------------------------------------------------------------------------
MANUAL_DIR = DATA_DIR / "manual"
MANUAL_SERIES = {
    # filename                : (series_id, description, unit)
    "bdi.csv":                 ("bdi", "Baltic Dry Index", "index"),
    # suggested exports from Bloomberg (ticker in brackets) - add/remove freely:
    "urea_me_futures.csv":     ("fut_urea_me", "Urea granular FOB Middle East futures, front month (CME UME)", "$/t"),
    "urea_brazil_futures.csv": ("fut_urea_brazil", "Urea granular CFR Brazil futures, front month (CME UFB)", "$/t"),
    "urea_nola_futures.csv":   ("fut_urea_nola", "Urea granular FOB US Gulf/NOLA futures, front month (CME UFV)", "$/short t"),
    "urea_egypt_futures.csv":  ("fut_urea_egypt", "Urea granular FOB Egypt futures, front month (CME UFE)", "$/t"),
    "corn_futures.csv":        ("fut_corn", "CBOT corn futures, front month (C 1 Comdty)", "USc/bu"),
    "wheat_futures.csv":       ("fut_wheat", "CBOT wheat futures, front month (W 1 Comdty)", "USc/bu"),
    "soybean_futures.csv":     ("fut_soybeans", "CBOT soybean futures, front month (S 1 Comdty)", "USc/bu"),
}

# ---------------------------------------------------------------------------
# USDA AMS - Illinois Production Cost Report (report 3195, every two weeks)
# Retail fertilizer prices paid by farmers -> a DESTINATION price, and the only
# free ammonia price. Needs AMS_API_KEY (free MyMarketNews account).
# Note: the MARS API may only hold the electronic history (my guess: ~2019+);
# older reports exist only as PDFs. The code prints the first date it finds.
# ---------------------------------------------------------------------------
AMS_URL = "https://marsapi.ams.usda.gov/services/v1.2/reports/3195"
AMS_START = "2024-01-01"   # the API holds this report only from April 2024 (older: PDF archive)
# keyword found in the product description -> (series_id, description). Checked in this order.
# USDA names, e.g. "Anhydrous Ammonia", "Liquid Nitrogen (28-0-0)", "Urea (46-0-0)",
# "DAP (Diammonium Phosphate 18-46-0)", "MAP (Monoammonium Phosphate 11-52-0)", "Potash (Red 0-0-60)"
AMS_PRODUCTS = {
    "anhydrous":    ("ams_il_ammonia", "Illinois retail anhydrous ammonia (82% N)"),
    "28-0-0":       ("ams_il_uan28", "Illinois retail liquid nitrogen 28%"),
    "32-0-0":       ("ams_il_uan32", "Illinois retail liquid nitrogen 32%"),
    "urea":         ("ams_il_urea", "Illinois retail urea (46% N)"),
    "diammonium":   ("ams_il_dap", "Illinois retail DAP (18-46-0)"),
    "monoammonium": ("ams_il_map", "Illinois retail MAP (11-52-0 and 11-34-0)"),
    "potash (red":  ("ams_il_potash", "Illinois retail potash (red 0-0-60)"),
}

# ---------------------------------------------------------------------------
# USDA FAS PSD - world grain balance sheets (bulk CSV, no key) -> stocks-to-use
# !! I could not test this URL from my side. If it fails: open
# https://apps.fas.usda.gov/psdonline/app/index.html#/app/downloads and copy the
# link of "Grains" (CSV, zip) here.
# ---------------------------------------------------------------------------
PSD_URL = "https://apps.fas.usda.gov/psdonline/downloads/psd_grains_pulses_csv.zip"
PSD_COMMODITIES = {"Corn": "maize", "Wheat": "wheat", "Rice, Milled": "rice"}

# ---------------------------------------------------------------------------
# EIA Monthly Energy Review (CSV, no key) - biofuel channel
# table id -> {EIA series code (MSN column): (series_id, description)}
# Each table repeats the same description in 3 units (thousand barrels, million gallons,
# trillion Btu), so series are selected by their exact code.
# ---------------------------------------------------------------------------
EIA_MER_URL = "https://www.eia.gov/totalenergy/data/browser/csv.php?tbl={table}"
EIA_SERIES = {
    "T10.03":  {"ENPRPUS": ("eia_ethanol_prod", "US fuel ethanol production")},
    "T10.04A": {"BDPRPUS": ("eia_biodiesel_prod", "US biodiesel production")},
}

# ---------------------------------------------------------------------------
# Derived variables (computed from the database, no download)
# ---------------------------------------------------------------------------
# Hormuz intensity = 1 - transits / normal transits.  "Normal" = 12 pre-war months.
HORMUZ_BASELINE = ("2025-03-01", "2026-02-01")
# Gulf exporters OUTSIDE the strait (ports on the Gulf of Oman / Arabian Sea)
OUTSIDE_STRAIT = {512: "Oman"}
TRADE_PRODUCTS = {"310210": "urea", "281410": "ammonia"}
