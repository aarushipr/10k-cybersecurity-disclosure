import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join("..", "results", "nist_csf_scores.csv")
OUTPUT_DIR = os.path.join("..", "visuals", "balance")
THRESHOLD = 0.6

SIZE_MAP = {
    "AAPL": "Large",
    "MSFT": "Large",
    "GOOGL": "Large",
    "AMZN": "Large",
    "JNJ": "Large",
    "LLY": "Large",
    "NKE": "Large",
    "PEP": "Large",
    "V": "Large",
    "AMD": "Large",
    "INTC": "Large",
    "CRWD": "Large",
    "PANW": "Large",
    "MOH": "Large",
    "CROX": "Mid",
    "ELF": "Mid",
    "WGO": "Mid",
    "MCFT": "Mid",
    "PRGS": "Mid",
    "RPD": "Mid",
    "S": "Mid",
    "VRNS": "Mid",
    "HLI": "Mid",
    "LC": "Mid",
    "UPST": "Mid",
    "ELMD": "Mid",
    "MODD": "Mid",
    "VKTX": "Mid",
    "BOOT": "Mid",
    "ETSY": "Mid",
    "SFIX": "Mid",
    "UPWK": "Mid",
    "CRUS": "Mid",
    "MXL": "Mid",
    "POWI": "Mid",
    "AMPL": "Mid",
    "GTLB": "Mid",
    "SCSC": "Mid",
    "U": "Mid",
    "PSEC": "Small",
    "NVEC": "Small",
}
SIZE_ORDER = ["Large", "Mid", "Small"]
SIZE_COLORS = {"Large": "#2C5F8A", "Mid": "#5BB56E", "Small": "#E8A838"}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load
df = pd.read_csv(INPUT_FILE)
df["size"] = df["ticker"].map(SIZE_MAP).fillna("Mid")
df.loc[df["ticker"] == "NVEC", "balance_score"] = (
    np.nan
)  # exclude NVEC (anomalous results)
df_calc = df[df["ticker"] != "NVEC"].copy()  # exclude from all calculations
print(f"Loaded {len(df)} filings | Excluded NVEC from calculations")


def save(name):
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved {name}")


# Fig: Balance Score Distribution (histogram)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(
    df_calc["balance_score"],
    bins=20,
    color="#2C5F8A",
    edgecolor="white",
    linewidth=0.6,
    alpha=0.85,
)
ax.axvline(
    THRESHOLD,
    color="#D95F3B",
    linewidth=2,
    linestyle="--",
    label=f"Threshold = {THRESHOLD}",
)
n_pass = (df_calc["balance_score"] >= THRESHOLD).sum()
n_fail = len(df_calc) - n_pass
ax.text(
    THRESHOLD + 0.005,
    ax.get_ylim()[1] * 0.9,
    f"Pass: {n_pass}  Fail: {n_fail}",
    color="#D95F3B",
    fontsize=10,
)
ax.set_xlabel("Balance Score")
ax.set_ylabel("Number of Filings")
ax.legend(frameon=False)
plt.tight_layout()
save("balance_distribution")

# Fig: Balance Score Over Time
if "year" in df.columns and df["year"].nunique() > 1:
    yearly = df_calc.groupby("year")["balance_score"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        yearly["year"],
        yearly["mean"],
        marker="o",
        color="#2C5F8A",
        linewidth=2,
        markersize=6,
        label="Mean balance score",
    )
    ax.fill_between(
        yearly["year"],
        yearly["mean"] - yearly["std"],
        yearly["mean"] + yearly["std"],
        alpha=0.15,
        color="#2C5F8A",
        label="±1 std dev",
    )
    ax.axhline(
        THRESHOLD,
        color="#D95F3B",
        linewidth=1.5,
        linestyle="--",
        label=f"Threshold = {THRESHOLD}",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Balance Score")
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    save("balance_by_year")

# Fig: Balance Score by Sector (box plot)
sectors = sorted(df_calc["sector"].dropna().unique())
sector_colors = sns.color_palette("tab10", len(sectors))
sector_groups = [
    df_calc[df_calc["sector"] == s]["balance_score"].dropna().values for s in sectors
]

fig, ax = plt.subplots(figsize=(11, 5))
bp = ax.boxplot(
    sector_groups,
    patch_artist=True,
    widths=0.5,
    medianprops=dict(color="white", linewidth=2),
)
for patch, c in zip(bp["boxes"], sector_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.85)
ax.axhline(
    THRESHOLD,
    color="#D95F3B",
    linewidth=1.5,
    linestyle="--",
    label=f"Threshold = {THRESHOLD}",
)
for i, (s, g) in enumerate(zip(sectors, sector_groups), 1):
    if len(g):
        ax.text(
            i,
            max(g) + 0.015,
            f"n={len(g)}\nμ={np.mean(g):.2f}",
            ha="center",
            fontsize=7.5,
        )
ax.set_xticks(range(1, len(sectors) + 1))
ax.set_xticklabels(sectors, rotation=20, ha="right")
ax.set_ylabel("Balance Score")
ax.legend(frameon=False)
plt.tight_layout()
save("balance_by_sector")

# Fig: Balance Score by Firm Size (box plot)
size_groups = [
    df_calc[df_calc["size"] == s]["balance_score"].dropna().values for s in SIZE_ORDER
]

fig, ax = plt.subplots(figsize=(7, 4.5))
bp = ax.boxplot(
    size_groups,
    patch_artist=True,
    widths=0.45,
    medianprops=dict(color="white", linewidth=2),
)
for patch, s in zip(bp["boxes"], SIZE_ORDER):
    patch.set_facecolor(SIZE_COLORS[s])
    patch.set_alpha(0.85)
ax.axhline(
    THRESHOLD,
    color="#D95F3B",
    linewidth=1.5,
    linestyle="--",
    label=f"Threshold = {THRESHOLD}",
)
for i, (s, g) in enumerate(zip(SIZE_ORDER, size_groups), 1):
    if len(g):
        ax.text(
            i,
            max(g) + 0.015,
            f"n={len(g)}\nμ={np.mean(g):.3f}",
            ha="center",
            fontsize=9,
        )
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([f"{s} Cap" for s in SIZE_ORDER])
ax.set_ylabel("Balance Score")
ax.legend(frameon=False)
plt.tight_layout()
save("balance_by_size")


# Fig: Pass / Fail Count by Year (dual line chart)
if "year" in df_calc.columns and df_calc["year"].nunique() > 1:
    yearly_pf = (
        df_calc.groupby("year")["balance_score"]
        .apply(
            lambda x: pd.Series(
                {
                    "Pass": (x >= THRESHOLD).sum(),
                    "Fail": (x < THRESHOLD).sum(),
                }
            )
        )
        .unstack()
    )
    years = yearly_pf.index.tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        years,
        yearly_pf["Pass"],
        marker="o",
        color="#2C5F8A",
        lw=2.5,
        markersize=7,
        label=f"Pass (≥ {THRESHOLD})",
    )
    ax.plot(
        years,
        yearly_pf["Fail"],
        marker="o",
        color="#D95F3B",
        lw=2.5,
        markersize=7,
        label=f"Fail (< {THRESHOLD})",
    )

    for year, p, f in zip(years, yearly_pf["Pass"], yearly_pf["Fail"]):
        ax.annotate(
            str(int(p)),
            xy=(year, p),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color="#2C5F8A",
        )
        ax.annotate(
            str(int(f)),
            xy=(year, f),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color="#D95F3B",
        )

    ax.set_xticks(years)
    ax.set_xlim(min(years) - 0.1, max(years) + 0.1)
    ax.set_ylim(0, 41)
    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel("Number of Reports")
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    save("balance_pass_fail_by_year")

print("\nAll balance score figures saved.")
