import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "visuals", "similarity")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


# laod & merge data
df = pd.read_csv(os.path.join(SCRIPT_DIR, "..", "results", "similarity_results.csv"))
length = pd.read_csv(os.path.join(SCRIPT_DIR, "..", "results", "length_results.csv"))[
    ["ticker", "year", "size"]
].drop_duplicates()
df = df.merge(length, on=["ticker", "year"], how="left")

sim = df[df["yoy_similarity"].notna()].copy()
sim["category"] = sim["yoy_similarity"].apply(
    lambda x: "High Similarity" if x >= 0.75 else "Low Similarity"
)

# style
sns.set_theme(style="whitegrid", font_scale=1.15)
BLUE = "#2E75B6"
ORANGE = "#ED7D31"
GRAY = "#7F7F7F"

SECTOR_ORDER = [
    "Semiconductors",
    "Retail & E-Commerce",
    "Healthcare",
    "Technology",
    "Finance",
    "Cybersecurity",
    "Consumer Goods",
]
SIZE_ORDER = ["Small", "Medium", "Large"]


# TAable: descriptive statistics: overall + by year
rows = []

d = sim["yoy_similarity"]
rows.append(
    {
        "Year": "All years",
        "N": len(d),
        "Mean": round(d.mean(), 3),
        "Median": round(d.median(), 3),
        "SD": round(d.std(), 3),
        "Min": round(d.min(), 3),
        "Max": round(d.max(), 3),
        "High Similarity (≥0.75)": f"{(d >= 0.75).sum()} ({(d >= 0.75).mean() * 100:.0f}%)",
        "Low Similarity (<0.75)": f"{(d < 0.75).sum()} ({(d < 0.75).mean() * 100:.0f}%)",
    }
)

for year in sorted(sim["year"].unique()):
    d = sim[sim["year"] == year]["yoy_similarity"]
    rows.append(
        {
            "Year": year,
            "N": len(d),
            "Mean": round(d.mean(), 3),
            "Median": round(d.median(), 3),
            "SD": round(d.std(), 3),
            "Min": round(d.min(), 3),
            "Max": round(d.max(), 3),
            "High Similarity (≥0.75)": f"{(d >= 0.75).sum()} ({(d >= 0.75).mean() * 100:.0f}%)",
            "Low Similarity (<0.75)": f"{(d < 0.75).sum()} ({(d < 0.75).mean() * 100:.0f}%)",
        }
    )

table = pd.DataFrame(rows)
table.to_csv(out("table_similarity_stats.csv"), index=False)
print("Saved table_similarity_stats.csv")
print(table.to_string(index=False))
print()


def stacked_bar(ax, crosstab_pct, colors, xlabel, horizontal=True):
    """Helper: draw a stacked bar chart from a normalised crosstab (%)."""
    ct = crosstab_pct[["Low Similarity", "High Similarity"]]
    bottom = np.zeros(len(ct))
    for cat, color in [("Low Similarity", ORANGE), ("High Similarity", BLUE)]:
        vals = ct[cat].values
        if horizontal:
            bars = ax.barh(
                ct.index,
                vals,
                left=bottom,
                color=color,
                alpha=0.85,
                edgecolor="white",
                height=0.55,
                label=cat,
            )
            for bar, val, bot in zip(bars, vals, bottom):
                if val > 6:
                    ax.text(
                        bot + val / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white",
                        fontweight="bold",
                    )
        else:
            bars = ax.bar(
                ct.index,
                vals,
                bottom=bottom,
                color=color,
                alpha=0.85,
                edgecolor="white",
                width=0.5,
                label=cat,
            )
            for bar, val, bot in zip(bars, vals, bottom):
                if val > 5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bot + val / 2,
                        f"{val:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white",
                        fontweight="bold",
                    )
        bottom += vals
    if horizontal:
        ax.set_xlim(0, 110)
        ax.set_xlabel(xlabel)
    else:
        ax.set_ylim(0, 110)
        ax.set_ylabel(xlabel)
    ax.legend(loc="lower right", fontsize=10)


# Fig: Distribution of similarity scores
fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(
    sim["yoy_similarity"],
    bins=30,
    color=BLUE,
    alpha=0.45,
    edgecolor="white",
    linewidth=0.5,
    density=True,
)
try:
    sim["yoy_similarity"].plot.kde(ax=ax, color=BLUE, lw=2)
except Exception:
    pass

ax.axvline(0.75, color="red", lw=1.8, ls="--", label="Threshold = 0.75")
ax.axvline(
    sim["yoy_similarity"].mean(),
    color="black",
    lw=1.4,
    ls="--",
    label=f"Mean = {sim['yoy_similarity'].mean():.3f}",
)
ax.axvline(
    sim["yoy_similarity"].median(),
    color=GRAY,
    lw=1.4,
    ls=":",
    label=f"Median = {sim['yoy_similarity'].median():.3f}",
)

ax.axvspan(0, 0.75, alpha=0.06, color=ORANGE, label="Low Similarity (<0.75, n=37)")
ax.axvspan(0.75, 1.05, alpha=0.06, color=BLUE, label="High Similarity (≥0.75, n=86)")

ax.set_xlabel("Cosine Similarity Score")
ax.set_ylabel("Density")
ax.set_xlim(0, 1.05)
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig(out("similarity_dist.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved similarity_dist.png")


# Fig: Stacked bar: category share by year
year_ct = pd.crosstab(sim["year"], sim["category"], normalize="index") * 100

fig, ax = plt.subplots(figsize=(8, 5))
stacked_bar(ax, year_ct, None, xlabel="Share of Filing Pairs (%)", horizontal=False)
ax.set_xticks([2023, 2024, 2025])
ax.set_xticklabels(["2022→2023", "2023→2024", "2024→2025"])
ax.set_ylabel("Share of Filing Pairs (%)")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(out("similarity_by_year.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved similarity_by_year.png")


# Fig: Stacked bar: category share by sector
sector_ct = pd.crosstab(sim["sector"], sim["category"], normalize="index") * 100
sector_ct = sector_ct.reindex(SECTOR_ORDER)

fig, ax = plt.subplots(figsize=(10, 6))
stacked_bar(ax, sector_ct, None, xlabel="Share of Filing Pairs (%)", horizontal=True)
plt.tight_layout()
plt.savefig(out("similarity_by_sector.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved similarity_by_sector.png")


# Fig: Stacked bar: category share by firm size
size_ct = pd.crosstab(sim["size"], sim["category"], normalize="index") * 100
size_ct = size_ct.reindex(SIZE_ORDER)

fig, ax = plt.subplots(figsize=(8, 4.5))
stacked_bar(ax, size_ct, None, xlabel="Share of Filing Pairs (%)", horizontal=True)
plt.tight_layout()
plt.savefig(out("similarity_by_size.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved similarity_by_size.png")


# Fig: Category split by has_1c
ic_ct = pd.crosstab(sim["has_1c"], sim["category"], normalize="index") * 100
ic_ct.index = ["Without Item 1C", "With Item 1C"]

fig, ax = plt.subplots(figsize=(8, 4))
stacked_bar(ax, ic_ct, None, xlabel="Share of Filing Pairs (%)", horizontal=True)
plt.tight_layout()
plt.savefig(out("similarity_1c.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved similarity_1c.png")

print("\nAll similarity outputs saved successfully.")
