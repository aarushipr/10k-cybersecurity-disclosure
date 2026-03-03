import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "visuals", "content")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


# Load & merge data
df = pd.read_excel(os.path.join(SCRIPT_DIR, "..", "results", "content_scores.xlsx"))
length = pd.read_csv(os.path.join(SCRIPT_DIR, "..", "results", "length_results.csv"))[
    ["ticker", "year", "sector", "size", "has_1c"]
].drop_duplicates()
df = df.merge(length, on=["ticker", "year"], how="left")

CATS = [
    "frameworks",
    "specific_controls",
    "named_individuals",
    "quantitative_data",
    "product_names",
    "technical_details",
]

CAT_LABELS = {
    "frameworks": "Frameworks",
    "specific_controls": "Specific Controls",
    "named_individuals": "Named Individuals",
    "quantitative_data": "Quantitative Data",
    "product_names": "Product Names",
    "technical_details": "Technical Details",
}

SECTOR_ORDER = [
    "Semiconductors",
    "Healthcare",
    "Consumer Goods",
    "Retail & E-Commerce",
    "Finance",
    "Technology",
    "Cybersecurity",
]
SIZE_ORDER = ["Small", "Medium", "Large"]

# style
sns.set_theme(style="whitegrid", font_scale=1.15)
BLUE = "#2E75B6"
ORANGE = "#ED7D31"
GRAY = "#7F7F7F"
GREEN = "#70AD47"

np.random.seed(42)


# Fig: Mean specificity score by year with jittered firm dots
yearly = df.groupby("year")["content_score"].agg(["mean", "std"]).reset_index()

fig, ax = plt.subplots(figsize=(9, 5))

# Jittered individual firm dots
for year in sorted(df["year"].unique()):
    vals = df[df["year"] == year]["content_score"].values
    jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(
        np.full(len(vals), year) + jitter, vals, color=BLUE, alpha=0.25, s=22, zorder=2
    )

# SD band
ax.fill_between(
    yearly["year"],
    (yearly["mean"] - yearly["std"]).clip(lower=0),
    (yearly["mean"] + yearly["std"]).clip(upper=1),
    color=BLUE,
    alpha=0.15,
    label="±1 SD",
)

# Mean line
ax.plot(
    yearly["year"],
    yearly["mean"],
    marker="o",
    color=BLUE,
    lw=2.5,
    zorder=3,
    label="Mean score",
)

# Annotate mean values
for _, row in yearly.iterrows():
    ax.annotate(
        f"{row['mean']:.3f}",
        xy=(row["year"], row["mean"]),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=10,
    )

ax.set_xticks([2022, 2023, 2024, 2025])
ax.set_xlabel("Fiscal Year")
ax.set_ylabel("Content Score (0–1)")
ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(out("content_by_year.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved content_by_year.png")


# Fig: Category adoption: with vs without Item 1C
ic_cats = (df.groupby("has_1c")[CATS].mean() * 100).T
ic_cats.columns = ["Without Item 1C", "With Item 1C"]
ic_cats.index = [CAT_LABELS[c] for c in ic_cats.index]

x = np.arange(len(ic_cats))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
bars1 = ax.bar(
    x - width / 2,
    ic_cats["Without Item 1C"],
    width,
    color=GRAY,
    alpha=0.85,
    edgecolor="white",
    label="Without Item 1C",
)
bars2 = ax.bar(
    x + width / 2,
    ic_cats["With Item 1C"],
    width,
    color=BLUE,
    alpha=0.85,
    edgecolor="white",
    label="With Item 1C",
)

# Annotate bar values
for bar in bars1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{bar.get_height():.0f}%",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
for bar in bars2:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{bar.get_height():.0f}%",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

ax.set_xticks(x)
ax.set_xticklabels(ic_cats.index, rotation=15, ha="right")
ax.set_ylabel("Share of Filings with Category Present (%)")
ax.set_ylim(0, 115)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(out("content_categories_by_1c.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved content_categories_by_1c.png")


# Fig: Mean specificity score by sector (bar + error bars)
sec = (
    df.groupby("sector")["content_score"]
    .agg(["mean", "std", "count"])
    .reindex(SECTOR_ORDER)
)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(
    sec.index,
    sec["mean"],
    xerr=sec["std"],
    color=BLUE,
    alpha=0.82,
    edgecolor="white",
    height=0.55,
    error_kw=dict(ecolor=GRAY, capsize=4, lw=1.3),
)

# Annotate mean values
for i, (idx, row) in enumerate(sec.iterrows()):
    ax.text(
        row["mean"] + row["std"] + 0.015,
        i,
        f"{row['mean']:.3f}",
        va="center",
        fontsize=9.5,
    )

ax.set_xlabel("Mean Content Score (0–1)")
ax.set_xlim(0, 1.0)
ax.axvline(
    df["content_score"].mean(),
    color=ORANGE,
    ls="--",
    lw=1.5,
    label=f"Overall mean = {df['content_score'].mean():.3f}",
)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(out("content_by_sector.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved content_by_sector.png")


# Fig: Heatmap: all firms × all years, clustered by firm size
pivot = df.pivot_table(index="ticker", columns="year", values="content_score")

size_map = df[["ticker", "size"]].drop_duplicates().set_index("ticker")["size"]
pivot["_size_order"] = pivot.index.map(size_map).map(
    {s: i for i, s in enumerate(SIZE_ORDER)}
)
year_cols = [c for c in pivot.columns if c != "_size_order"]
pivot["_mean"] = pivot[year_cols].mean(axis=1)
pivot = pivot.sort_values(by=["_size_order", "_mean"], ascending=[True, False])
pivot = pivot.drop(columns=["_size_order", "_mean"])

size_labels = size_map.reindex(pivot.index)
group_sizes = [
    int((size_labels == s).sum()) for s in SIZE_ORDER if s in size_labels.values
]
active_sizes = [s for s in SIZE_ORDER if s in size_labels.values]

fig, ax = plt.subplots(figsize=(7, 13))
sns.heatmap(
    pivot,
    ax=ax,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    linewidths=0.4,
    linecolor="#e0e0e0",
    cbar_kws={"label": "Content Score", "shrink": 0.6},
    vmin=0,
    vmax=1,
    annot_kws={"size": 8},
)

# Build per-cell has_1c lookup: (ticker, year) -> bool
has_1c_cell = df.set_index(["ticker", "year"])["has_1c"].to_dict()

# Draw orange border on individual cells WITHOUT Item 1C
for row_idx, ticker in enumerate(pivot.index):
    for col_idx, year in enumerate(pivot.columns):
        if not has_1c_cell.get((ticker, year), True):
            rect = plt.Rectangle(
                (col_idx, row_idx),
                1,
                1,
                fill=False,
                edgecolor="#C0504D",
                lw=2.0,
                clip_on=True,
            )
            ax.add_patch(rect)

# Legend for Item 1C border
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="none", edgecolor="#C0504D", lw=2.0, label="No Item 1C")
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.8)

# Divider lines between size groups
cumulative_boundaries = np.cumsum(group_sizes[:-1])
for boundary in cumulative_boundaries:
    ax.axhline(boundary, color="black", lw=1.5, ls="--")

# Dark gray side bars with rotated size labels
ax_pos = ax.get_position()
group_start = 0
for sz, count in zip(active_sizes, group_sizes):
    y_bottom = group_start / len(pivot)
    y_top = (group_start + count) / len(pivot)
    y_mid = (y_bottom + y_top) / 2

    gap = 0.004  # small gap at top and bottom of each bar
    fig_y0 = ax_pos.y0 + (1 - y_top) * ax_pos.height + gap
    fig_height = (count / len(pivot)) * ax_pos.height - 2 * gap
    fig_x0 = ax_pos.x1 + 0.005

    label_ax = fig.add_axes([fig_x0, fig_y0, 0.025, fig_height])
    label_ax.set_facecolor("#4A4A4A")
    label_ax.set_xticks([])
    label_ax.set_yticks([])
    for spine in label_ax.spines.values():
        spine.set_visible(False)
    label_ax.text(
        0.5,
        0.5,
        sz,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
        rotation=90,
        transform=label_ax.transAxes,
    )
    group_start += count

ax.set_xlabel("Fiscal Year")
ax.set_ylabel("")
plt.savefig(out("content_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved content_heatmap.png")

print("\nAll content score outputs saved successfully.")
