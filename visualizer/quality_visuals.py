import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "visuals", "quality")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


# load data
df = pd.read_csv(os.path.join(SCRIPT_DIR, "..", "results", "quality_results.csv"))
df_scored = df.dropna(subset=["quality_score"])

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

# Fig: Mean Quality Score by Year with Jittered Firm Dots
yearly = df_scored.groupby("year")["quality_score"].agg(["mean", "std"]).reset_index()

fig, ax = plt.subplots(figsize=(9, 5))

for year in sorted(df_scored["year"].unique()):
    vals = df_scored[df_scored["year"] == year]["quality_score"].values
    jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(
        np.full(len(vals), year) + jitter, vals, color=BLUE, alpha=0.25, s=22, zorder=2
    )

ax.fill_between(
    yearly["year"],
    (yearly["mean"] - yearly["std"]).clip(lower=0),
    (yearly["mean"] + yearly["std"]).clip(upper=100),
    color=BLUE,
    alpha=0.15,
    label="±1 SD",
)

ax.plot(
    yearly["year"],
    yearly["mean"],
    marker="o",
    color=BLUE,
    lw=2.5,
    zorder=3,
    label="Mean score",
)

for _, row in yearly.iterrows():
    ax.annotate(
        f"{row['mean']:.1f}",
        xy=(row["year"], row["mean"]),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=10,
    )

ax.set_xticks([2023, 2024, 2025])
ax.set_xlabel("Fiscal Year")
ax.set_ylabel("Quality Score (0–100)")
ax.set_ylim(-5, 110)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(out("quality_by_year.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved quality_by_year.png")

# Fig: Mean Quality Score by Sector (Horizontal Bar + SD)
sec = (
    df_scored.groupby("sector")["quality_score"]
    .agg(["mean", "std"])
    .reindex(SECTOR_ORDER)
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(
    sec.index,
    sec["mean"],
    xerr=sec["std"],
    color=BLUE,
    alpha=0.82,
    edgecolor="white",
    height=0.55,
    error_kw=dict(ecolor=GRAY, capsize=4, lw=1.3),
)

for i, (idx, row) in enumerate(sec.iterrows()):
    ax.text(
        row["mean"] + row["std"] + 1.5,
        i,
        f"{row['mean']:.1f}",
        va="center",
        fontsize=9.5,
    )

ax.set_xlabel("Mean Quality Score (0–100)")
ax.set_xlim(0, 105)
ax.axvline(
    df_scored["quality_score"].mean(),
    color=ORANGE,
    ls="--",
    lw=1.5,
    label=f"Overall mean = {df_scored['quality_score'].mean():.1f}",
)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(out("quality_by_sector.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved quality_by_sector.png")

# Fig: Quality Score: Item 1C Adopters vs Non-Adopters
q_1c = df_scored.groupby("has_1c")["quality_score"].agg(["mean", "std"]).reset_index()
q_1c["label"] = q_1c["has_1c"].map({True: "With Item 1C", False: "Without Item 1C"})

x = np.arange(len(q_1c))
width = 0.4
colors = [GRAY, BLUE]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    x,
    q_1c["mean"],
    width,
    yerr=q_1c["std"],
    color=colors,
    alpha=0.85,
    edgecolor="white",
    error_kw=dict(ecolor=GRAY, capsize=5, lw=1.3),
)

for bar, val in zip(bars, q_1c["mean"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 3,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax.set_xticks(x)
ax.set_xticklabels(q_1c["label"])
ax.set_ylabel("Mean Quality Score (0–100)")
ax.set_ylim(0, 85)
plt.tight_layout()
plt.savefig(out("quality_by_1c.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved quality_by_1c.png")

# Fig: Quality Score Boxplot by Firm Size
size_means = df_scored.groupby("size")["quality_score"].mean()

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=df_scored,
    x="size",
    y="quality_score",
    order=SIZE_ORDER,
    color=BLUE,
    ax=ax,
    flierprops=dict(marker=".", markersize=5, alpha=0.5),
)

for i, sz in enumerate(SIZE_ORDER):
    ax.text(
        i,
        size_means[sz] + 3,
        f"Mean\n{size_means[sz]:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_xlabel("Firm Size")
ax.set_ylabel("Quality Score (0–100)")
ax.set_ylim(-5, 120)
plt.tight_layout()
plt.savefig(out("quality_by_size.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved quality_by_size.png")

# Fig: Heatmap: All Firms × All Years, clustered by firm size
pivot = df_scored.pivot_table(index="ticker", columns="year", values="quality_score")

# Per-cell has_1c lookup
has_1c_cell = df_scored.set_index(["ticker", "year"])["has_1c"].to_dict()

# Sort by size group, then by mean score descending within each group
size_map = df_scored[["ticker", "size"]].drop_duplicates().set_index("ticker")["size"]
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
    fmt=".0f",
    cmap="Blues",
    linewidths=0.4,
    linecolor="#e0e0e0",
    cbar_kws={"label": "Quality Score (0–100)", "shrink": 0.6},
    vmin=0,
    vmax=100,
    annot_kws={"size": 8},
)

# Orange-free border on cells WITHOUT Item 1C
from matplotlib.patches import Patch

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
legend_elements = [
    Patch(facecolor="none", edgecolor="#C0504D", lw=2.0, label="No Item 1C")
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.8)

# Dashed divider lines between size groups
cumulative_boundaries = np.cumsum(group_sizes[:-1])
for boundary in cumulative_boundaries:
    ax.axhline(boundary, color="black", lw=1.5, ls="--")

# Dark gray side bars with rotated size labels and small gaps
ax_pos = ax.get_position()
group_start = 0
for sz, count in zip(active_sizes, group_sizes):
    y_top = (group_start + count) / len(pivot)

    gap = 0.004
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
plt.savefig(out("quality_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved quality_heatmap.png")

# Fig: Quality scores by presence of 1C

# aggregate data
# Calculate mean and standard deviation grouped by Item 1C presence
q_1c = df_scored.groupby("has_1c")["quality_score"].agg(["mean", "std"]).reset_index()
q_1c["label"] = q_1c["has_1c"].map({True: "With Item 1C", False: "Without Item 1C"})

# Sort to ensure "Without Item 1C" is first for better visual comparison
q_1c = q_1c.sort_values("has_1c")

# visualization
sns.set_theme(style="whitegrid", font_scale=1.15)
BLUE = "#2E75B6"
GRAY = "#7F7F7F"

fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(q_1c))
width = 0.5
colors = [GRAY, BLUE]

bars = ax.bar(
    x,
    q_1c["mean"],
    width,
    yerr=q_1c["std"],
    color=colors,
    alpha=0.85,
    edgecolor="white",
    error_kw=dict(ecolor=GRAY, capsize=5, lw=1.5),
)

# Annotate bars with the exact mean values
for bar, val in zip(bars, q_1c["mean"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    # Formatting
ax.set_xticks(x)
ax.set_xticklabels(q_1c["label"])
ax.set_ylabel("Mean Quality Score (0–100)")
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("quality_by_1c.png", dpi=150, bbox_inches="tight")
print("Visualization saved as quality_by_1c.png")

print("\nAll quality score outputs saved successfully.")
