"""Exploratory Data Analysis for the UCI Student Performance (Portuguese) dataset.

Run:  python eda.py
Writes figures to outputs/figures/ and a text summary to outputs/eda_summary.txt
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from common import (CATEGORICAL_FEATURES, CLASS_COLORS, CLASS_NAMES, CLASS_RULES, FEATURE_INFO, FIG_DIR, GRID,
                    INK, INK_2, NUMERIC_FEATURES, OUTPUTS_DIR, SERIES, SURFACE, TARGET, apply_chart_style,
                    grade_to_class, load_data)

DIVERGING = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])


def save(fig, name):
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"  saved outputs/figures/{name}")


def text_summary(df: pd.DataFrame) -> str:
    cls = df[TARGET].map(grade_to_class).map(lambda i: CLASS_NAMES[i])
    counts = cls.value_counts().reindex(CLASS_NAMES)
    corr = df[NUMERIC_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    lines = [
        "UCI Student Performance - Portuguese course",
        "=" * 60,
        f"Records: {len(df)}   Attributes: {df.shape[1]} ({df.shape[1] - 1} inputs + target G3)",
        f"Missing values: {int(df.isna().sum().sum())}   Duplicate rows: {int(df.duplicated().sum())}",
        f"Categorical inputs: {len(CATEGORICAL_FEATURES)}   Numeric / ordinal inputs: {len(NUMERIC_FEATURES)}",
        "",
        "Final grade (G3) statistics",
        df[TARGET].describe().round(2).to_string(),
        f"Students with G3 = 0: {int((df[TARGET] == 0).sum())}",
        "",
        "Performance category distribution",
    ]
    lines += [f"  {c:<7} {CLASS_RULES[c]:<28} {counts[c]:>4}  ({counts[c] / len(df):.1%})" for c in CLASS_NAMES]
    lines += ["", "Correlation of numeric inputs with G3 (Pearson)", corr.round(3).to_string(), "",
              "Numeric feature summary", df[NUMERIC_FEATURES].describe().T.round(2).to_string(), "",
              "Mean G3 by categorical feature"]
    for col in CATEGORICAL_FEATURES:
        grp = df.groupby(col)[TARGET].agg(["count", "mean"]).round(2)
        lines.append(f"  {col}: " + ", ".join(f"{k}={v['mean']} (n={int(v['count'])})" for k, v in grp.iterrows()))
    return "\n".join(lines)


def plot_grade_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    counts = df[TARGET].value_counts().reindex(range(21), fill_value=0)
    colors = [CLASS_COLORS[CLASS_NAMES[grade_to_class(g)]] for g in counts.index]
    ax.bar(counts.index, counts.values, color=colors, width=0.8, edgecolor=SURFACE, linewidth=1.5)
    ax.set(title="Final grade (G3) distribution", xlabel="Final grade G3 (0-20)", ylabel="Students")
    ax.set_xticks(range(0, 21, 2))
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c]) for c in CLASS_NAMES],
              labels=[f"{c}: {CLASS_RULES[c]}" for c in CLASS_NAMES], loc="upper left")

    ax = axes[1]
    cls = df[TARGET].map(grade_to_class).value_counts().sort_index()
    bars = ax.bar(CLASS_NAMES, cls.values, color=[CLASS_COLORS[c] for c in CLASS_NAMES], width=0.6)
    ax.bar_label(bars, labels=[f"{v}\n{v / len(df):.0%}" for v in cls.values], padding=3, color=INK)
    ax.set(title="Performance category balance", ylabel="Students", ylim=(0, cls.max() * 1.2))
    ax.grid(axis="x", visible=False)
    save(fig, "01_grade_distribution.png")


def plot_correlation(df):
    cols = NUMERIC_FEATURES + [TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else INK)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-.5, len(cols)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson correlation")
    ax.set_title("Correlation between numeric features and the final grade")
    save(fig, "02_correlation_heatmap.png")


def plot_g3_correlation_bar(df):
    corr = df[NUMERIC_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(corr.index, corr.values, color=[SERIES[0] if v >= 0 else SERIES[1] for v in corr.values], height=0.65)
    for y, v in enumerate(corr.values):
        ax.text(v + (0.015 if v >= 0 else -0.015), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set(title="Which numeric features move with the final grade?", xlabel="Pearson correlation with G3",
           xlim=(corr.min() - 0.15, 1.08))
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SERIES[0]), plt.Rectangle((0, 0), 1, 1, color=SERIES[1])],
              labels=["Positive", "Negative"], loc="lower right")
    save(fig, "03_correlation_with_g3.png")


def plot_period_grades(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    rng = np.random.default_rng(0)
    cls = df[TARGET].map(grade_to_class)
    for ax, col in zip(axes, ["G1", "G2"]):
        for i, name in enumerate(CLASS_NAMES):
            m = cls == i
            ax.scatter(df.loc[m, col] + rng.uniform(-.25, .25, m.sum()),
                       df.loc[m, TARGET] + rng.uniform(-.25, .25, m.sum()),
                       s=22, color=CLASS_COLORS[name], alpha=0.7, edgecolor=SURFACE, linewidth=0.5, label=name)
        ax.set(title=f"{col} vs final grade (r = {df[col].corr(df[TARGET]):.2f})",
               xlabel=FEATURE_INFO[col]["label"], xlim=(-1, 20.5))
    axes[0].set_ylabel("Final grade G3")
    axes[0].legend(title="Category", loc="upper left")
    save(fig, "04_period_grades_vs_final.png")


def plot_category_shares(df):
    """Share of Low / Medium / High students for the most informative features."""
    feats = ["failures", "higher", "school", "studytime", "Medu", "Dalc", "internet", "address"]
    cls = df[TARGET].map(grade_to_class)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.ravel(), feats):
        tab = pd.crosstab(df[col], cls, normalize="index").reindex(columns=[0, 1, 2], fill_value=0) * 100
        n = df[col].value_counts().reindex(tab.index)
        labels = [f"{FEATURE_INFO[col]['options'].get(k, k)}\n(n={n[k]})" for k in tab.index]
        left = np.zeros(len(tab))
        for i, name in enumerate(CLASS_NAMES):
            vals = tab[i].values
            ax.barh(labels, vals, left=left, color=CLASS_COLORS[name], height=0.62,
                    edgecolor=SURFACE, linewidth=2, label=name)
            for y, (v, l) in enumerate(zip(vals, left)):
                if v >= 12:
                    ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=8)
            left += vals
        ax.set_title(FEATURE_INFO[col]["label"], fontsize=10.5)
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.grid(False)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xticks([0, 50, 100], ["0%", "50%", "100%"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, title="Performance category")
    fig.suptitle("Share of performance categories within each group", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "05_category_share_by_feature.png")


def plot_absences(df):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    ax.hist(df["absences"], bins=range(0, 34, 2), color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
    ax.set(title="Absences are heavily right-skewed", xlabel="Number of absences", ylabel="Students")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    cls = df[TARGET].map(grade_to_class)
    data = [df.loc[cls == i, "absences"] for i in range(3)]
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55, medianprops={"color": INK, "linewidth": 1.5},
                    flierprops={"marker": "o", "markersize": 3, "markerfacecolor": GRID, "markeredgecolor": INK_2},
                    whiskerprops={"color": INK_2}, capprops={"color": INK_2})
    for patch, name in zip(bp["boxes"], CLASS_NAMES):
        patch.set(facecolor=CLASS_COLORS[name], edgecolor=INK_2, alpha=0.9)
    ax.set_yticks([1, 2, 3], CLASS_NAMES)
    ax.set(title="Absences by performance category", xlabel="Number of absences")
    ax.grid(axis="y", visible=False)
    save(fig, "06_absences.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_chart_style()
    df = load_data()

    summary = text_summary(df)
    (OUTPUTS_DIR / "eda_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print("\nFigures:")
    plot_grade_distribution(df)
    plot_correlation(df)
    plot_g3_correlation_bar(df)
    plot_period_grades(df)
    plot_category_shares(df)
    plot_absences(df)


if __name__ == "__main__":
    main()
