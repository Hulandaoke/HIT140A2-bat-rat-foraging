#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf


def generate_figures_and_stats(data_dir: Path, fig_dir: Path) -> None:
    # Set up plotting style
    sns.set_theme(style="whitegrid")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load the two datasets
    df1 = pd.read_csv(data_dir / "dataset1.csv")
    df2 = pd.read_csv(data_dir / "dataset2.csv")

    # Make sure numeric columns are properly formatted
    df1["risk"] = pd.to_numeric(df1["risk"], errors="coerce")
    df1["season"] = pd.to_numeric(df1["season"], errors="coerce")
    df1["seconds_after_rat_arrival"] = pd.to_numeric(df1["seconds_after_rat_arrival"], errors="coerce")
    df1["hours_after_sunset"] = pd.to_numeric(df1["hours_after_sunset"], errors="coerce")

    df2["bat_landing_number"] = pd.to_numeric(df2["bat_landing_number"], errors="coerce")
    df2["rat_minutes"] = pd.to_numeric(df2["rat_minutes"], errors="coerce")
    df2["hours_after_sunset"] = pd.to_numeric(df2["hours_after_sunset"], errors="coerce")

    # Create first figure: Risk rate by season
    risk_by_season = (
        df1.dropna(subset=["season", "risk"]).groupby("season", as_index=False)["risk"].mean()
    )
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=risk_by_season, x="season", y="risk", color="#5B8FF9")
    ax.set_xlabel("Season")
    ax.set_ylabel("Risk rate")
    ax.set_title("Risk rate by season")
    ax.set_ylim(0, 1)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9)
    plt.tight_layout()
    out1 = fig_dir / "risk_rate_by_season.png"
    plt.savefig(out1, dpi=150)
    plt.close()

    # Figure 2: Bat landings vs rat minutes
    plt.figure(figsize=(7, 5))
    ax = sns.regplot(
        data=df2.dropna(subset=["bat_landing_number", "rat_minutes"]),
        x="rat_minutes", y="bat_landing_number",
        scatter_kws={"alpha": 0.6, "s": 35}, line_kws={"color": "#EF6C00"}
    )
    ax.set_xlabel("Rat minutes (per 30-min bin)")
    ax.set_ylabel("Bat landings")
    ax.set_title("Bat landings vs rat minutes")
    x = df2["rat_minutes"]
    y = df2["bat_landing_number"]
    mask = x.notna() & y.notna()
    r, p = stats.pearsonr(x[mask], y[mask])
    ax.text(0.02, 0.95, f"r = {r:.3f}\np = {p:.3g}", transform=ax.transAxes,
            va="top", ha="left", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    plt.tight_layout()
    out2 = fig_dir / "bat_landings_vs_rat_minutes.png"
    plt.savefig(out2, dpi=150)
    plt.close()

    # Stats: Chi-square (season x risk)
    ct = pd.crosstab(df1["season"], df1["risk"])
    chi2, pval, dof, _ = stats.chi2_contingency(ct)
    print("Chi-square test (season x risk)")
    print(f"chi2 = {chi2:.3f}, df = {dof}, p = {pval:.5f}")

    # Stats: Logistic regression
    model_df = df1[["risk", "seconds_after_rat_arrival", "hours_after_sunset", "season"]].dropna().copy()
    model_df["risk"] = model_df["risk"].astype(int)
    logit = smf.logit(
        "risk ~ seconds_after_rat_arrival + hours_after_sunset + C(season)",
        data=model_df
    ).fit(disp=False)
    params = logit.params
    conf = logit.conf_int()
    or_ = np.exp(params)
    ci_low = np.exp(conf[0])
    ci_high = np.exp(conf[1])
    out = pd.DataFrame({
        "term": or_.index,
        "OR": or_.values.round(4),
        "CI_lower": ci_low.values.round(4),
        "CI_upper": ci_high.values.round(4),
        "p_value": logit.pvalues.values.round(4),
    })
    print("\nLogistic regression: risk ~ seconds_after_rat_arrival + hours_after_sunset + C(season)")
    print(out.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Generate figures and stats for HIT140 A2 Investigation A")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to directory containing dataset1.csv and dataset2.csv",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for generated figures",
    )
    args = parser.parse_args()

    generate_figures_and_stats(args.data_dir, args.fig_dir)


if __name__ == "__main__":
    main()

