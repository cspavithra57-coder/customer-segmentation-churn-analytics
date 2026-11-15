"""
Customer Segmentation & Churn Pattern Analytics
Batch analysis script for European_Bank.csv

Run:
    python analysis.py --data European_Bank.csv --output outputs
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REQUIRED_COLUMNS = [
    "Year", "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
    "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary", "Exited"
]

def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")
    if df.empty:
        raise ValueError("The CSV contains no customer records.")

    numeric = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[numeric].isna().any().any():
        bad = df[numeric].columns[df[numeric].isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric values found in: {', '.join(bad)}")

    if not set(df["Exited"].unique()).issubset({0, 1}):
        raise ValueError("Exited must contain only 0 (retained) and 1 (churned).")
    if not set(df["HasCrCard"].unique()).issubset({0, 1}):
        raise ValueError("HasCrCard must contain only 0 and 1.")
    if not set(df["IsActiveMember"].unique()).issubset({0, 1}):
        raise ValueError("IsActiveMember must contain only 0 and 1.")
    if (df["Balance"] < 0).any():
        raise ValueError("Balance contains negative values; review the source data.")

    return df

def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[-np.inf, 29, 45, 59, np.inf],
        labels=["Under 30", "30–45", "46–59", "60+"]
    )
    data["CreditScoreBand"] = pd.cut(
        data["CreditScore"],
        bins=[-np.inf, 579, 669, np.inf],
        labels=["Low (≤579)", "Medium (580–669)", "High (≥670)"]
    )
    data["TenureGroup"] = pd.cut(
        data["Tenure"],
        bins=[-np.inf, 2, 6, np.inf],
        labels=["New (0–2 years)", "Mid-term (3–6 years)", "Long-term (7+ years)"]
    )

    nonzero = data.loc[data["Balance"] > 0, "Balance"]
    median_nonzero = float(nonzero.median()) if not nonzero.empty else 0.0
    data["BalanceSegment"] = np.select(
        [
            data["Balance"].eq(0),
            data["Balance"].gt(0) & data["Balance"].le(median_nonzero),
            data["Balance"].gt(median_nonzero)
        ],
        ["Zero balance", "Lower non-zero balance", "Higher non-zero balance"],
        default="Unclassified"
    )

    high_balance_threshold = float(data["Balance"].quantile(0.75))
    data["HighValueCustomer"] = data["Balance"].ge(high_balance_threshold)
    data["CustomerStatus"] = np.where(data["Exited"].eq(1), "Churned", "Retained")
    return data

def segment_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    result = (
        df.groupby(column, observed=False, dropna=False)
          .agg(Customers=("Exited", "size"), Churned=("Exited", "sum"))
          .reset_index()
          .rename(columns={column: "Segment"})
    )
    result["Retained"] = result["Customers"] - result["Churned"]
    result["ChurnRatePct"] = np.where(
        result["Customers"] > 0, result["Churned"] / result["Customers"] * 100, 0
    )
    total_churned = int(df["Exited"].sum())
    result["ChurnContributionPct"] = (
        result["Churned"] / total_churned * 100 if total_churned else 0
    )
    result["ChurnRatePct"] = result["ChurnRatePct"].round(2)
    result["ChurnContributionPct"] = result["ChurnContributionPct"].round(2)
    return result

def save_bar_chart(summary: pd.DataFrame, title: str, filename: Path) -> None:
    plot_data = summary.copy()
    plot_data["Segment"] = plot_data["Segment"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(plot_data["Segment"], plot_data["ChurnRatePct"])
    ax.set_title(title)
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("Customer segment")
    ax.set_ylim(0, max(5, float(plot_data["ChurnRatePct"].max()) * 1.2))
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, plot_data["ChurnRatePct"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.2f}%",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)

def run_analysis(data_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    charts = out / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    raw = load_and_validate(data_path)
    df = add_segments(raw)

    total = len(df)
    churned = int(df["Exited"].sum())
    retained = total - churned
    overview = pd.DataFrame([{
        "Rows": total,
        "Columns": raw.shape[1],
        "MissingCells": int(raw.isna().sum().sum()),
        "DuplicateRows": int(raw.duplicated().sum()),
        "Churned": churned,
        "Retained": retained,
        "OverallChurnRatePct": round(churned / total * 100, 2)
    }])
    overview.to_csv(out / "dataset_overview.csv", index=False)
    raw.describe().T.to_csv(out / "descriptive_statistics.csv")

    dimensions = [
        ("Geography", "02_country.png"),
        ("AgeGroup", "03_age.png"),
        ("Gender", "04_gender.png"),
        ("CreditScoreBand", "05_credit.png"),
        ("TenureGroup", "06_tenure.png"),
        ("BalanceSegment", "07_balance.png"),
        ("NumOfProducts", "08_products.png"),
        ("IsActiveMember", "09_active.png"),
        ("HasCrCard", "10_card.png"),
        ("HighValueCustomer", "11_highvalue.png"),
    ]

    status = pd.DataFrame({
        "CustomerStatus": ["Retained", "Churned"],
        "Customers": [retained, churned],
        "Percentage": [round(retained/total*100, 2), round(churned/total*100, 2)]
    })
    status.to_csv(out / "overall_churn.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(status["CustomerStatus"], status["Percentage"])
    ax.set_title("Retained and churned customers")
    ax.set_ylabel("Customers (%)")
    ax.set_ylim(0, 100)
    for i, v in enumerate(status["Percentage"]):
        ax.text(i, v + 1, f"{v:.2f}%", ha="center")
    fig.tight_layout()
    fig.savefig(charts / "01_overall_churn.png", dpi=160)
    plt.close(fig)

    for col, chart_name in dimensions:
        summary = segment_summary(df, col)
        safe_name = col.lower()
        summary.to_csv(out / f"segment_{safe_name}.csv", index=False)
        save_bar_chart(summary, f"Churn rate by {col}", charts / chart_name)

    # Geographic risk index: 100 is the overall churn-rate benchmark.
    geo = segment_summary(df, "Geography")
    overall_rate = churned / total * 100
    geo["GeographicRiskIndex"] = (
        geo["ChurnRatePct"] / overall_rate * 100 if overall_rate else 0
    ).round(2)
    geo.to_csv(out / "geographic_risk_index.csv", index=False)

    # Aggregate cross-segment tables; no direct customer identifiers are exported.
    for left, right in [
        ("Geography", "AgeGroup"),
        ("Geography", "IsActiveMember"),
        ("AgeGroup", "TenureGroup"),
        ("NumOfProducts", "IsActiveMember")
    ]:
        cross = (
            df.groupby([left, right], observed=False)
              .agg(Customers=("Exited", "size"), Churned=("Exited", "sum"))
              .reset_index()
        )
        cross["Retained"] = cross["Customers"] - cross["Churned"]
        cross["ChurnRatePct"] = (
            cross["Churned"] / cross["Customers"] * 100
        ).round(2)
        cross.to_csv(out / f"cross_{left.lower()}_{right.lower()}.csv", index=False)

    print(f"Analysis complete. Results saved to: {out.resolve()}")
    print(f"Customers: {total:,} | Churned: {churned:,} | "
          f"Retained: {retained:,} | Churn rate: {churned/total*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="European bank customer churn analysis")
    parser.add_argument("--data", default="European_Bank.csv", help="Path to input CSV")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()
    run_analysis(args.data, args.output)
