"""
Streamlit dashboard: Customer Segmentation & Churn Pattern Analytics

Run:
    streamlit run app.py

Place European_Bank.csv in the same folder, or upload it in the sidebar.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="European Bank | Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

REQUIRED_COLUMNS = [
    "Year", "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
    "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary", "Exited"
]

@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    data = pd.read_csv(BytesIO(file_bytes))
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if data.empty:
        raise ValueError("The CSV contains no records.")

    numeric = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
    ]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if data[numeric].isna().any().any():
        raise ValueError("One or more required numeric fields contain missing or invalid values.")
    if not set(data["Exited"].unique()).issubset({0, 1}):
        raise ValueError("Exited must contain only 0 and 1.")
    if not set(data["HasCrCard"].unique()).issubset({0, 1}):
        raise ValueError("HasCrCard must contain only 0 and 1.")
    if not set(data["IsActiveMember"].unique()).issubset({0, 1}):
        raise ValueError("IsActiveMember must contain only 0 and 1.")
    return data

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["AgeGroup"] = pd.cut(
        df["Age"], [-np.inf, 29, 45, 59, np.inf],
        labels=["Under 30", "30–45", "46–59", "60+"]
    )
    df["CreditScoreBand"] = pd.cut(
        df["CreditScore"], [-np.inf, 579, 669, np.inf],
        labels=["Low (≤579)", "Medium (580–669)", "High (≥670)"]
    )
    df["TenureGroup"] = pd.cut(
        df["Tenure"], [-np.inf, 2, 6, np.inf],
        labels=["New (0–2 years)", "Mid-term (3–6 years)", "Long-term (7+ years)"]
    )
    nonzero = df.loc[df["Balance"] > 0, "Balance"]
    median_nonzero = float(nonzero.median()) if not nonzero.empty else 0.0
    df["BalanceSegment"] = np.select(
        [
            df["Balance"].eq(0),
            df["Balance"].gt(0) & df["Balance"].le(median_nonzero),
            df["Balance"].gt(median_nonzero)
        ],
        ["Zero balance", "Lower non-zero balance", "Higher non-zero balance"],
        default="Unclassified"
    )
    df["HighValueCustomer"] = df["Balance"] >= df["Balance"].quantile(0.75)
    df["CustomerStatus"] = np.where(df["Exited"].eq(1), "Churned", "Retained")
    return df

def make_summary(data: pd.DataFrame, segment: str) -> pd.DataFrame:
    summary = (
        data.groupby(segment, observed=False, dropna=False)
            .agg(Customers=("Exited", "size"), Churned=("Exited", "sum"))
            .reset_index()
            .rename(columns={segment: "Segment"})
    )
    summary["Retained"] = summary["Customers"] - summary["Churned"]
    summary["Churn rate (%)"] = (
        summary["Churned"] / summary["Customers"].replace(0, np.nan) * 100
    ).fillna(0).round(2)
    churn_total = int(data["Exited"].sum())
    summary["Churn contribution (%)"] = (
        summary["Churned"] / churn_total * 100 if churn_total else 0
    )
    summary["Churn contribution (%)"] = summary["Churn contribution (%)"].round(2)
    return summary

def bar_chart(summary: pd.DataFrame, title: str):
    plot_df = summary.copy()
    plot_df["Segment"] = plot_df["Segment"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(plot_df["Segment"], plot_df["Churn rate (%)"])
    ax.set_title(title)
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, max(5, float(plot_df["Churn rate (%)"].max()) * 1.2))
    for bar, value in zip(bars, plot_df["Churn rate (%)"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig

st.title("Customer Segmentation & Churn Pattern Analytics")
st.caption("European banking dataset | Descriptive analysis and interactive exploration")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload European_Bank.csv", type=["csv"])
    if uploaded is not None:
        source_bytes = uploaded.getvalue()
        source_label = uploaded.name
    else:
        default_path = Path(__file__).resolve().parent / "European_Bank.csv"
        if not default_path.exists():
            st.info("Upload European_Bank.csv to begin.")
            st.stop()
        source_bytes = default_path.read_bytes()
        source_label = default_path.name

try:
    raw = load_csv(source_bytes)
    df = prepare_data(raw)
except Exception as exc:
    st.error(f"Unable to load or validate the dataset: {exc}")
    st.stop()

with st.sidebar:
    st.divider()
    st.header("Filters")
    countries = sorted(df["Geography"].dropna().unique().tolist())
    selected_countries = st.multiselect("Country", countries, default=countries)
    age_options = [str(x) for x in df["AgeGroup"].dropna().unique()]
    selected_ages = st.multiselect("Age group", age_options, default=age_options)
    tenure_options = [str(x) for x in df["TenureGroup"].dropna().unique()]
    selected_tenure = st.multiselect("Tenure group", tenure_options, default=tenure_options)
    genders = sorted(df["Gender"].dropna().unique().tolist())
    selected_genders = st.multiselect("Gender", genders, default=genders)
    activity = st.selectbox("Activity status", ["All", "Active", "Inactive"])
    high_balance = st.selectbox("High-balance proxy", ["All", "Top 25% balance", "Other customers"])

filtered = df[
    df["Geography"].isin(selected_countries)
    & df["AgeGroup"].astype(str).isin(selected_ages)
    & df["TenureGroup"].astype(str).isin(selected_tenure)
    & df["Gender"].isin(selected_genders)
].copy()

if activity == "Active":
    filtered = filtered[filtered["IsActiveMember"] == 1]
elif activity == "Inactive":
    filtered = filtered[filtered["IsActiveMember"] == 0]

if high_balance == "Top 25% balance":
    filtered = filtered[filtered["HighValueCustomer"]]
elif high_balance == "Other customers":
    filtered = filtered[~filtered["HighValueCustomer"]]

st.caption(f"Source: {source_label} · Showing {len(filtered):,} of {len(df):,} customer records")
if filtered.empty:
    st.warning("No customers match the selected filters. Change one or more filters.")
    st.stop()

total = len(filtered)
churned = int(filtered["Exited"].sum())
retained = total - churned
churn_rate = churned / total * 100
retention_rate = retained / total * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total customers", f"{total:,}")
c2.metric("Churned customers", f"{churned:,}")
c3.metric("Retained customers", f"{retained:,}")
c4.metric("Churn rate", f"{churn_rate:.2f}%")
st.progress(min(max(churn_rate / 100, 0.0), 1.0), text=f"Retention rate: {retention_rate:.2f}%")

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Overall customer status")
    status = pd.DataFrame({
        "Status": ["Retained", "Churned"],
        "Customers": [retained, churned]
    }).set_index("Status")
    st.bar_chart(status)
with right:
    st.subheader("Customer profile")
    profile = pd.DataFrame({
        "Measure": ["Average age", "Average credit score", "Average tenure",
                    "Average balance", "Average products"],
        "Value": [
            filtered["Age"].mean(),
            filtered["CreditScore"].mean(),
            filtered["Tenure"].mean(),
            filtered["Balance"].mean(),
            filtered["NumOfProducts"].mean()
        ]
    })
    st.dataframe(profile.style.format({"Value": "{:,.2f}"}), use_container_width=True, hide_index=True)

st.divider()
st.header("Segment analysis")
dimensions = [
    ("Geography", "Country"),
    ("AgeGroup", "Age group"),
    ("Gender", "Gender"),
    ("CreditScoreBand", "Credit score band"),
    ("TenureGroup", "Tenure group"),
    ("BalanceSegment", "Balance segment"),
    ("NumOfProducts", "Number of products"),
    ("IsActiveMember", "Activity status"),
    ("HasCrCard", "Credit-card ownership"),
    ("HighValueCustomer", "High-balance proxy")
]
labels = {key: label for key, label in dimensions}
chosen_dimension = st.selectbox(
    "Choose a segment dimension",
    options=[x[0] for x in dimensions],
    format_func=lambda x: labels[x]
)
summary = make_summary(filtered, chosen_dimension)
chart = bar_chart(summary, f"Churn rate by {labels[chosen_dimension]}")
st.pyplot(chart, use_container_width=True)
plt.close(chart)
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("Geographic risk index")
st.caption("100 represents the overall churn-rate benchmark for the currently filtered population.")
geo = make_summary(filtered, "Geography")
geo["Geographic risk index"] = (
    geo["Churn rate (%)"] / churn_rate * 100 if churn_rate else 0
).round(2)
st.dataframe(geo, use_container_width=True, hide_index=True)

st.divider()
st.header("Download aggregate summaries")
st.caption("Exports contain aggregate segment metrics only; customer identifiers and surnames are not included.")
csv_bytes = summary.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download current segment summary (CSV)",
    data=csv_bytes,
    file_name=f"{chosen_dimension.lower()}_churn_summary.csv",
    mime="text/csv"
)

st.info(
    "Interpretation note: results are descriptive patterns in the supplied dataset. "
    "They do not establish causes of churn or predict individual customer exits. "
    "The top-25% balance group is a high-balance proxy, not verified profitability "
    "or customer lifetime value."
)
