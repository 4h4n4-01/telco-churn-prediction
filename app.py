import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config must be set before any stramlit comments 
st.set_page_config(
    page_title="Telco Churn Dashboard",
    layout="wide",
)

#  Constants 
TIER_ORDER = ["Critical", "High", "Medium", "Low"]
TIER_COLORS = {
    "Critical": "#dc2626",
    "High":     "#f59e0b",
    "Medium":   "#3b82f6",
    "Low":      "#16a34a",
}

# Data
DATA_PATH = Path(__file__).parent / "data" / "processed" / "churn_scored_customers.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["risk_tier"] = pd.Categorical(df["risk_tier"], categories=TIER_ORDER, ordered=True)
    return df


df = load_data()

# Sidebar
with st.sidebar:
    st.title("Filters")

    selected_tiers = st.multiselect(
        "Risk Tier",
        options=TIER_ORDER,
        default=TIER_ORDER,
    )

    contract_options = sorted(df["Contract"].unique().tolist())
    selected_contracts = st.multiselect(
        "Contract Type",
        options=contract_options,
        default=contract_options,
    )

    prob_min, prob_max = st.slider(
        "Churn Probability Range",
        min_value=0.0, max_value=1.0,
        value=(0.0, 1.0), step=0.05,
    )

    st.divider()
    st.caption(
        "**Threshold = 0.3** and any customer above this "
        "is flagged as a predicted churner to maximise recall."
    )

filtered = df[
    df["risk_tier"].isin(selected_tiers)
    & df["Contract"].isin(selected_contracts)
    & df["churn_probability"].between(prob_min, prob_max)
]

# Header
st.title("Telco Churn Risk Dashboard")
st.caption(
    f"IBM Telco dataset · **{len(df):,} customers** scored · "
    "Model: XGBoost (F1-tuned, 44 features) · "
    "Source: `data/processed/churn_scored_customers.csv`"
)
st.divider()

# KPI cards 
k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total Customers",
    f"{len(df):,}",
)
k2.metric(
    "Predicted Churners",
    f"{int(df['predicted_churn'].sum()):,}",
    f"{df['predicted_churn'].mean():.1%} of base",
)
k3.metric(
    "Critical Risk",
    f"{int((df['risk_tier'] == 'Critical').sum()):,}",
    f"{(df['risk_tier'] == 'Critical').mean():.1%} of base",
    delta_color="inverse",
)
k4.metric(
    "Avg Churn Probability",
    f"{df['churn_probability'].mean():.3f}",
)

st.divider()

# Row 1 — tier bar + probability histogram 
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Customer Count by Risk Tier")

    tier_counts = (
        df["risk_tier"]
        .value_counts()
        .reindex(TIER_ORDER)
        .reset_index()
        .rename(columns={"risk_tier": "Risk Tier", "count": "Customers"})
    )

    fig_tier = px.bar(
        tier_counts,
        x="Risk Tier", y="Customers",
        color="Risk Tier",
        color_discrete_map=TIER_COLORS,
        text="Customers",
        category_orders={"Risk Tier": TIER_ORDER},
    )
    fig_tier.update_traces(textposition="outside", texttemplate="%{text:,}")
    fig_tier.update_layout(showlegend=False, margin=dict(t=10, b=10), height=380)
    st.plotly_chart(fig_tier, use_container_width=True)

with col_right:
    st.subheader("Churn Probability Distribution")

    fig_hist = px.histogram(
        df, x="churn_probability",
        nbins=40,
        color_discrete_sequence=["#6366f1"],
        labels={"churn_probability": "Churn Probability", "count": "Customers"},
    )
    fig_hist.add_vline(
        x=0.3, line_dash="dash", line_color=TIER_COLORS["High"], line_width=2,
        annotation_text="Threshold (0.3)",
        annotation_position="top right",
    )
    fig_hist.add_vline(
        x=0.5, line_dash="dot", line_color=TIER_COLORS["Critical"], line_width=1.5,
        annotation_text="0.5",
        annotation_position="top right",
    )
    fig_hist.add_vline(
        x=0.7, line_dash="dot", line_color=TIER_COLORS["Critical"], line_width=1.5,
        annotation_text="Critical (0.7)",
        annotation_position="top right",
    )
    fig_hist.update_layout(margin=dict(t=10, b=10), height=380)
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# Row 2 — grouped bar: avg tenure + charges by tier 
st.subheader("Average Tenure and Monthly Charges by Risk Tier")

tier_profile = (
    df.groupby("risk_tier", observed=True)[["tenure", "MonthlyCharges"]]
    .mean()
    .reindex(TIER_ORDER)
    .reset_index()
)

fig_group = go.Figure()
fig_group.add_trace(go.Bar(
    name="Avg Tenure (months)",
    x=tier_profile["risk_tier"],
    y=tier_profile["tenure"].round(1),
    marker_color="#6366f1",
    text=tier_profile["tenure"].round(1),
    textposition="outside",
    texttemplate="%{text} mo",
))
fig_group.add_trace(go.Bar(
    name="Avg Monthly Charges ($)",
    x=tier_profile["risk_tier"],
    y=tier_profile["MonthlyCharges"].round(2),
    marker_color=TIER_COLORS["High"],
    text=tier_profile["MonthlyCharges"].round(2),
    textposition="outside",
    texttemplate="$%{text}",
))
fig_group.update_layout(
    barmode="group",
    xaxis=dict(categoryorder="array", categoryarray=TIER_ORDER),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40, b=10),
    height=380,
)
st.plotly_chart(fig_group, use_container_width=True)

st.divider()

# Filtered table
n_filtered = len(filtered)
n_total = len(df)
st.subheader(f"Customer Detail:  {n_filtered:,} of {n_total:,} customers")
st.caption("Sorted by churn probability (highest first). Use sidebar filters to narrow the view.")

st.dataframe(
    filtered
    .sort_values("churn_probability", ascending=False)
    .reset_index(drop=True),
    use_container_width=True,
    height=420,
    column_config={
        "customer_id": st.column_config.NumberColumn("Customer ID", format="%d"),
        "churn_probability": st.column_config.ProgressColumn(
            "Churn Probability",
            help="Model-predicted probability of churn",
            format="%.3f",
            min_value=0.0,
            max_value=1.0,
        ),
        "risk_tier": st.column_config.TextColumn("Risk Tier"),
        "predicted_churn": st.column_config.CheckboxColumn(
            "Predicted Churn",
            help="Flagged at threshold = 0.3",
        ),
        "tenure": st.column_config.NumberColumn("Tenure (mo)", format="%d mo"),
        "MonthlyCharges": st.column_config.NumberColumn(
            "Monthly Charges", format="$%.2f"
        ),
        "Contract": st.column_config.TextColumn("Contract"),
    },
)
