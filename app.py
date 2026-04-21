import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Austin Food Safety Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp {
    background-color: #fffff;
    color: #e8e8e8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #2a2f3e;
}
section[data-testid="stSidebar"] * {
    color: #c9d1e0 !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a2035 0%, #1e263a 100%);
    border: 1px solid #2e3650;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
[data-testid="metric-container"] label {
    color: #8892a4 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8eaf6 !important;
    font-family: 'DM Serif Display', serif !important;
    font-size: 2rem !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background-color: #161b27;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8892a4;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.85rem;
    border: none;
    padding: 0.5rem 1.25rem;
}
.stTabs [aria-selected="true"] {
    background: #2e3a5e !important;
    color: #a8c7fa !important;
}

/* Headers */
h1 {
    font-family: 'DM Serif Display', serif !important;
    color: #e8eaf6 !important;
    font-size: 2.4rem !important;
    letter-spacing: -0.02em !important;
}
h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #d0d5e8 !important;
}

/* Section dividers */
hr {
    border-color: #2a2f3e;
}

/* Conclusion box */
.conclusion-box {
    background: linear-gradient(135deg, #1a2035 0%, #1c2540 100%);
    border-left: 3px solid #5b8dee;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    margin-top: 0.75rem;
    color: #b0bcd0;
    font-size: 0.9rem;
    line-height: 1.65;
}

/* Plotly chart containers */
.stPlotlyChart {
    background: transparent !important;
}

/* Multiselect tags */
span[data-baseweb="tag"] {
    background-color: #2e3a5e !important;
    color: #a8c7fa !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    padding: 0 4px;
}

/* Badge pill */
.badge {
    display: inline-block;
    background: #2e3a5e;
    color: #a8c7fa;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df_insp_raw = pd.read_csv("2024_Food_Establishment_Inspection_Scores_Edited.csv")
    df_census_raw = pd.read_csv("Census_Data.csv")

    # Clean inspection data
    df_insp = df_insp_raw.dropna(subset=["Zip Code", "Score"]).copy()
    df_insp["Zip Code"] = df_insp["Zip Code"].astype(str).str[:5]
    df_insp["Score"] = pd.to_numeric(df_insp["Score"], errors="coerce")
    df_insp["Inspection Date"] = pd.to_datetime(df_insp["Inspection Date"], errors="coerce")
    df_insp["Facility ID"] = df_insp["Facility ID"].astype(str)
    df_insp["Restaurant Type"] = df_insp["Restaurant Type"].str.strip().str.capitalize()
    df_insp = df_insp[(df_insp["Score"] >= 0) & (df_insp["Score"] <= 100)]

    # Clean census data
    df_census = df_census_raw.iloc[1:].copy()
    df_census["Zip Code"] = (
        df_census["NAME"].str.replace("ZCTA5 ", "", regex=False).str.strip().str[:5]
    )
    df_census["Median_Income"] = (
        df_census["S1903_C03_001E"]
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace("-", "", regex=False)
    )
    df_census["Median_Income"] = pd.to_numeric(df_census["Median_Income"], errors="coerce")
    df_census["Household_Count"] = pd.to_numeric(
        df_census.get("S1903_C01_001E", pd.Series(dtype=float)), errors="coerce"
    )
    df_census_final = df_census[["Zip Code", "Median_Income", "Household_Count"]].dropna(subset=["Zip Code", "Median_Income"])

    # Race columns
    race_cols_map = {
        "S1903_C03_001E": "Overall",
        "S1903_C03_002E": "White",
        "S1903_C03_003E": "Black",
        "S1903_C03_005E": "Asian",
        "S1903_C03_009E": "Hispanic",
    }
    for col in race_cols_map:
        if col in df_census.columns:
            df_census[col] = pd.to_numeric(
                df_census[col].astype(str).str.replace(r"[,\+\-]", "", regex=True),
                errors="coerce",
            )

    df_2024 = df_insp[df_insp["Inspection Date"].dt.year == 2024].copy()
    df_2024_final = pd.merge(df_2024, df_census_final, on="Zip Code", how="inner").reset_index(drop=True)

    # Time-series data (all years)
    df_time = (
        df_insp.assign(Inspection_Year=df_insp["Inspection Date"].dt.year)
        .groupby(["Zip Code", "Inspection_Year"])
        .agg(Average_Inspection_Score=("Score", "mean"))
        .reset_index()
    )

    return df_2024_final, df_time, df_census, race_cols_map, df_census_final, df_2024

# Load
with st.spinner("Loading data…"):
    df, df_time, df_census, race_cols_map, df_census_final, df_2024 = load_data()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ Austin Food Safety")
    st.markdown("**CS329E Final Project**")
    st.markdown("---")

    st.markdown("### Filters")
    all_types = sorted(df["Restaurant Type"].dropna().unique())
    selected_types = st.multiselect(
        "Restaurant Type", options=all_types, default=all_types
    )

    score_range = st.slider("Inspection Score Range", 0, 100, (60, 100))

    all_zips = sorted(df["Zip Code"].unique())
    selected_zips = st.multiselect(
        "Zip Codes (leave empty = all)", options=all_zips, default=[]
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#555f7e'>Data: 2024 Austin Food Establishment Inspection Scores + US Census ACS</small>",
        unsafe_allow_html=True,
    )

# Apply filters
filtered = df[
    (df["Restaurant Type"].isin(selected_types)) &
    (df["Score"].between(score_range[0], score_range[1]))
]
if selected_zips:
    filtered = filtered[filtered["Zip Code"].isin(selected_zips)]

# ─────────────────────────────────────────────
# PLOTLY DARK TEMPLATE
# ─────────────────────────────────────────────
TEMPLATE = "plotly_dark"
PAPER_BG = "rgba(0,0,0,0)"
PLOT_BG = "rgba(0,0,0,0)"
GRID_COLOR = "#1e2436"
FONT_COLOR = "#c9d1e0"

def apply_dark(fig, height=420):
    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="DM Sans", color=FONT_COLOR, size=12),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
    )
    return fig

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("<h1>Austin Food Safety Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#8892a4; font-size:1rem; margin-top:-0.5rem; margin-bottom:1.5rem;'>"
    "Exploring food inspection scores, neighborhood wealth, and restaurant type across Austin ZIP codes — 2024</p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Inspections", f"{len(filtered):,}")
c2.metric("Avg Score", f"{filtered['Score'].mean():.1f}" if len(filtered) else "—")
c3.metric("ZIP Codes", filtered["Zip Code"].nunique())
c4.metric("Median Income (Avg)", f"${filtered['Median_Income'].mean():,.0f}" if len(filtered) else "—")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📊 Income vs. Score",
    "🔁 Frequency vs. Score",
    "📈 Scores Over Time",
    "💰 Income by Race",
    "🏪 Restaurant Composition",
    "⚖️ Local vs. Chain",
])

# ── TAB 1: Income vs Score ───────────────────
with tabs[0]:
    st.markdown('<div class="badge">Argument 1</div>', unsafe_allow_html=True)
    st.markdown("### Does neighborhood wealth predict food safety?")

    zip_stats = (
        filtered.groupby("Zip Code")
        .agg(
            Avg_Score=("Score", "mean"),
            Median_Income=("Median_Income", "first"),
            Inspection_Frequency=("Restaurant Name", "count"),
        )
        .reset_index()
    )
    zip_stats["Avg_Score"] = zip_stats["Avg_Score"].round(2)

    if not zip_stats.empty:
        fig1 = px.scatter(
            zip_stats,
            x="Median_Income", y="Avg_Score",
            size="Inspection_Frequency",
            color="Avg_Score",
            hover_name="Zip Code",
            hover_data={"Median_Income": ":$,.0f", "Avg_Score": ":.2f", "Inspection_Frequency": True},
            trendline="ols",
            color_continuous_scale="Blues",
            labels={"Median_Income": "Median Income ($)", "Avg_Score": "Avg Inspection Score"},
            title="Median Income vs. Average Inspection Score by ZIP Code",
        )
        fig1.update_layout(xaxis_tickformat="$,.0f", coloraxis_showscale=False)
        apply_dark(fig1)
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown(
        '<div class="conclusion-box"><strong>Finding:</strong> No meaningful relationship exists '
        "between average restaurant inspection scores and neighborhood median income — suggesting "
        "that inspection standards are applied uniformly regardless of socioeconomic status. "
        "High-income areas are not inherently cleaner, and low-income areas are not inherently less hygienic.</div>",
        unsafe_allow_html=True,
    )

# ── TAB 2: Frequency vs Score ────────────────
with tabs[1]:
    st.markdown('<div class="badge">Argument 2</div>', unsafe_allow_html=True)
    st.markdown("### Do more inspections lead to lower scores?")

    fig2_stats = (
        filtered.groupby("Zip Code")
        .agg(
            Avg_Score=("Score", "mean"),
            Median_Income=("Median_Income", "first"),
            Inspection_Frequency=("Restaurant Name", "count"),
            Household_Count=("Household_Count", "first"),
        )
        .reset_index()
    )

    if not fig2_stats.empty:
        fig2 = px.scatter(
            fig2_stats,
            x="Inspection_Frequency", y="Avg_Score",
            size="Household_Count",
            color="Avg_Score",
            hover_name="Zip Code",
            hover_data={"Inspection_Frequency": True, "Avg_Score": ":.2f", "Median_Income": ":$,.0f"},
            trendline="ols",
            color_continuous_scale="Teal",
            labels={
                "Inspection_Frequency": "Total Number of Inspections",
                "Avg_Score": "Avg Inspection Score",
                "Household_Count": "Total Households",
            },
            title="Inspection Frequency vs. Average Score (bubble size = households)",
        )
        fig2.update_layout(coloraxis_showscale=False)
        apply_dark(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        '<div class="conclusion-box"><strong>Finding:</strong> No statistically significant relationship '
        "between inspection frequency and average score. A slight negative slope may suggest that "
        "ZIP codes receiving more inspections could be flagged for prior issues — but the effect is weak.</div>",
        unsafe_allow_html=True,
    )

# ── TAB 3: Scores Over Time ──────────────────
with tabs[2]:
    st.markdown('<div class="badge">Argument 3</div>', unsafe_allow_html=True)
    st.markdown("### How have inspection scores changed year over year?")

    important_zips = ["78701", "78705", "78739", "78704", "78621", "78641", "78751", "78721"]
    highlight_zips = st.multiselect(
        "Highlight ZIP codes", options=sorted(df_time["Zip Code"].unique()), default=important_zips
    )

    fig3 = go.Figure()
    for zip_code in df_time["Zip Code"].unique():
        temp = df_time[df_time["Zip Code"] == zip_code].sort_values("Inspection_Year")
        is_imp = str(zip_code) in highlight_zips
        fig3.add_trace(go.Scatter(
            x=temp["Inspection_Year"],
            y=temp["Average_Inspection_Score"],
            mode="lines+markers",
            name=str(zip_code),
            visible=True if is_imp else "legendonly",
            line=dict(width=3 if is_imp else 1, color=None),
            opacity=1.0 if is_imp else 0.3,
            hovertemplate=f"<b>ZIP {zip_code}</b><br>Year: %{{x}}<br>Score: %{{y:.2f}}<extra></extra>",
        ))
    fig3.update_layout(
        title="Avg Inspection Score Over Time by ZIP Code",
        xaxis_title="Year",
        yaxis_title="Avg Score",
        yaxis=dict(range=[75, 101]),
        xaxis=dict(dtick=1),
        legend_title="ZIP (click to toggle)",
    )
    apply_dark(fig3, height=480)
    st.plotly_chart(fig3, use_container_width=True)

# ── TAB 4: Income by Race ────────────────────
with tabs[3]:
    st.markdown('<div class="badge">Argument 4</div>', unsafe_allow_html=True)
    st.markdown("### How does median income vary by race/ethnicity across ZIP codes?")

    available_race_cols = {k: v for k, v in race_cols_map.items() if k in df_census.columns}
    if available_race_cols:
        df_plot = df_census[["Zip Code"] + list(available_race_cols.keys())].melt(
            id_vars="Zip Code",
            value_vars=list(available_race_cols.keys()),
            var_name="Group",
            value_name="Median Household Income",
        )
        df_plot["Group"] = df_plot["Group"].map(available_race_cols)
        df_plot = df_plot.dropna(subset=["Zip Code", "Median Household Income"])
        zip_order = sorted(df_plot["Zip Code"].unique(), key=lambda z: (len(z), z))

        fig4 = px.scatter(
            df_plot,
            x="Zip Code", y="Median Household Income",
            color="Group",
            category_orders={"Zip Code": zip_order},
            title="Median Household Income by Race/Ethnicity Within ZIP Codes",
            color_discrete_sequence=px.colors.qualitative.Bold,
            hover_data={"Median Household Income": ":,.0f"},
        )
        fig4.update_traces(marker=dict(size=8, opacity=0.85))
        fig4.update_layout(yaxis_range=[0, 200000], yaxis_tickprefix="$", yaxis_tickformat=",")
        apply_dark(fig4, height=480)
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown(
            '<div class="conclusion-box"><strong>Finding:</strong> Significant income gaps exist '
            "between racial/ethnic groups within the same ZIP codes, with White and Asian households "
            "consistently showing higher median incomes than Black and Hispanic households across Austin.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Race columns not found in Census data.")

# ── TAB 5: Restaurant Composition ────────────
with tabs[4]:
    st.markdown('<div class="badge">Argument 5</div>', unsafe_allow_html=True)
    st.markdown("### How are local vs. chain restaurants distributed across ZIP codes?")

    type_counts = (
        filtered.groupby(["Zip Code", "Restaurant Type"]).size().reset_index(name="Count")
    )
    zip_totals = filtered.groupby("Zip Code").size().reset_index(name="Total_in_Zip")
    plot_df = pd.merge(type_counts, zip_totals, on="Zip Code")
    plot_df["Composition_Pct"] = (plot_df["Count"] / plot_df["Total_in_Zip"] * 100).round(2)

    col_a, col_b = st.columns([3, 1])
    with col_b:
        top_n = st.radio("Show", ["All ZIPs", "Top 10 Hubs"], index=1)

    if top_n == "Top 10 Hubs":
        top_local = type_counts[type_counts["Restaurant Type"] == "Local"].nlargest(5, "Count")["Zip Code"]
        top_chain = type_counts[type_counts["Restaurant Type"] == "Chain"].nlargest(5, "Count")["Zip Code"]
        selected = pd.concat([top_local, top_chain]).unique()
        plot_df = plot_df[plot_df["Zip Code"].isin(selected)]

    if not plot_df.empty:
        fig5 = px.bar(
            plot_df,
            x="Zip Code", y="Count",
            color="Restaurant Type",
            barmode="stack",
            hover_data={"Composition_Pct": ":.1f", "Total_in_Zip": True},
            color_discrete_map={"Local": "#4a90d9", "Chain": "#e8a838"},
            title="Restaurant Composition by ZIP Code",
            labels={"Count": "Establishments", "Composition_Pct": "Composition %"},
        )
        fig5.update_layout(xaxis={"categoryorder": "total descending"})
        apply_dark(fig5)
        st.plotly_chart(fig5, use_container_width=True)

# ── TAB 6: Local vs Chain Scores ─────────────
with tabs[5]:
    st.markdown('<div class="badge">Argument 6</div>', unsafe_allow_html=True)
    st.markdown("### Do local and chain restaurants score differently?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Score Distribution")
        fig6a = px.box(
            filtered,
            x="Restaurant Type", y="Score",
            color="Restaurant Type",
            points="outliers",
            notched=False,
            color_discrete_map={"Local": "#4a90d9", "Chain": "#e8a838"},
            title="Inspection Score Statistics",
        )
        fig6a.update_traces(boxmean=True)
        fig6a.update_layout(showlegend=False)
        apply_dark(fig6a, height=380)
        st.plotly_chart(fig6a, use_container_width=True)

    with col2:
        st.markdown("#### Income vs. Score by Type")
        scatter_df = (
            filtered.groupby(["Zip Code", "Restaurant Type"])
            .agg(Score=("Score", "mean"), Median_Income=("Median_Income", "first"))
            .reset_index()
        )
        fig6b = px.scatter(
            scatter_df,
            x="Median_Income", y="Score",
            color="Restaurant Type",
            trendline="ols",
            hover_data=["Zip Code"],
            color_discrete_map={"Local": "#4a90d9", "Chain": "#e8a838"},
            title="Wealth vs. Scores: Local vs. Chain",
            labels={"Median_Income": "Median Income ($)", "Score": "Avg Score"},
        )
        fig6b.update_layout(xaxis_tickformat="$,.0f")
        apply_dark(fig6b, height=380)
        st.plotly_chart(fig6b, use_container_width=True)

    # Correlation stats
    st.markdown("#### Pearson Correlation: Income × Score")
    corr_cols = st.columns(len(all_types) or 1)
    for i, r_type in enumerate(all_types):
        subset = scatter_df[scatter_df["Restaurant Type"] == r_type].dropna()
        if len(subset) >= 2:
            corr, pval = pearsonr(subset["Median_Income"], subset["Score"])
            corr_cols[i].metric(
                f"{r_type} restaurants",
                f"r = {corr:.3f}",
                f"p = {pval:.3f}",
            )

    st.markdown(
        '<div class="conclusion-box"><strong>Finding:</strong> Local and chain restaurants show '
        "comparable inspection score distributions. The relationship between neighborhood income "
        "and scores is similarly weak for both categories, reinforcing that inspection outcomes "
        "are driven by establishment-level practices rather than wealth or restaurant type.</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# FOOTER: Raw Data
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 View raw filtered data"):
    st.dataframe(
        filtered[["Zip Code", "Restaurant Name", "Restaurant Type", "Score", "Inspection Date", "Median_Income"]]
        .sort_values("Score", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=320,
    )
    st.caption(f"{len(filtered):,} rows shown based on current filters.")
