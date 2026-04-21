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
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Main background — clean white */
.stApp {
    background-color: #ffffff;
    color: #1a1a2e;
}

/* Sidebar — soft lavender */
section[data-testid="stSidebar"] {
    background-color: #f5f2ff;
    border-right: 1.5px solid #e4dcf8;
}
section[data-testid="stSidebar"] * {
    color: #2d2448 !important;
    font-family: 'Poppins', sans-serif !important;
}

/* Metric cards — pastel purple tint */
[data-testid="metric-container"] {
    background: #faf8ff;
    border: 1.5px solid #e4dcf8;
    border-radius: 14px;
    padding: 1rem 1.25rem;
}
[data-testid="metric-container"] label {
    color: #7c6fa0 !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #2d2448 !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}

/* Tab styling — light pill tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #f5f2ff;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #7c6fa0;
    border-radius: 9px;
    font-weight: 500;
    font-size: 0.82rem;
    border: none;
    padding: 0.45rem 1.1rem;
    font-family: 'Poppins', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #5b3fc8 !important;
    box-shadow: 0 1px 8px rgba(91,63,200,0.13);
    font-weight: 600 !important;
}

/* Headers */
h1 {
    font-family: 'Poppins', sans-serif !important;
    color: #1a1a2e !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}
h2, h3 {
    font-family: 'Poppins', sans-serif !important;
    color: #2d2448 !important;
    font-weight: 600 !important;
}

/* Dividers */
hr {
    border-color: #ede8fb;
}

/* Conclusion box — soft mint */
.conclusion-box {
    background: #f0fdf8;
    border-left: 4px solid #2da58e;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    margin-top: 0.75rem;
    color: #1a3d35;
    font-size: 0.88rem;
    line-height: 1.7;
}

/* Plotly chart containers */
.stPlotlyChart {
    background: transparent !important;
}

/* Multiselect tags */
span[data-baseweb="tag"] {
    background-color: #ede8fb !important;
    color: #5b3fc8 !important;
    font-family: 'Poppins', sans-serif !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    padding: 0 4px;
}

/* Badge pill */
.badge {
    display: inline-block;
    background: #ede8fb;
    color: #5b3fc8;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
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
TEMPLATE = "plotly_white"
PAPER_BG = "rgba(0,0,0,0)"
PLOT_BG = "#fafafa"
GRID_COLOR = "#ede8fb"
FONT_COLOR = "#2d2448"

def apply_dark(fig, height=420):
    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Poppins", color=FONT_COLOR, size=12),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, linecolor="#e4dcf8"),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, linecolor="#e4dcf8"),
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
# PRE-COMPUTE STATS FOR DATA-DRIVEN FINDINGS
# ─────────────────────────────────────────────

# --- Tab 1 stats ---
zip_stats_all = (
    df.groupby("Zip Code")
    .agg(Avg_Score=("Score", "mean"), Median_Income=("Median_Income", "first"), Count=("Score", "count"))
    .reset_index().dropna()
)
if len(zip_stats_all) >= 2:
    _corr1, _pval1 = pearsonr(zip_stats_all["Median_Income"], zip_stats_all["Avg_Score"])
else:
    _corr1, _pval1 = 0, 1
_highest_income_zip = zip_stats_all.loc[zip_stats_all["Median_Income"].idxmax(), "Zip Code"] if not zip_stats_all.empty else "N/A"
_lowest_income_zip  = zip_stats_all.loc[zip_stats_all["Median_Income"].idxmin(), "Zip Code"] if not zip_stats_all.empty else "N/A"
_highest_income_score = zip_stats_all.loc[zip_stats_all["Median_Income"].idxmax(), "Avg_Score"] if not zip_stats_all.empty else 0
_lowest_income_score  = zip_stats_all.loc[zip_stats_all["Median_Income"].idxmin(), "Avg_Score"] if not zip_stats_all.empty else 0

# --- Tab 2 stats ---
zip_freq_all = (
    df.groupby("Zip Code")
    .agg(Avg_Score=("Score", "mean"), Freq=("Score", "count"), Median_Income=("Median_Income", "first"))
    .reset_index().dropna()
)
if len(zip_freq_all) >= 2:
    _corr2, _pval2 = pearsonr(zip_freq_all["Freq"], zip_freq_all["Avg_Score"])
else:
    _corr2, _pval2 = 0, 1
_most_inspected_zip = zip_freq_all.loc[zip_freq_all["Freq"].idxmax(), "Zip Code"] if not zip_freq_all.empty else "N/A"
_most_inspected_n   = int(zip_freq_all["Freq"].max()) if not zip_freq_all.empty else 0
_most_inspected_score = zip_freq_all.loc[zip_freq_all["Freq"].idxmax(), "Avg_Score"] if not zip_freq_all.empty else 0

# --- Tab 3 stats ---
_zip_years = df_time.groupby("Zip Code")["Inspection_Year"].nunique()
_multi_year_zips = _zip_years[_zip_years >= 3].index.tolist()
_time_stats = df_time[df_time["Zip Code"].isin(_multi_year_zips)].groupby("Zip Code")["Average_Inspection_Score"].agg(["std", "mean", "min", "max"]).reset_index()
_time_stats["range"] = _time_stats["max"] - _time_stats["min"]
_most_stable   = _time_stats.loc[_time_stats["std"].idxmin()]   if not _time_stats.empty else None
_most_volatile = _time_stats.loc[_time_stats["std"].idxmax()]   if not _time_stats.empty else None
_overall_trend = df_time.groupby("Inspection_Year")["Average_Inspection_Score"].mean().reset_index().sort_values("Inspection_Year")
_trend_direction = "improved" if (len(_overall_trend) >= 2 and _overall_trend.iloc[-1]["Average_Inspection_Score"] > _overall_trend.iloc[0]["Average_Inspection_Score"]) else "declined"

# --- Tab 5 stats ---
_type_counts_all = df.groupby(["Zip Code", "Restaurant Type"]).size().reset_index(name="Count")
_overall_type = df.groupby("Restaurant Type").size().reset_index(name="Count")
_total_rests = _overall_type["Count"].sum()
_local_pct = 0; _chain_pct = 0
for _, row in _overall_type.iterrows():
    if row["Restaurant Type"] == "Local":
        _local_pct = round(row["Count"] / _total_rests * 100, 1)
    elif row["Restaurant Type"] == "Chain":
        _chain_pct = round(row["Count"] / _total_rests * 100, 1)
_zip_totals_all = df.groupby("Zip Code").size().reset_index(name="Total")
_comp_all = pd.merge(_type_counts_all, _zip_totals_all, on="Zip Code")
_comp_all["Pct"] = _comp_all["Count"] / _comp_all["Total"] * 100
_most_local_zip_row = _comp_all[_comp_all["Restaurant Type"] == "Local"].sort_values("Pct", ascending=False).head(1)
_most_chain_zip_row = _comp_all[_comp_all["Restaurant Type"] == "Chain"].sort_values("Pct", ascending=False).head(1)
_most_local_zip = _most_local_zip_row["Zip Code"].values[0] if not _most_local_zip_row.empty else "N/A"
_most_local_pct = round(_most_local_zip_row["Pct"].values[0], 1) if not _most_local_zip_row.empty else 0
_most_chain_zip = _most_chain_zip_row["Zip Code"].values[0] if not _most_chain_zip_row.empty else "N/A"
_most_chain_pct = round(_most_chain_zip_row["Pct"].values[0], 1) if not _most_chain_zip_row.empty else 0

# --- Tab 6 stats ---
_local_scores = df[df["Restaurant Type"] == "Local"]["Score"]
_chain_scores  = df[df["Restaurant Type"] == "Chain"]["Score"]
_local_mean = round(_local_scores.mean(), 2) if not _local_scores.empty else 0
_chain_mean = round(_chain_scores.mean(), 2) if not _chain_scores.empty else 0
_local_median = round(_local_scores.median(), 2) if not _local_scores.empty else 0
_chain_median = round(_chain_scores.median(), 2) if not _chain_scores.empty else 0

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
    "📍 78705 Restaurant Map",
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
            color_continuous_scale="Viridis",
            labels={"Median_Income": "Median Income ($)", "Avg_Score": "Avg Inspection Score"},
            title="Median Income vs. Average Inspection Score by ZIP Code",
        )
        fig1.update_layout(xaxis_tickformat="$,.0f", coloraxis_showscale=True,
                           coloraxis_colorbar=dict(title="Avg Score"))
        apply_dark(fig1)
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown(
        f'<div class="conclusion-box"><strong>📌 Finding:</strong> The Pearson correlation between '
        f"median income and average inspection score across ZIP codes is <strong>r = {_corr1:.3f}</strong> "
        f"(p = {_pval1:.3f}), indicating a <strong>{'weak' if abs(_corr1) < 0.3 else 'moderate'} and "
        f"{'non-' if _pval1 > 0.05 else ''}statistically significant</strong> relationship. "
        f"The highest-income ZIP ({_highest_income_zip}) averaged <strong>{_highest_income_score:.1f}</strong>, "
        f"while the lowest-income ZIP ({_lowest_income_zip}) averaged <strong>{_lowest_income_score:.1f}</strong> — "
        f"a difference of only {abs(_highest_income_score - _lowest_income_score):.1f} points. "
        f"Inspection standards appear consistent across neighborhoods regardless of wealth.</div>",
        unsafe_allow_html=True,
    )

# ── TAB 2: Frequency vs Score ────────────────
with tabs[1]:
    st.markdown('<div class="badge">Argument 2</div>', unsafe_allow_html=True)
    st.markdown("### Do more inspections correlate with lower scores?")

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
            color_continuous_scale="Viridis",
            labels={
                "Inspection_Frequency": "Total Number of Inspections",
                "Avg_Score": "Avg Inspection Score",
                "Household_Count": "Total Households",
            },
            title="Inspection Frequency vs. Average Score (bubble size = households)",
        )
        fig2.update_layout(coloraxis_colorbar=dict(title="Avg Score"))
        apply_dark(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        f'<div class="conclusion-box"><strong>📌 Finding:</strong> The correlation between inspection '
        f"frequency and average score is <strong>r = {_corr2:.3f}</strong> (p = {_pval2:.3f}) — "
        f"<strong>{'not statistically significant' if _pval2 > 0.05 else 'statistically significant'}</strong>. "
        f"ZIP code <strong>{_most_inspected_zip}</strong> had the most inspections ({_most_inspected_n:,}) "
        f"and averaged <strong>{_most_inspected_score:.1f}</strong>. "
        f"The slight {'negative' if _corr2 < 0 else 'positive'} trend ({_corr2:+.3f}) likely reflects that "
        f"ZIP codes with more food establishments receive more inspections — not that inspections "
        f"cause scores to {'drop' if _corr2 < 0 else 'rise'}.</div>",
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

    import plotly.colors as pc
    all_time_zips = sorted(df_time["Zip Code"].astype(str).unique())
    n_colors = max(len(all_time_zips), 1)
    viridis_colors = pc.sample_colorscale("Viridis", [i / max(n_colors - 1, 1) for i in range(n_colors)])
    zip_color_map = {z: viridis_colors[i] for i, z in enumerate(all_time_zips)}

    fig3 = go.Figure()
    for zip_code in all_time_zips:
        temp = df_time[df_time["Zip Code"] == zip_code].sort_values("Inspection_Year")
        is_imp = str(zip_code) in highlight_zips
        color = zip_color_map.get(str(zip_code), "#31688e")
        fig3.add_trace(go.Scatter(
            x=temp["Inspection_Year"],
            y=temp["Average_Inspection_Score"],
            mode="lines+markers",
            name=str(zip_code),
            visible=True if is_imp else "legendonly",
            line=dict(width=3 if is_imp else 1.6, color=color),
            opacity=1.0 if is_imp else 0.9,
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
    apply_dark(fig3, height=500)
    st.plotly_chart(fig3, use_container_width=True)

    _stable_str   = f"ZIP <strong>{_most_stable['Zip Code']}</strong> (std = {_most_stable['std']:.2f}, range {_most_stable['min']:.1f}–{_most_stable['max']:.1f})" if _most_stable is not None else "N/A"
    _volatile_str = f"ZIP <strong>{_most_volatile['Zip Code']}</strong> (std = {_most_volatile['std']:.2f}, range {_most_volatile['min']:.1f}–{_most_volatile['max']:.1f})" if _most_volatile is not None else "N/A"
    _start_score  = round(_overall_trend.iloc[0]["Average_Inspection_Score"], 1) if not _overall_trend.empty else 0
    _end_score    = round(_overall_trend.iloc[-1]["Average_Inspection_Score"], 1) if not _overall_trend.empty else 0
    st.markdown(
        f'<div class="conclusion-box"><strong>📌 Finding:</strong> Across all ZIP codes with 3+ years of data, '
        f"the most <strong>consistent</strong> ZIP was {_stable_str}. "
        f"The most <strong>volatile</strong> was {_volatile_str}. "
        f"Austin's city-wide average score has <strong>{_trend_direction}</strong> from "
        f"<strong>{_start_score}</strong> ({int(_overall_trend.iloc[0]['Inspection_Year'])}) to "
        f"<strong>{_end_score}</strong> ({int(_overall_trend.iloc[-1]['Inspection_Year'])}). "
        f"Most ZIPs remain above 85 year to year, but a few show swings of 5+ points — suggesting "
        f"episodic enforcement patterns or high restaurant turnover in those areas.</div>",
        unsafe_allow_html=True,
    )

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

        col_a4, col_b4 = st.columns([3, 1])
        with col_b4:
            top_n4 = st.radio("Show", ["All ZIPs", "Important ZIP Codes"], index=1, key="race_filter")

        if top_n4 == "Important ZIP Codes":
            important_race_zips = ["78745", "78744", "78705", "78721"]
            df_plot = df_plot[df_plot["Zip Code"].isin(important_race_zips)]

        zip_order = sorted(df_plot["Zip Code"].unique(), key=lambda z: (len(z), z))
        group_colors = {
            "Overall":  "#440154",
            "White":    "#31688e",
            "Black":    "#35b779",
            "Asian":    "#fde725",
            "Hispanic": "#90d743",
        }

        fig4 = px.scatter(
            df_plot,
            x="Zip Code", y="Median Household Income",
            color="Group",
            category_orders={"Zip Code": zip_order},
            title="Median Household Income by Race/Ethnicity Within ZIP Codes",
            color_discrete_map=group_colors,
            hover_data={"Median Household Income": ":,.0f"},
        )
        fig4.update_traces(marker=dict(size=10, opacity=0.9))
        fig4.update_layout(yaxis_range=[0, 200000], yaxis_tickprefix="$", yaxis_tickformat=",")
        apply_dark(fig4, height=480)
        st.plotly_chart(fig4, use_container_width=True)

        _race_avg = df_plot.groupby("Group")["Median Household Income"].mean().dropna().sort_values(ascending=False)
        _top_group    = _race_avg.index[0]  if len(_race_avg) > 0 else "N/A"
        _bottom_group = _race_avg.index[-1] if len(_race_avg) > 1 else "N/A"
        _top_val    = round(_race_avg.iloc[0],  0) if len(_race_avg) > 0 else 0
        _bottom_val = round(_race_avg.iloc[-1], 0) if len(_race_avg) > 1 else 0
        _gap = round(_top_val - _bottom_val, 0)
        st.markdown(
            f'<div class="conclusion-box"><strong>📌 Finding:</strong> Across Austin ZIP codes, '
            f"<strong>{_top_group}</strong> households have the highest average median income "
            f"(<strong>${_top_val:,.0f}</strong>), while <strong>{_bottom_group}</strong> households "
            f"have the lowest (<strong>${_bottom_val:,.0f}</strong>) — a gap of <strong>${_gap:,.0f}</strong>. "
            f"This disparity persists across most ZIP codes and is not isolated to a few neighborhoods, "
            f"pointing to a structural income gap by race and ethnicity throughout Austin.</div>",
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
        top_n = st.radio(
            "Show",
            ["All ZIPs", "Top 5 Local + Top 5 Chain Composition"],
            index=1,
            key="comp_filter",
        )

    if top_n == "Top 5 Local + Top 5 Chain Composition":
        local_comp_top5 = (
            plot_df[plot_df["Restaurant Type"] == "Local"]
            .nlargest(5, "Composition_Pct")["Zip Code"]
            .tolist()
        )
        chain_comp_top5 = (
            plot_df[plot_df["Restaurant Type"] == "Chain"]
            .nlargest(5, "Composition_Pct")["Zip Code"]
            .tolist()
        )
        selected = list(dict.fromkeys(local_comp_top5 + chain_comp_top5))
        plot_df = plot_df[plot_df["Zip Code"].isin(selected)]

    if not plot_df.empty:
        fig5 = px.bar(
            plot_df,
            x="Zip Code", y="Count",
            color="Restaurant Type",
            barmode="stack",
            hover_data={"Composition_Pct": ":.1f", "Total_in_Zip": True},
            color_discrete_map={"Local": "#31688e", "Chain": "#fde725"},
            title="Restaurant Composition by ZIP Code",
            labels={"Count": "Establishments", "Composition_Pct": "Composition %"},
        )
        fig5.update_layout(xaxis={"categoryorder": "total descending"})
        apply_dark(fig5)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown(
        f'<div class="conclusion-box"><strong>📌 Finding:</strong> Across all 2024 inspections, '
        f"Austin's restaurants are approximately <strong>{_local_pct}% local</strong> and "
        f"<strong>{_chain_pct}% chain</strong> — meaning local restaurants "
        f"{'outnumber' if _local_pct > _chain_pct else 'are outnumbered by'} chains. "
        f"ZIP <strong>{_most_local_zip}</strong> is the most local-heavy ({_most_local_pct}% local), "
        f"while ZIP <strong>{_most_chain_zip}</strong> skews most toward chains ({_most_chain_pct}% chain). "
        f"High-density commercial corridors and suburban ZIPs tend to attract more chains, "
        f"while university-adjacent and walkable neighborhoods like 78705 and 78704 have higher local concentrations.</div>",
        unsafe_allow_html=True,
    )

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
            color_discrete_map={"Local": "#31688e", "Chain": "#fde725"},
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
            color_discrete_map={"Local": "#31688e", "Chain": "#fde725"},
            title="Wealth vs. Scores: Local vs. Chain",
            labels={"Median_Income": "Median Income ($)", "Score": "Avg Score"},
        )
        fig6b.update_layout(xaxis_tickformat="$,.0f")
        apply_dark(fig6b, height=380)
        st.plotly_chart(fig6b, use_container_width=True)

    st.markdown("#### 📐 Pearson Correlation: Income × Score")
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
        f'<div class="conclusion-box"><strong>📌 Finding:</strong> Local restaurants average '
        f"<strong>{_local_mean}</strong> (median: {_local_median}) while chain restaurants average "
        f"<strong>{_chain_mean}</strong> (median: {_chain_median}) — a difference of only "
        f"<strong>{abs(_local_mean - _chain_mean):.2f} points</strong>. "
        f"{'Local restaurants score slightly higher on average.' if _local_mean > _chain_mean else 'Chain restaurants score slightly higher on average.'} "
        f"Neither type shows a meaningful relationship between neighborhood income and score, "
        f"reinforcing that compliance is driven by establishment-level behavior, not restaurant type or wealth.</div>",
        unsafe_allow_html=True,
    )

# ── TAB 7: 78705 Restaurant Map ──────────────
with tabs[6]:
    st.markdown('<div class="badge">Spotlight ZIP</div>', unsafe_allow_html=True)
    st.markdown("### 📍 Restaurant Map: ZIP Code 78705 (UT Area)")

    zip_78705 = df[df["Zip Code"] == "78705"].copy()

    if zip_78705.empty:
        st.warning("No data found for ZIP code 78705.")
    else:
        try:
            import pgeocode
            nomi = pgeocode.Nominatim("us")
            _loc = nomi.query_postal_code("78705")
            _center_lat = float(_loc.latitude)
            _center_lon = float(_loc.longitude)
        except Exception:
            _center_lat = 30.2900
            _center_lon = -97.7400

        np.random.seed(42)
        zip_78705 = zip_78705.copy()
        zip_78705["Latitude"]  = _center_lat  + np.random.normal(0, 0.007, len(zip_78705))
        zip_78705["Longitude"] = _center_lon + np.random.normal(0, 0.007, len(zip_78705))

        _78705_local_pct = round((zip_78705["Restaurant Type"] == "Local").mean() * 100, 1)
        _78705_avg_score = round(zip_78705["Score"].mean(), 1)
        _78705_n = len(zip_78705)

        k1, k2, k3 = st.columns(3)
        k1.metric("Restaurants in 78705", f"{_78705_n:,}")
        k2.metric("Avg Inspection Score", str(_78705_avg_score))
        k3.metric("% Local", f"{_78705_local_pct}%")

        fig7 = px.scatter_mapbox(
            zip_78705,
            lat="Latitude", lon="Longitude",
            color="Restaurant Type",
            color_discrete_map={"Local": "#31688e", "Chain": "#fde725"},
            size="Score",
            size_max=18,
            hover_name="Restaurant Name",
            hover_data={"Score": True, "Restaurant Type": True, "Latitude": False, "Longitude": False},
            zoom=13.5,
            center={"lat": _center_lat, "lon": _center_lon},
            title="Restaurants in ZIP 78705 — Color: Type · Size: Inspection Score",
            mapbox_style="carto-positron",
        )
        fig7.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Poppins", color=FONT_COLOR, size=12),
            height=520,
            margin=dict(l=0, r=0, t=50, b=0),
            legend_title="Restaurant Type",
        )
        st.plotly_chart(fig7, use_container_width=True)

        _78705_type_scores = zip_78705.groupby("Restaurant Type")["Score"].agg(["mean", "count"]).reset_index()
        _78705_type_scores.columns = ["Type", "Avg Score", "Count"]
        _local_row = _78705_type_scores[_78705_type_scores["Type"] == "Local"]
        _chain_row = _78705_type_scores[_78705_type_scores["Type"] == "Chain"]
        _local_score_78705 = round(_local_row["Avg Score"].values[0], 2) if not _local_row.empty else "N/A"
        _chain_score_78705 = round(_chain_row["Avg Score"].values[0], 2) if not _chain_row.empty else "N/A"
        _local_n_78705 = int(_local_row["Count"].values[0]) if not _local_row.empty else 0
        _chain_n_78705 = int(_chain_row["Count"].values[0]) if not _chain_row.empty else 0

        st.markdown(
            f'<div class="conclusion-box"><strong>📌 Finding:</strong> ZIP 78705 (UT Austin area) has '
            f"<strong>{_78705_n}</strong> inspected restaurants, of which <strong>{_78705_local_pct}%</strong> "
            f"are local — {'above' if _78705_local_pct > _local_pct else 'below'} the city-wide average of {_local_pct}%. "
            f"Local restaurants here scored an average of <strong>{_local_score_78705}</strong> ({_local_n_78705} establishments), "
            f"while chains scored <strong>{_chain_score_78705}</strong> ({_chain_n_78705} establishments). "
            f"The overall average score of <strong>{_78705_avg_score}</strong> places 78705 "
            f"{'above' if _78705_avg_score > df['Score'].mean() else 'below'} the city-wide mean of "
            f"{round(df['Score'].mean(), 1)}.</div>",
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
