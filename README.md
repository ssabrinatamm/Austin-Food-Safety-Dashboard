# Austin Food Safety Dashboard

A Streamlit dashboard that explores whether food establishment inspection scores in Austin are associated with neighborhood socioeconomic conditions. The project combines Austin food inspection data with U.S. Census ACS data to study relationships between inspection outcomes, household income, race/ethnicity, restaurant type, and ZIP code-level patterns.

## Project Thesis

Our initial goal was to determine whether restaurant inspection scores in Austin are associated with neighborhood income levels. Specifically, we tested whether lower median household income is significantly linked to lower food establishment inspection scores across ZIP codes.

Our thesis was that socioeconomic context may influence food safety outcomes, and that any measurable disparities can be identified through statistical and machine learning analysis. If a significant relationship exists, this would suggest that food safety risks are not evenly distributed across the city, raising important implications for public health equity. If no significant relationship is found, it would indicate that inspection standards are being applied consistently across neighborhoods regardless of income level.

As we explored the datasets further, we expanded the project to study restaurant type and race demographics. We examined local and chain restaurants across Austin ZIP codes, and we also analyzed median household income and race/ethnicity patterns throughout Austin ZIP codes.

## What This Project Analyzes

This dashboard focuses on the following questions:

1. Does neighborhood wealth predict food safety inspection scores?
2. Do ZIP codes with more inspections tend to have lower scores?
3. How have inspection scores changed over time by ZIP code?
4. How does median household income vary by race/ethnicity across ZIP codes?
5. How are local vs. chain restaurants distributed across Austin?
6. Do local and chain restaurants score differently?
7. What does the restaurant landscape look like in ZIP code 78705 near UT Austin?

## Data Sources

- 2024_Food_Establishment_Inspection_Scores_Edited.csv
- Census_Data.csv

### Inspection Dataset

The food inspection dataset is cleaned and filtered to:

- keep only rows with valid ZIP codes and scores
- standardize ZIP codes to 5 digits
- convert scores to numeric values
- parse inspection dates
- keep scores between 0 and 100
- normalize restaurant type labels

### Census Dataset

The Census data is used to provide neighborhood context, including:

- median household income
- household count
- race/ethnicity-based income measures

## Tools and Libraries

Visualization and analysis packages used in the project:

- Plotly Express
- Plotly Graph Objects
- Matplotlib
- Pandas

Additional libraries used in the dashboard implementation:

- Streamlit
- NumPy
- SciPy
- pgeocode for ZIP code coordinate lookup when available

## Dashboard Overview

The main application is implemented as a Streamlit dashboard with sidebar filters for:

- restaurant type
- inspection score range
- ZIP code selection

The dashboard also includes KPI cards showing:

- total inspections
- average score
- number of ZIP codes represented
- average median income

## Visualizations and Analyses

### 1. Income vs. Score

Chart type:

- Scatter plot with an OLS trendline

What it shows:

- ZIP code median household income on the x-axis
- average inspection score on the y-axis
- bubble size representing inspection frequency
- color encoding average score

Purpose:

- tests whether wealthier neighborhoods have better inspection outcomes
- evaluates the original thesis about income and food safety

Statistical method used:

- Pearson correlation

### 2. Frequency vs. Score

Chart type:

- Scatter plot with an OLS trendline

What it shows:

- inspection frequency by ZIP code on the x-axis
- average inspection score on the y-axis
- bubble size representing household count
- color encoding average score

Purpose:

- checks whether areas that are inspected more often tend to have lower scores
- explores whether inspection frequency is a proxy for prior issues or establishment density

Statistical method used:

- Pearson correlation

### 3. Scores Over Time

Chart types:

- Multi-line time series chart by ZIP code
- ZIP code choropleth map of average inspection scores

What it shows:

- year-over-year changes in average inspection scores by ZIP code
- selected highlight ZIPs for easier comparison
- citywide geographic score patterns in 2024

Purpose:

- identifies whether scores are improving or declining over time
- highlights stable versus volatile ZIP codes
- shows spatial variation in inspection outcomes across Austin

### 4. Income by Race

Chart types:

- Scatter plot of median household income by ZIP code and race/ethnicity
- ZIP code choropleth map of median income

What it shows:

- how median household income differs across race/ethnicity groups within each ZIP code
- geographic distribution of neighborhood wealth across Austin

Purpose:

- examines structural income disparities across racial and ethnic groups
- adds deeper socioeconomic context beyond the original income-only thesis

### 5. Restaurant Composition

Chart types:

- Stacked bar chart of local vs. chain restaurant counts by ZIP code
- ZIP code choropleth map showing local-minus-chain composition

What it shows:

- how many local and chain restaurants appear in each ZIP code
- where local restaurants dominate versus where chains dominate

Purpose:

- maps the distribution of restaurant business types across Austin
- helps explain whether neighborhood composition may affect inspection patterns

### 6. Local vs. Chain

Chart types:

- Box plot of score distributions by restaurant type
- Scatter plot of ZIP code income versus average score, colored by local/chain dominance

What it shows:

- score distribution differences between local and chain restaurants
- whether neighborhood income relates differently to scores for local and chain restaurants
- ZIP-level establishment dominance patterns

Statistical methods used:

- Welch t-test for local vs. chain mean score comparison
- Pearson correlation for income vs. score gap analysis
- OLS trendline on ZIP-level scatter plots

### 7. 78705 Restaurant Map

Chart type:

- Mapbox scatter map of restaurants in ZIP code 78705

What it shows:

- restaurant locations near UT Austin
- restaurant type coloring
- inspection score encoded by marker size

Purpose:

- provides a localized neighborhood spotlight
- makes the data more tangible by showing restaurant concentration in one ZIP code

## Main Findings Summarized in the Dashboard

The dashboard is designed around the idea that restaurant inspection outcomes are not strongly explained by neighborhood income alone. Across the explored views, the analysis suggests:

- the relationship between median income and inspection scores is weak
- inspection frequency is not strongly associated with lower scores
- year-over-year inspection patterns vary by ZIP code, but most areas remain relatively stable
- race/ethnicity is associated with income disparities across ZIP codes
- local and chain restaurants show different spatial patterns and slightly different score distributions
- the 78705 area offers a useful case study for a concentrated restaurant environment near UT Austin

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch the dashboard:

```bash
streamlit run final-dashboard.py
```

If you prefer one of the alternate versions included in the folder, you can also run `app.py` or `final.py` depending on which dashboard variant you want to view.

## Project Structure

- final-dashboard.py - primary Streamlit dashboard
- app.py - alternate dashboard version
- final.py - alternate dashboard version
- 2024_Food_Establishment_Inspection_Scores_Edited.csv - inspection dataset
- Census_Data.csv - Census ACS dataset
- requirements.txt - Python dependencies

## Notes

- The dashboard uses Plotly for interactive visualizations and Streamlit for the user interface.
- Some ZIP code visuals use geocoding or geographic mapping helpers, with fallback behavior when coordinates are unavailable.
- The conclusions shown in the app are written to match the statistical outputs computed from the filtered data in the dashboard.
