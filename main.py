from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from aux_1 import (
    AQI_CATEGORY_COLORS,
    AQI_CATEGORY_DESCRIPTIONS,
    AQI_CATEGORY_ORDER,
    POLLUTANT_VALUE_COLS,
    bootstrap_mean_aqi_ci,
    category_counts,
    clean_air_quality_data,
    country_summary,
    get_summary_metrics,
    load_raw_data,
    predict_aqi_category,
    spearman_correlation,
    train_aqi_classifier,
)


st.set_page_config(
    page_title="Global Air Pollution AQI Analysis",
    page_icon="AQI",
    layout="wide",
)

sns.set_theme(style="whitegrid")


@st.cache_data
def get_data() -> pd.DataFrame:
    return clean_air_quality_data(load_raw_data())


@st.cache_resource
def get_model(df: pd.DataFrame) -> dict[str, object]:
    return train_aqi_classifier(df)


def draw_category_distribution(df: pd.DataFrame) -> plt.Figure:
    counts = category_counts(df)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [AQI_CATEGORY_COLORS[category] for category in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_xlabel("")
    ax.set_ylabel("Number of observations")
    ax.set_title("AQI category distribution")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def draw_pollutant_scatter(df: pd.DataFrame, pollutant: str) -> plt.Figure:
    sample = df.sample(min(len(df), 2500), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.scatterplot(
        data=sample,
        x=pollutant,
        y="AQI Value",
        hue="AQI Category",
        hue_order=AQI_CATEGORY_ORDER,
        s=18,
        alpha=0.55,
        ax=ax,
    )
    ax.set_title(f"Overall AQI vs. {pollutant}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout()
    return fig


def draw_confusion_matrix(model_info: dict[str, object]) -> plt.Figure:
    cm = model_info["confusion_matrix"]
    labels = model_info["labels"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    ax.set_title("Random Forest confusion matrix")
    fig.tight_layout()
    return fig


df = get_data()
model_info = get_model(df)

st.title("Global Air Pollution AQI Analysis and Forecasting")
st.caption("Interactive analysis and AQI category prediction using the Kaggle Global Air Pollution Dataset.")

with st.sidebar:
    st.header("Filters")
    countries = ["All countries"] + sorted(df["Country"].unique().tolist())
    selected_country = st.selectbox("Country", countries)
    selected_categories = st.multiselect(
        "AQI categories",
        AQI_CATEGORY_ORDER,
        default=AQI_CATEGORY_ORDER,
    )

filtered_df = df.copy()
if selected_country != "All countries":
    filtered_df = filtered_df[filtered_df["Country"] == selected_country]
if selected_categories:
    filtered_df = filtered_df[filtered_df["AQI Category"].isin(selected_categories)]

if filtered_df.empty:
    st.warning("No records match the current filters. Select at least one AQI category or choose a broader country filter.")
    st.stop()

overview_tab, country_tab, statistics_tab, model_tab = st.tabs(
    ["Overview", "Country analysis", "Statistics", "Prediction model"]
)

with overview_tab:
    metrics = get_summary_metrics(filtered_df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{metrics['rows']:,}")
    col2.metric("Countries", f"{metrics['countries']:,}")
    col3.metric("Mean AQI", f"{metrics['mean_aqi']:.1f}")
    col4.metric("Most common class", metrics["most_common_category"])

    chart_col, table_col = st.columns([1.3, 1])
    with chart_col:
        st.pyplot(draw_category_distribution(filtered_df), use_container_width=True)
    with table_col:
        st.subheader("Filtered data sample")
        st.dataframe(filtered_df.head(25), use_container_width=True, hide_index=True)

with country_tab:
    st.subheader("Countries with the highest average AQI")
    min_records = st.slider("Minimum observations per country", 5, 200, 30, step=5)
    country_table = country_summary(filtered_df, min_records=min_records)

    top_country_table = country_table.head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top_country_table, y="Country", x="mean_aqi", color="#de6b48", ax=ax)
    ax.set_xlabel("Mean AQI")
    ax.set_ylabel("")
    ax.set_title("Highest average AQI by country")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.dataframe(country_table, use_container_width=True, hide_index=True)

with statistics_tab:
    st.subheader("Pollutant relationships")
    corr_table = spearman_correlation(filtered_df)
    ci_low, ci_high = bootstrap_mean_aqi_ci(filtered_df)

    stat_col1, stat_col2 = st.columns([1, 1])
    with stat_col1:
        st.metric("Bootstrap 95% CI for mean AQI", f"{ci_low:.1f} to {ci_high:.1f}")
        st.dataframe(corr_table, use_container_width=True, hide_index=True)
    with stat_col2:
        pollutant = st.selectbox("Pollutant AQI value", POLLUTANT_VALUE_COLS, index=3)
        st.pyplot(draw_pollutant_scatter(filtered_df, pollutant), use_container_width=True)

with model_tab:
    st.subheader("AQI category classifier")
    model_col1, model_col2 = st.columns([1, 1])

    with model_col1:
        st.metric("Held-out test accuracy", f"{model_info['accuracy']:.3f}")
        st.metric("Test observations", f"{model_info['test_size']:,}")
        report_df = pd.DataFrame(model_info["classification_report"]).T
        st.dataframe(report_df.round(3), use_container_width=True)

    with model_col2:
        st.pyplot(draw_confusion_matrix(model_info), use_container_width=True)

    st.divider()
    st.subheader("Try a prediction")
    input_col1, input_col2, input_col3 = st.columns(3)
    with input_col1:
        country = st.selectbox("Prediction country", sorted(df["Country"].unique().tolist()))
        co_aqi = st.slider("CO AQI Value", 0, 150, 1)
    with input_col2:
        ozone_aqi = st.slider("Ozone AQI Value", 0, 250, 35)
        no2_aqi = st.slider("NO2 AQI Value", 0, 100, 3)
    with input_col3:
        pm25_aqi = st.slider("PM2.5 AQI Value", 0, 500, 60)

    prediction, probability_df = predict_aqi_category(
        model_info["pipeline"],
        country=country,
        co_aqi=co_aqi,
        ozone_aqi=ozone_aqi,
        no2_aqi=no2_aqi,
        pm25_aqi=pm25_aqi,
    )
    st.success(f"Predicted AQI category: {prediction}")
    st.caption(AQI_CATEGORY_DESCRIPTIONS.get(prediction, ""))
    st.dataframe(probability_df, use_container_width=True, hide_index=True)
