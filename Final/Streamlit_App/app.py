from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.sparse import csr_matrix, hstack


st.set_page_config(
    page_title="White Sturgeon Monitoring Explorer",
    page_icon="🐟",
    layout="wide"
)

st.title("White Sturgeon Monitoring Explorer")
st.write(
    "This app explores the locations and timing patterns associated with large mature "
    "White Sturgeon in the historical capture records. A large mature encounter is "
    "defined as fork length of at least 150 cm."
)

DATA_FOLDER = Path(__file__).parent

segment_df = pd.read_csv(DATA_FOLDER / "sturgeon_segment_summary.csv")
priority_df = pd.read_csv(DATA_FOLDER / "sturgeon_priority_locations.csv")
month_df = pd.read_csv(DATA_FOLDER / "sturgeon_month_summary.csv")
model_bundle = joblib.load(DATA_FOLDER / "sturgeon_logistic_model.joblib")

model = model_bundle["model"]
scaler = model_bundle["scaler"]
common_locations = model_bundle["common_locations"]
location_categories = model_bundle["location_categories"]
location_columns = model_bundle["location_columns"]
year_min = model_bundle["year_min"]
year_max = model_bundle["year_max"]


# Make one prediction using the same feature setup as the notebook.
def predict_probability(location, month, year):
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    numeric_df = pd.DataFrame({
        "capture_year": [year],
        "month_sin": [month_sin],
        "month_cos": [month_cos]
    })
    numeric_scaled = scaler.transform(numeric_df)

    if location not in common_locations:
        location = "Other / sparse segment"

    location_series = pd.Series(
        pd.Categorical([location], categories=location_categories)
    )
    location_dummy = pd.get_dummies(
        location_series,
        prefix="location",
        drop_first=True,
        dtype=int
    )
    location_dummy = location_dummy.reindex(columns=location_columns, fill_value=0)

    X = hstack([
        csr_matrix(numeric_scaled),
        csr_matrix(location_dummy.to_numpy())
    ]).tocsr()

    probability = model.predict_proba(X)[0, 1]
    return probability


st.sidebar.header("Explore the Results")
waterbody_options = ["All"] + sorted(segment_df["waterbody"].dropna().unique().tolist())
selected_waterbody = st.sidebar.selectbox("Waterbody", waterbody_options)
minimum_records = st.sidebar.slider(
    "Minimum size-complete encounters",
    min_value=100,
    max_value=1000,
    value=100,
    step=50
)

priority_tab, explore_tab, prediction_tab = st.tabs([
    "Monitoring Priorities",
    "Explore Locations",
    "Model Prediction"
])


with priority_tab:
    st.subheader("Candidate Monitoring Priorities")
    st.write(
        "These locations stay strong when I compare the EDA with the held-out logistic model. "
        "The ranking is based on recorded encounters, not survey effort."
    )

    top_priority = priority_df.iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Top candidate", top_priority["location"])
    col2.metric(
        "Held-out model probability",
        f"{top_priority['logistic_mean_probability_percent']:.1f}%"
    )
    col3.metric(
        "Held-out observed share",
        f"{top_priority['held_out_observed_large_percent']:.1f}%"
    )

    plot_priority = priority_df.head(10).sort_values(
        "logistic_mean_probability_percent",
        ascending=True
    )

    fig_priority = px.bar(
        plot_priority,
        x="logistic_mean_probability_percent",
        y="location",
        orientation="h",
        labels={
            "logistic_mean_probability_percent": "Mean model probability (%)",
            "location": "Location"
        },
        title="Held-Out Model Ranking for Candidate Locations",
        color_discrete_sequence=["#E45756"]
    )
    st.plotly_chart(fig_priority, use_container_width=True)

    priority_table = priority_df[[
        "location",
        "fork_complete",
        "large_share_percent",
        "held_out_encounters",
        "held_out_observed_large_percent",
        "logistic_mean_probability_percent"
    ]].head(12).copy()

    priority_table.columns = [
        "Location",
        "Size-complete encounters",
        "EDA large mature share (%)",
        "Held-out encounters",
        "Held-out observed share (%)",
        "Mean model probability (%)"
    ]
    st.dataframe(priority_table.round(1), use_container_width=True, hide_index=True)


with explore_tab:
    st.subheader("Explore the Historical Location Pattern")

    filtered = segment_df[
        segment_df["fork_complete"] >= minimum_records
    ].copy()

    if selected_waterbody != "All":
        filtered = filtered[filtered["waterbody"] == selected_waterbody]

    filtered = filtered.sort_values("large_share_percent", ascending=False).head(20)

    if len(filtered) == 0:
        st.warning("No locations match the current filters.")
    else:
        plot_segments = filtered.sort_values("large_share_percent", ascending=True)
        fig_segments = px.bar(
            plot_segments,
            x="large_share_percent",
            y="location",
            color="waterbody",
            orientation="h",
            hover_data=["fork_complete", "large_count", "median_fork_length"],
            labels={
                "large_share_percent": "Large mature share (%)",
                "location": "Location",
                "waterbody": "Waterbody"
            },
            title="Large Mature Share by Location",
            color_discrete_sequence=["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
        )
        st.plotly_chart(fig_segments, use_container_width=True)

    st.subheader("Timing Pattern")
    month_plot = month_df.rename(columns={
        "observed_large_percent": "Observed held-out share",
        "logistic_mean_probability_percent": "Mean logistic probability"
    })

    month_long = month_plot.melt(
        id_vars="capture_month",
        value_vars=["Observed held-out share", "Mean logistic probability"],
        var_name="Series",
        value_name="Percent"
    )

    fig_month = px.line(
        month_long,
        x="capture_month",
        y="Percent",
        color="Series",
        markers=True,
        labels={"capture_month": "Capture month", "Percent": "Large mature share / probability (%)"},
        title="Held-Out Large Mature Pattern by Month",
        color_discrete_map={
            "Observed held-out share": "#4C78A8",
            "Mean logistic probability": "#F58518"
        }
    )
    fig_month.update_xaxes(dtick=1)
    st.plotly_chart(fig_month, use_container_width=True)


with prediction_tab:
    st.subheader("Try the Logistic Regression Model")
    st.write(
        "Select a location, month, and year to see the model probability for the large mature class. "
        "This uses the same logistic regression feature setup as the final notebook."
    )

    display_locations = sorted(common_locations)
    selected_location = st.selectbox(
        "Waterbody and RKM segment",
        display_locations,
        format_func=lambda value: value.replace(" | ", " | RKM ")
    )
    selected_month = st.slider("Capture month", 1, 12, 8)
    selected_year = st.slider("Capture year", year_min, year_max, year_max)
    threshold = st.slider("Classification threshold", 0.30, 0.70, 0.50, 0.05)

    probability = predict_probability(
        selected_location,
        selected_month,
        selected_year
    )

    predicted_class = "Large mature" if probability >= threshold else "Below 150 cm"

    result1, result2 = st.columns(2)
    result1.metric("Model probability", f"{probability * 100:.1f}%")
    result2.metric("Classification", predicted_class)

    st.progress(float(probability))
    st.write(
        f"At a threshold of {threshold:.2f}, the model classifies this location and timing "
        f"profile as **{predicted_class}**."
    )

    st.info(
        "The probability is based on patterns inside historical successful capture records. "
        "It is not the probability of catching a large fish on a new survey and it is not an estimate of abundance."
    )


with st.expander("Important limitation"):
    st.write(
        "The dataset does not contain survey hours, unsuccessful sampling attempts, gear effort, or cost by location. "
        "Because of this, the app can show where large mature fish were more common among recorded encounters, "
        "but it cannot show which location produces the most large fish per unit of monitoring effort."
    )
