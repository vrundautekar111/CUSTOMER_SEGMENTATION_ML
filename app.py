

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

MODEL_PATH = "models/customer_segmentation_model.pkl"
DATA_PATH = "segmented_customers.csv"

st.set_page_config(
    page_title="Customer Segmentation ML",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Segmentation Using Machine Learning")
st.markdown(
    "Analyze customer purchasing behavior using "
    "**K-Means Clustering**."
)

# Check that the model and training output exist.
if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
    st.warning("The trained model or customer data is missing.")
    st.code("python train_model.py")
    st.stop()

try:
    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
except Exception as error:
    st.error(f"Could not load project files: {error}")
    st.info("Run python train_model.py to regenerate the files.")
    st.stop()

if df.empty or "Cluster" not in df.columns:
    st.error("The customer dataset is empty or has no cluster labels.")
    st.info("Run python train_model.py again.")
    st.stop()

features = bundle["features"]
model = bundle["model"]
scaler = bundle["scaler"]

# Sidebar filters.
st.sidebar.title("Dashboard Filters")

if "Gender" in df.columns:
    gender_options = sorted(df["Gender"].dropna().unique().tolist())
    selected_genders = st.sidebar.multiselect(
        "Filter by gender",
        options=gender_options,
        default=gender_options,
    )

    filtered = df[df["Gender"].isin(selected_genders)].copy()
else:
    filtered = df.copy()

if filtered.empty:
    st.warning("No customers match your selected filters.")
    st.stop()

# KPI cards.
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{len(filtered):,}")
col2.metric("Customer Segments", filtered["Cluster"].nunique())
col3.metric(
    "Average Spending Score",
    f"{filtered['Spending Score (1-100)'].mean():.1f}/100",
)

st.divider()

overview_tab, segments_tab, prediction_tab = st.tabs([
    "📈 Overview",
    "👥 Customer Segments",
    "🔮 Predict Customer Segment",
])

# Overview tab.
with overview_tab:
    st.subheader("Customer Segmentation Chart")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.scatterplot(
        data=filtered,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="tab10",
        s=80,
        ax=ax,
    )

    ax.set_title("Annual Income vs Spending Score")
    ax.set_xlabel("Annual Income (thousands of dollars)")
    ax.set_ylabel("Spending Score (1–100)")
    st.pyplot(fig)
    plt.close(fig)

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Age Distribution")

        if "Age" in filtered.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(
                data=filtered,
                x="Age",
                bins=15,
                kde=True,
                ax=ax,
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Age data is not available.")

    with right:
        st.subheader("Annual Income Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            data=filtered,
            x="Annual Income (k$)",
            bins=15,
            kde=True,
            ax=ax,
        )
        st.pyplot(fig)
        plt.close(fig)

# Customer segments tab.
with segments_tab:
    st.subheader("Customer Segment Summary")

    profiles = (
        filtered.groupby("Cluster")
        .agg(
            Customers=("Cluster", "size"),
            Average_Income=("Annual Income (k$)", "mean"),
            Average_Spending=("Spending Score (1-100)", "mean"),
        )
        .round(2)
        .reset_index()
    )

    st.dataframe(profiles, use_container_width=True)

    st.subheader("Customer Records")
    st.dataframe(filtered, use_container_width=True)

    st.download_button(
        label="⬇️ Download Customer Data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="segmented_customers.csv",
        mime="text/csv",
    )

# Prediction tab.
with prediction_tab:
    st.subheader("Predict a New Customer's Segment")
    st.write(
        "Enter annual income and spending score to identify "
        "the closest learned customer cluster."
    )

    with st.form("prediction_form"):
        income = st.number_input(
            "Annual Income (in thousands of dollars)",
            min_value=0.0,
            max_value=1000.0,
            value=50.0,
            step=1.0,
        )

        spending = st.slider(
            "Spending Score",
            min_value=1,
            max_value=100,
            value=50,
        )

        predict_button = st.form_submit_button("Predict Segment")

    if predict_button:
        new_customer = pd.DataFrame(
            [[income, spending]],
            columns=features,
        )

        scaled_customer = scaler.transform(new_customer)
        cluster = int(model.predict(scaled_customer)[0])

        st.success(f"Predicted Customer Segment: Cluster {cluster}")

        # Find the matching training-data profile.
        profiles = bundle["profiles"]

        if cluster in profiles.index:
            profile = profiles.loc[cluster]

            st.metric(
                "Customers in this Segment",
                int(profile["Customers"]),
            )

            st.metric(
                "Average Annual Income",
                f"${profile['Avg_Income']:.1f}k",
            )

            st.metric(
                "Average Spending Score",
                f"{profile['Avg_Spending']:.1f}/100",
            )

        st.caption(
            "Cluster numbers are algorithm-generated labels. "
            "Use the segment averages and chart to interpret them."
        )

st.divider()
st.caption("Customer Segmentation ML Project | Python + Scikit-learn + Streamlit")