
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

DATA_PATH = "Mall_Customers.csv"
OUTPUT_PATH = "segmented_customers.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "customer_segmentation_model.pkl")

FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]


def generate_sample_data():
    """Create sample customer data for project testing."""
    rng = np.random.default_rng(42)

    groups = [
        (25, 25, 45),
        (25, 80, 40),
        (55, 50, 45),
        (85, 20, 35),
        (85, 80, 35),
    ]

    records = []
    customer_id = 1

    for income_center, spending_center, count in groups:
        incomes = np.clip(
            rng.normal(income_center, 9, count), 15, 140
        )
        scores = np.clip(
            rng.normal(spending_center, 10, count), 1, 100
        )
        ages = rng.integers(18, 71, count)

        for age, income, score in zip(ages, incomes, scores):
            records.append({
                "CustomerID": customer_id,
                "Gender": rng.choice(["Male", "Female"]),
                "Age": int(age),
                "Annual Income (k$)": round(float(income), 2),
                "Spending Score (1-100)": round(float(score), 2),
            })
            customer_id += 1

    df = pd.DataFrame(records)
    df.to_csv(DATA_PATH, index=False)
    print(f"Sample data created: {DATA_PATH}")
    return df


def load_customer_data():
    """Load valid CSV data or regenerate sample data when necessary."""
    if not os.path.exists(DATA_PATH):
        print("CSV file not found. Creating sample data...")
        return generate_sample_data()

    if os.path.getsize(DATA_PATH) == 0:
        print("CSV file is empty. Creating sample data...")
        return generate_sample_data()

    try:
        df = pd.read_csv(DATA_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError,
            UnicodeDecodeError) as error:
        print(f"CSV cannot be read ({error}). Creating sample data...")
        return generate_sample_data()

    if df.empty or len(df.columns) == 0:
        print("CSV has no usable rows. Creating sample data...")
        return generate_sample_data()

    missing = [column for column in FEATURES if column not in df.columns]

    if missing:
        print(f"Required columns missing: {missing}")
        print("Creating compatible sample data instead.")
        return generate_sample_data()

    # Remove rows without usable income or spending values.
    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=FEATURES).copy()

    if len(df) < 5:
        print("Fewer than 5 valid customer rows. Creating sample data.")
        return generate_sample_data()

    return df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_customer_data()
    X = df[FEATURES]

    # Scale features before clustering.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the K-Means model.
    model = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(X_scaled)

    # Evaluate cluster separation when possible.
    if df["Cluster"].nunique() > 1:
        score = silhouette_score(X_scaled, df["Cluster"])
        print(f"\nSilhouette score: {score:.3f}")
    else:
        print("\nSilhouette score unavailable.")

    # Summarize the clusters.
    profiles = (
        df.groupby("Cluster")
        .agg(
            Customers=("Cluster", "size"),
            Avg_Income=("Annual Income (k$)", "mean"),
            Avg_Spending=("Spending Score (1-100)", "mean"),
        )
        .round(2)
    )

    # Save model, scaler, features and cluster profiles together.
    bundle = {
        "model": model,
        "scaler": scaler,
        "features": FEATURES,
        "profiles": profiles,
    }

    joblib.dump(bundle, MODEL_PATH)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nCUSTOMER SEGMENTATION MODEL TRAINED SUCCESSFULLY")
    print("\nCluster Summary:")
    print(profiles)
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Segmented data saved: {OUTPUT_PATH}")
    print(f"Total customers processed: {len(df)}")


if __name__ == "__main__":
    main()