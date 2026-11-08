
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from preprocessing import (
    validate_merchant_data,
    build_behavior_features,
    calculate_revenue_exposure
)


# ==========================================
# MODEL FILE LOCATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR /
    "models" /
    "merchant_intelligence_models.joblib"
)


# ==========================================
# FUNCTION 1: LOAD TRAINED MODELS
# ==========================================

def load_model_bundle(model_path=None):

    path = Path(model_path) if model_path else MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Trained model file not found: {path}"
        )

    models = joblib.load(path)

    required_objects = [
        "imputer",
        "scaler",
        "kmeans",
        "isolation_forest",
        "feature_names",
        "cluster_names"
    ]

    missing = [
        item for item in required_objects
        if item not in models
    ]

    if missing:
        raise ValueError(
            f"Model bundle is incomplete: {missing}"
        )

    return models


# ==========================================
# FUNCTION 2: GENERATE MERCHANT PREDICTIONS
# ==========================================

def predict_merchants(input_data, model_bundle=None):

    # Load saved models if not already provided
    if model_bundle is None:
        model_bundle = load_model_bundle()

    # Validate original merchant data
    data = validate_merchant_data(input_data)

    # Generate behavioural features
    features = build_behavior_features(data)

    # Match training feature order
    expected_features = model_bundle["feature_names"]

    if list(features.columns) != expected_features:
        raise ValueError(
            "Input feature structure does not match training."
        )

    # Apply fitted preprocessing objects
    imputed_features = (
        model_bundle["imputer"].transform(features)
    )

    scaled_features = (
        model_bundle["scaler"].transform(imputed_features)
    )

    # ======================================
    # K-MEANS PREDICTION
    # ======================================

    cluster_labels = (
        model_bundle["kmeans"].predict(scaled_features)
    )

    cluster_names = model_bundle["cluster_names"]

    merchant_segments = [
        cluster_names[int(label)]
        for label in cluster_labels
    ]

    # ======================================
    # ISOLATION FOREST PREDICTION
    # ======================================

    anomaly_labels = (
        model_bundle["isolation_forest"]
        .predict(scaled_features)
    )

    anomaly_scores = (
        -model_bundle["isolation_forest"]
        .decision_function(scaled_features)
    )

    anomaly_classification = np.where(
        anomaly_labels == -1,
        "Anomalous",
        "Normal"
    )

    # ======================================
    # REVENUE CALCULATION
    # ======================================

    monthly_revenue, annual_revenue = (
        calculate_revenue_exposure(data)
    )

    # ======================================
    # CONSOLIDATED MERCHANT RESULTS
    # ======================================

    results = pd.DataFrame({

        "merchant_id": data["merchant_id"],

        "business_name": data["business_name"],

        "segment": data["segment"] if "segment" in data else np.nan,

        "industry": data["industry"] if "industry" in data else np.nan,

        "latest_monthly_tpv_lakh":
            data["tpv_m6_inr_lakh"],

        "monthly_net_revenue_lakh":
            monthly_revenue,

        "annual_revenue_exposure_lakh":
            annual_revenue,

        "merchant_cluster":
            cluster_labels,

        "merchant_segment":
            merchant_segments,

        "anomaly_classification":
            anomaly_classification,

        "anomaly_score":
            anomaly_scores,

        "tpv_trend_pct":
            features["tpv_trend_pct"],

        "tpv_volatility_pct":
            features["tpv_volatility_pct"],

        "payment_success_change_pp":
            features["payment_success_change_pp"],

        "latest_payment_success_pct":
            features["latest_payment_success_pct"],

        "support_tickets":
            features["support_tickets"],

        "unresolved_tickets":
            features["unresolved_tickets"],

        "csat":
            features["csat"],

        "products_used":
            features["products_used"],

        "logins_last_30d":
            features["logins_last_30d"],

        "renewal_days":
            features["renewal_days"],

        "tenure_months":
            features["tenure_months"]

    })

    # ======================================
    # VALUE-FIRST MERCHANT RANKING
    # ======================================

    results = results.sort_values(
        "annual_revenue_exposure_lakh",
        ascending=False
    ).reset_index(drop=True)

    results["value_rank"] = np.arange(
        1,
        len(results) + 1
    )

    return results
