
import pandas as pd
import numpy as np


# ==========================================
# REQUIRED ORIGINAL MERCHANT COLUMNS
# ==========================================

REQUIRED_COLUMNS = [
    "merchant_id",
    "business_name",
    "tenure_months",
    "tpv_m1_inr_lakh",
    "tpv_m2_inr_lakh",
    "tpv_m3_inr_lakh",
    "tpv_m4_inr_lakh",
    "tpv_m5_inr_lakh",
    "tpv_m6_inr_lakh",
    "success_rate_m1_m3_pct",
    "success_rate_m4_m6_pct",
    "support_tickets_last_90d",
    "unresolved_tickets",
    "last_csat_1_to_5",
    "products_used",
    "logins_last_30d",
    "days_to_contract_renewal"
]


# ==========================================
# NUMERICAL INPUT COLUMNS
# ==========================================

NUMERIC_COLUMNS = [
    "tenure_months",
    "tpv_m1_inr_lakh",
    "tpv_m2_inr_lakh",
    "tpv_m3_inr_lakh",
    "tpv_m4_inr_lakh",
    "tpv_m5_inr_lakh",
    "tpv_m6_inr_lakh",
    "success_rate_m1_m3_pct",
    "success_rate_m4_m6_pct",
    "support_tickets_last_90d",
    "unresolved_tickets",
    "last_csat_1_to_5",
    "products_used",
    "logins_last_30d",
    "days_to_contract_renewal"
]


# ==========================================
# BEHAVIOURAL FEATURES USED BY OUR ML MODELS
# ==========================================

FEATURE_NAMES = [
    "tpv_trend_pct",
    "tpv_volatility_pct",
    "payment_success_change_pp",
    "latest_payment_success_pct",
    "support_tickets",
    "unresolved_tickets",
    "csat",
    "products_used",
    "logins_last_30d",
    "renewal_days",
    "tenure_months"
]


# ==========================================
# FUNCTION 1: VALIDATE INPUT DATA
# ==========================================

def validate_merchant_data(input_data):

    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if input_data.empty:
        raise ValueError("The uploaded dataset is empty.")

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in input_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data = input_data.copy().reset_index(drop=True)

    if data["merchant_id"].isna().any():
        raise ValueError("Merchant IDs cannot be missing.")

    if data["merchant_id"].duplicated().any():
        raise ValueError("Duplicate merchant IDs detected.")

    if data["business_name"].isna().any():
        raise ValueError("Business names cannot be missing.")

    # Convert original numeric columns
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    return data


# ==========================================
# FUNCTION 2: GENERATE BEHAVIOURAL FEATURES
# ==========================================

def build_behavior_features(data):

    # Average TPV across first three months
    early_tpv = data[
        [
            "tpv_m1_inr_lakh",
            "tpv_m2_inr_lakh",
            "tpv_m3_inr_lakh"
        ]
    ].mean(axis=1)

    # Average TPV across latest three months
    recent_tpv = data[
        [
            "tpv_m4_inr_lakh",
            "tpv_m5_inr_lakh",
            "tpv_m6_inr_lakh"
        ]
    ].mean(axis=1)

    tpv_columns = [
        f"tpv_m{i}_inr_lakh"
        for i in range(1, 7)
    ]

    features = pd.DataFrame(index=data.index)

    # TPV trend
    features["tpv_trend_pct"] = (
        (recent_tpv - early_tpv) /
        early_tpv.replace(0, np.nan)
    ) * 100

    # TPV volatility
    features["tpv_volatility_pct"] = (
        data[tpv_columns].std(axis=1, ddof=0) /
        data[tpv_columns].mean(axis=1).replace(0, np.nan)
    ) * 100

    # Payment performance change
    features["payment_success_change_pp"] = (
        data["success_rate_m4_m6_pct"] -
        data["success_rate_m1_m3_pct"]
    )

    features["latest_payment_success_pct"] = (
        data["success_rate_m4_m6_pct"]
    )

    # Support features
    features["support_tickets"] = (
        data["support_tickets_last_90d"]
    )

    features["unresolved_tickets"] = (
        data["unresolved_tickets"]
    )

    features["csat"] = (
        data["last_csat_1_to_5"]
    )

    # Engagement features
    features["products_used"] = (
        data["products_used"]
    )

    features["logins_last_30d"] = (
        data["logins_last_30d"]
    )

    # Commercial and relationship features
    features["renewal_days"] = (
        data["days_to_contract_renewal"]
    )

    features["tenure_months"] = (
        data["tenure_months"]
    )

    return features[FEATURE_NAMES]


# ==========================================
# FUNCTION 3: CALCULATE REVENUE EXPOSURE
# ==========================================

def calculate_revenue_exposure(data):

    # Net revenue assumption: 1% of TPV
    monthly_revenue = (
        data["tpv_m6_inr_lakh"] * 0.01
    )

    annual_revenue = monthly_revenue * 12

    return monthly_revenue, annual_revenue
