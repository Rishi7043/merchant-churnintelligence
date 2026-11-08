
import pandas as pd
import numpy as np


# ==========================================
# CONFIGURABLE BUSINESS THRESHOLDS
# ==========================================

PAYMENT_DROP_THRESHOLD_PP = -5
LOW_CSAT_THRESHOLD = 2
UNRESOLVED_TICKETS_THRESHOLD = 2
LOW_LOGIN_THRESHOLD = 10
LOW_PRODUCT_THRESHOLD = 2
RENEWAL_WINDOW_DAYS = 60
SIGNIFICANT_TPV_DECLINE = -10


PLAYBOOK_NAMES = {
    "PB-1": "Performance deep-dive",
    "PB-2": "Support recovery",
    "PB-3": "Commercial review",
    "PB-4": "Re-engagement and adoption",
    "PB-5": "Growth / expansion check-in",
    "PB-6": "Monitor only"
}


# ==========================================
# KEYWORD DETECTION
# ==========================================

COMPETITOR_KEYWORDS = [
    "competitor", "competitors", "switching",
    "switch to", "alternative provider",
    "other provider", "rival", "moving volume",
    "shift volume"
]

SEASONAL_KEYWORDS = [
    "seasonal", "seasonality", "festive",
    "off-season", "off season", "monsoon",
    "holiday demand"
]

GROWTH_KEYWORDS = [
    "growth", "growing", "expand", "expansion",
    "higher volume", "increase in volume",
    "new outlet", "new location"
]

NO_RESPONSE_KEYWORDS = [
    "no response", "not responding",
    "unresponsive", "not replied", "no reply"
]


def contains_keyword(text, keywords):

    text = str(text).lower()

    return any(
        word in text for word in keywords
    )


# ==========================================
# MAIN RECOMMENDATION ENGINE
# ==========================================

def recommend_csm_actions(
    merchant_results,
    original_data,
    outreach_limit=15
):

    if not isinstance(outreach_limit, int) or outreach_limit < 1:
        raise ValueError(
            "Outreach limit must be a positive integer."
        )

    required_results = [
        "merchant_id",
        "business_name",
        "annual_revenue_exposure_lakh",
        "merchant_segment",
        "anomaly_classification",
        "tpv_trend_pct",
        "payment_success_change_pp",
        "unresolved_tickets",
        "csat",
        "products_used",
        "logins_last_30d",
        "renewal_days"
    ]

    missing = [
        col for col in required_results
        if col not in merchant_results.columns
    ]

    if missing:
        raise ValueError(
            f"Missing prediction columns: {missing}"
        )

    results = merchant_results.copy()

    # --------------------------------------
    # 1. ATTACH ORIGINAL ACCOUNT NOTES
    # --------------------------------------

    if "merchant_id" not in original_data.columns:
        raise ValueError(
            "Original data must contain merchant_id."
        )

    notes_data = original_data[
        ["merchant_id"]
    ].copy()

    if "account_manager_note" in original_data.columns:

        notes_data["account_manager_note"] = (
            original_data["account_manager_note"]
            .fillna("")
            .astype(str)
        )

    else:
        notes_data["account_manager_note"] = ""

    if notes_data["merchant_id"].duplicated().any():
        raise ValueError(
            "Duplicate merchant IDs in original data."
        )

    results = results.merge(
        notes_data,
        on="merchant_id",
        how="left",
        validate="one_to_one"
    )

    results["account_manager_note"] = (
        results["account_manager_note"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------
    # 2. DETECT ACCOUNT NOTE CONTEXT
    # --------------------------------------

    results["competitor_mention_detected"] = (
        results["account_manager_note"].apply(
            lambda x: contains_keyword(
                x, COMPETITOR_KEYWORDS
            )
        )
    )

    results["seasonal_context_detected"] = (
        results["account_manager_note"].apply(
            lambda x: contains_keyword(
                x, SEASONAL_KEYWORDS
            )
        )
    )

    results["growth_note_detected"] = (
        results["account_manager_note"].apply(
            lambda x: contains_keyword(
                x, GROWTH_KEYWORDS
            )
        )
    )

    results["no_response_detected"] = (
        results["account_manager_note"].apply(
            lambda x: contains_keyword(
                x, NO_RESPONSE_KEYWORDS
            )
        )
    )

    # --------------------------------------
    # 3. CREATE OFFICIAL PLAYBOOK TRIGGERS
    # --------------------------------------

    # PB-1: Performance deep-dive
    results["PB1_trigger"] = (
        results["payment_success_change_pp"]
        <= PAYMENT_DROP_THRESHOLD_PP
    )

    # PB-2: Support recovery
    results["PB2_trigger"] = (
        (
            results["unresolved_tickets"]
            >= UNRESOLVED_TICKETS_THRESHOLD
        )
        |
        (
            results["csat"].notna()
            &
            (results["csat"] <= LOW_CSAT_THRESHOLD)
        )
    )

    # PB-3: Commercial review
    results["PB3_trigger"] = (
        results["competitor_mention_detected"]
        |
        (results["renewal_days"] <= RENEWAL_WINDOW_DAYS)
    )

    # PB-4: Re-engagement and adoption
    results["PB4_trigger"] = (
        (results["logins_last_30d"] <= LOW_LOGIN_THRESHOLD)
        |
        (results["products_used"] <= LOW_PRODUCT_THRESHOLD)
        |
        results["no_response_detected"]
    )

    # --------------------------------------
    # 4. PB-5: HEALTHY AND GROWING
    # --------------------------------------

    no_primary_risk = ~(
        results["PB1_trigger"]
        |
        results["PB2_trigger"]
        |
        results["PB3_trigger"]
        |
        results["PB4_trigger"]
    )

    healthy = (
        (results["merchant_segment"] == "Stable / Growing")
        &
        (results["anomaly_classification"] == "Normal")
    )

    growing = (
        (results["tpv_trend_pct"] > 0)
        |
        results["growth_note_detected"]
    )

    results["PB5_trigger"] = (
        no_primary_risk & healthy & growing
    )

    # --------------------------------------
    # 5. INTERNAL ANALYST REVIEW FLAGS
    # --------------------------------------

    unexplained_tpv_decline = (
        (results["tpv_trend_pct"] <= SIGNIFICANT_TPV_DECLINE)
        &
        ~results["seasonal_context_detected"]
        &
        ~results["PB1_trigger"]
        &
        ~results["PB2_trigger"]
    )

    unexplained_anomaly = (
        (results["anomaly_classification"] == "Anomalous")
        &
        ~results["PB1_trigger"]
        &
        ~results["PB2_trigger"]
        &
        ~results["PB3_trigger"]
        &
        ~results["PB4_trigger"]
        &
        ~results["seasonal_context_detected"]
    )

    results["manual_review_required"] = (
        unexplained_tpv_decline | unexplained_anomaly
    )

    def generate_review_reason(row):

        reasons = []

        if (
            row["tpv_trend_pct"] <= SIGNIFICANT_TPV_DECLINE
            and not row["seasonal_context_detected"]
            and not row["PB1_trigger"]
            and not row["PB2_trigger"]
        ):
            reasons.append(
                "Significant non-seasonal TPV decline; investigate merchant volume"
            )

        if (
            row["anomaly_classification"] == "Anomalous"
            and not row["PB1_trigger"]
            and not row["PB2_trigger"]
            and not row["PB3_trigger"]
            and not row["PB4_trigger"]
            and not row["seasonal_context_detected"]
        ):
            reasons.append(
                "Unusual behavioural pattern requires analyst verification"
            )

        return "; ".join(reasons)

    results["manual_review_reason"] = results.apply(
        generate_review_reason,
        axis=1
    )

    # --------------------------------------
    # 6. PB-6: MONITOR ONLY
    # --------------------------------------

    results["PB6_trigger"] = ~(
        results["PB1_trigger"]
        |
        results["PB2_trigger"]
        |
        results["PB3_trigger"]
        |
        results["PB4_trigger"]
        |
        results["PB5_trigger"]
    )

    # --------------------------------------
    # 7. PRIMARY AND ADDITIONAL PLAYBOOKS
    # --------------------------------------

    trigger_map = {
        "PB-1": "PB1_trigger",
        "PB-2": "PB2_trigger",
        "PB-3": "PB3_trigger",
        "PB-4": "PB4_trigger",
        "PB-5": "PB5_trigger",
        "PB-6": "PB6_trigger"
    }

    results["recommended_playbooks"] = results.apply(
        lambda row: [
            pb for pb, trigger in trigger_map.items()
            if bool(row[trigger])
        ],
        axis=1
    )

    results["primary_playbook_id"] = (
        results["recommended_playbooks"].apply(
            lambda x: x[0] if x else "PB-6"
        )
    )

    results["primary_playbook"] = (
        results["primary_playbook_id"].map(PLAYBOOK_NAMES)
    )

    results["additional_playbooks"] = (
        results["recommended_playbooks"].apply(
            lambda x: ", ".join(x[1:])
            if len(x) > 1 else "None"
        )
    )

    # --------------------------------------
    # 8. EXPLAIN RECOMMENDATIONS
    # --------------------------------------

    def explain_recommendation(row):

        reasons = []

        if row["PB1_trigger"]:
            reasons.append(
                "Payment success deterioration of at least 5 percentage points"
            )

        if row["PB2_trigger"]:

            if row["unresolved_tickets"] >= 2:
                reasons.append(
                    "Multiple unresolved support tickets"
                )

            if (
                pd.notna(row["csat"])
                and row["csat"] <= LOW_CSAT_THRESHOLD
            ):
                reasons.append("Low merchant CSAT")

        if row["PB3_trigger"]:

            if row["competitor_mention_detected"]:
                reasons.append("Competitor concern in account notes")

            if row["renewal_days"] <= RENEWAL_WINDOW_DAYS:
                reasons.append("Contract renewal within 60 days")

        if row["PB4_trigger"]:

            if row["logins_last_30d"] <= LOW_LOGIN_THRESHOLD:
                reasons.append("Low login activity")

            if row["products_used"] <= LOW_PRODUCT_THRESHOLD:
                reasons.append("Limited product adoption")

            if row["no_response_detected"]:
                reasons.append("No response to previous outreach")

        if row["PB5_trigger"]:
            reasons.append("Healthy merchant with growth indicators")

        if row["PB6_trigger"]:

            if row["seasonal_context_detected"]:
                reasons.append("Seasonal context detected")

            else:
                reasons.append("No direct CSM playbook trigger")

        if row["manual_review_required"]:
            reasons.append(
                row["manual_review_reason"]
            )

        return "; ".join(reasons)

    results["recommendation_reason"] = results.apply(
        explain_recommendation,
        axis=1
    )

    # --------------------------------------
    # 9. PROACTIVE OUTREACH ELIGIBILITY
    # --------------------------------------

    results["proactive_outreach_recommended"] = (
        results["primary_playbook_id"] != "PB-6"
    )

    # --------------------------------------
    # 10. VALUE-FIRST PRIORITY REVIEW
    # --------------------------------------

    results = results.sort_values(
        "annual_revenue_exposure_lakh",
        ascending=False
    ).reset_index(drop=True)

    results["value_rank"] = np.arange(
        1, len(results) + 1
    )

    results["selected_for_priority_review"] = (
        results["value_rank"] <= 15
    )

    priority_review = results.head(15).copy()

    # --------------------------------------
    # 11. SELECT UP TO 15 OUTREACH MERCHANTS
    # FROM THE ENTIRE PORTFOLIO
    # --------------------------------------

    outreach_candidates = results[
        results["proactive_outreach_recommended"]
    ].copy()

    outreach_candidates = outreach_candidates.head(
        outreach_limit
    )

    selected_ids = set(
        outreach_candidates["merchant_id"]
    )

    results["selected_for_outreach"] = (
        results["merchant_id"].isin(selected_ids)
    )

    outreach_list = results[
        results["selected_for_outreach"]
    ].copy()

    # Merchants reviewed but not selected for outreach
    monitor_list = priority_review[
        ~priority_review["proactive_outreach_recommended"]
        &
        ~priority_review["manual_review_required"]
    ].copy()

    # Separate internal review queue
    manual_review_list = results[
        results["manual_review_required"]
    ].copy()

    # Additional outreach outside the top-15 value review
    additional_outreach = outreach_list[
        outreach_list["value_rank"] > 15
    ].copy()

    # --------------------------------------
    # 12. BUSINESS SUMMARY
    # --------------------------------------

    portfolio_exposure = (
        results["annual_revenue_exposure_lakh"].sum()
    )

    priority_exposure = (
        priority_review["annual_revenue_exposure_lakh"].sum()
    )

    outreach_exposure = (
        outreach_list["annual_revenue_exposure_lakh"].sum()
    )

    summary = {

        "total_merchants": len(results),

        "priority_review_capacity": 15,

        "priority_review_merchants": len(priority_review),

        "proactive_outreach_capacity": outreach_limit,

        "selected_outreach_merchants": len(outreach_list),

        "additional_outreach_outside_top_15":
            len(additional_outreach),

        "monitor_only_in_top_15": len(monitor_list),

        "manual_review_merchants": len(manual_review_list),

        "portfolio_annual_exposure_lakh":
            portfolio_exposure,

        "priority_review_exposure_lakh":
            priority_exposure,

        "outreach_exposure_lakh":
            outreach_exposure
    }

    return (
        results,
        priority_review,
        outreach_list,
        monitor_list,
        manual_review_list,
        additional_outreach,
        summary
    )
