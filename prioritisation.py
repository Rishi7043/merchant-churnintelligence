
import pandas as pd


# ==========================================
# MERCHANT PRIORITISATION MODULE
# ==========================================

def prioritise_merchants(
    merchant_results,
    outreach_limit=15
):

    # 1. Validate input
    required_columns = [
        "merchant_id",
        "business_name",
        "annual_revenue_exposure_lakh",
        "merchant_segment",
        "anomaly_classification",
        "anomaly_score",
        "tpv_trend_pct",
        "payment_success_change_pp",
        "unresolved_tickets",
        "logins_last_30d"
    ]

    missing = [
        col for col in required_columns
        if col not in merchant_results.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if not isinstance(outreach_limit, int) or outreach_limit < 1:
        raise ValueError(
            "Outreach limit must be a positive integer."
        )

    results = merchant_results.copy()

    # 2. Sort merchants by annualised revenue exposure
    results = results.sort_values(
        "annual_revenue_exposure_lakh",
        ascending=False
    ).reset_index(drop=True)

    # 3. Assign commercial value rank
    results["value_rank"] = range(
        1, len(results) + 1
    )

    # 4. Identify the recommended business action
    def recommend_action(row):

        segment = row["merchant_segment"]
        anomaly = row["anomaly_classification"]

        if segment == "Payment Deterioration":
            return "Investigate payment performance"

        elif segment == "Support Challenged":
            return "Initiate support recovery"

        elif segment == "Low Adoption":
            return "Review merchant engagement and adoption"

        elif anomaly == "Anomalous":
            return "Investigate unusual merchant behaviour"

        elif (
            row["tpv_trend_pct"] > 0
            and row["payment_success_change_pp"] >= 0
        ):
            return "Proactive growth and relationship review"

        else:
            return "Routine monitoring"

    results["recommended_action"] = results.apply(
        recommend_action,
        axis=1
    )

    # 5. Identify why the merchant was selected
    def generate_reason(row):

        reasons = []

        if row["anomaly_classification"] == "Anomalous":
            reasons.append(
                "Unusual behavioural pattern detected"
            )

        if row["tpv_trend_pct"] <= -10:
            reasons.append(
                "Significant TPV deterioration"
            )

        if row["payment_success_change_pp"] <= -2:
            reasons.append(
                "Payment success deterioration"
            )

        if row["unresolved_tickets"] >= 2:
            reasons.append(
                "Multiple unresolved support tickets"
            )

        if row["logins_last_30d"] <= 10:
            reasons.append(
                "Low merchant login activity"
            )

        if not reasons:
            reasons.append(
                "High commercial value based on annualised exposure"
            )

        return "; ".join(reasons)

    results["recommendation_reason"] = results.apply(
        generate_reason,
        axis=1
    )

    # 6. Apply the monthly outreach limit
    results["outreach_selected"] = (
        results["value_rank"] <= outreach_limit
    )

    # 7. Create the final top-15 shortlist
    top_outreach = results[
        results["outreach_selected"]
    ].copy()

    # 8. Calculate portfolio exposure
    portfolio_exposure = (
        results["annual_revenue_exposure_lakh"].sum()
    )

    selected_exposure = (
        top_outreach["annual_revenue_exposure_lakh"].sum()
    )

    summary = {
        "total_merchants": len(results),

        "outreach_capacity": outreach_limit,

        "selected_merchants": len(top_outreach),

        "portfolio_annual_exposure_lakh":
            portfolio_exposure,

        "selected_annual_exposure_lakh":
            selected_exposure,

        "selected_exposure_share_pct": (
            selected_exposure / portfolio_exposure * 100
            if portfolio_exposure > 0 else 0
        )
    }

    return results, top_outreach, summary
