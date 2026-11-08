
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import gradio as gr

from preprocessing import REQUIRED_COLUMNS
from ml_models import load_model_bundle, predict_merchants
from csm_recommendations import recommend_csm_actions


# =====================================================
# 1. LOAD TRAINED MODELS ON APPLICATION STARTUP
# =====================================================

MODEL_BUNDLE = load_model_bundle()

OUTREACH_LIMIT = 15


# =====================================================
# 2. READ CSV OR EXCEL FILE
# =====================================================

def read_merchant_file(file_path):

    if not file_path:
        raise ValueError("Please upload a CSV or Excel file.")

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".csv":

        data = pd.read_csv(path)

    elif extension == ".xlsx":

        excel_file = pd.ExcelFile(path)

        # Prefer the Problem 2 merchant worksheet
        if "P2_Merchants" in excel_file.sheet_names:

            data = pd.read_excel(
                path,
                sheet_name="P2_Merchants"
            )

        else:

            # Otherwise find a sheet containing the required fields
            data = None

            for sheet in excel_file.sheet_names:

                candidate = pd.read_excel(
                    path,
                    sheet_name=sheet
                )

                if set(REQUIRED_COLUMNS).issubset(
                    candidate.columns
                ):
                    data = candidate
                    break

            if data is None:
                raise ValueError(
                    "No worksheet contains the required merchant columns."
                )

    else:
        raise ValueError(
            "Unsupported file type. Upload CSV or XLSX."
        )

    return data


# =====================================================
# 3. PREPARE EXCEL-SAFE DATAFRAMES
# =====================================================

def make_excel_safe(dataframe):

    safe_df = dataframe.copy()

    for column in safe_df.columns:

        if safe_df[column].apply(
            lambda x: isinstance(x, (list, tuple, dict))
        ).any():

            safe_df[column] = safe_df[column].apply(
                lambda x: (
                    ", ".join(map(str, x))
                    if isinstance(x, (list, tuple))
                    else str(x)
                    if isinstance(x, dict)
                    else x
                )
            )

    return safe_df


# =====================================================
# 4. MAIN APPLICATION ANALYSIS FUNCTION
# =====================================================

def analyze_uploaded_file(file_path):

    try:

        # Read uploaded data
        original_data = read_merchant_file(file_path)

        # Generate ML predictions
        merchant_predictions = predict_merchants(
            original_data,
            MODEL_BUNDLE
        )

        # Generate CSM recommendations and priority queues
        (
            all_actions,
            priority_review,
            outreach_list,
            monitor_list,
            manual_review_list,
            additional_outreach,
            summary
        ) = recommend_csm_actions(
            merchant_results=merchant_predictions,
            original_data=original_data,
            outreach_limit=OUTREACH_LIMIT
        )

        # ---------------------------------------------
        # SELECT DASHBOARD TABLE COLUMNS
        # ---------------------------------------------

        priority_columns = [
            "value_rank",
            "merchant_id",
            "business_name",
            "annual_revenue_exposure_lakh",
            "merchant_segment",
            "anomaly_classification",
            "primary_playbook",
            "manual_review_required",
            "recommendation_reason"
        ]

        outreach_columns = [
            "value_rank",
            "merchant_id",
            "business_name",
            "annual_revenue_exposure_lakh",
            "merchant_segment",
            "primary_playbook",
            "additional_playbooks",
            "recommendation_reason"
        ]

        manual_columns = [
            "value_rank",
            "merchant_id",
            "business_name",
            "annual_revenue_exposure_lakh",
            "merchant_segment",
            "anomaly_classification",
            "manual_review_reason"
        ]

        monitor_columns = [
            "value_rank",
            "merchant_id",
            "business_name",
            "annual_revenue_exposure_lakh",
            "merchant_segment",
            "recommendation_reason"
        ]

        all_columns = [
            "value_rank",
            "merchant_id",
            "business_name",
            "segment",
            "industry",
            "annual_revenue_exposure_lakh",
            "merchant_segment",
            "anomaly_classification",
            "anomaly_score",
            "primary_playbook",
            "additional_playbooks",
            "manual_review_required",
            "proactive_outreach_recommended"
        ]

        priority_display = (
            priority_review[priority_columns]
            .round(3)
        )

        outreach_display = (
            outreach_list[outreach_columns]
            .round(3)
        )

        manual_display = (
            manual_review_list[manual_columns]
            .round(3)
        )

        monitor_display = (
            monitor_list[monitor_columns]
            .round(3)
        )

        all_display = (
            all_actions[all_columns]
            .round(3)
        )

        # ---------------------------------------------
        # PLAYBOOK SUMMARY
        # ---------------------------------------------

        playbook_summary = (
            all_actions["primary_playbook"]
            .value_counts()
            .rename_axis("Playbook")
            .reset_index(name="Merchant Count")
        )

        # ---------------------------------------------
        # CREATE DASHBOARD SUMMARY
        # ---------------------------------------------

        total_merchants = summary["total_merchants"]

        anomaly_count = (
            all_actions["anomaly_classification"]
            == "Anomalous"
        ).sum()

        portfolio_exposure = (
            summary["portfolio_annual_exposure_lakh"]
        )

        priority_exposure = (
            summary["priority_review_exposure_lakh"]
        )

        exposure_share = (
            priority_exposure / portfolio_exposure * 100
            if portfolio_exposure > 0 else 0
        )

        dashboard_summary = f"""
# Merchant Churn Intelligence

<small>Analysis completed successfully.</small>

## Portfolio overview

| Metric | Result |
|---|---:|
| Merchants analysed | **{total_merchants}** |
| Behavioural anomalies | **{anomaly_count}** |
| Annualised portfolio revenue exposure | **₹{portfolio_exposure:,.2f} lakh** |
| Priority review merchants | **{summary['priority_review_merchants']}** |
| Proactive outreach recommendations | **{summary['selected_outreach_merchants']}** |
| Manual review cases | **{summary['manual_review_merchants']}** |
| Annualised exposure represented by top 15 | **₹{priority_exposure:,.2f} lakh** |
| Top-15 share of portfolio exposure | **{exposure_share:.2f}%** |

## Merchant action overview

- Priority review capacity: {summary['priority_review_capacity']}
- Proactive outreach capacity: {summary['proactive_outreach_capacity']}
- Additional outreach beyond the top-15 value ranking: {summary['additional_outreach_outside_top_15']}
- Monitor-only merchants within the priority review queue: {summary['monitor_only_in_top_15']}

## Interpretation

The application uses behavioural clustering and anomaly detection to identify unusual merchant patterns. Revenue exposure is used for commercial prioritisation.

**These results are not validated churn probabilities.** Actual supervised churn prediction and ROC-AUC evaluation require historical churn outcome labels.
"""

        # ---------------------------------------------
        # CREATE DOWNLOADABLE EXCEL REPORT
        # ---------------------------------------------

        output_directory = "outputs"
        os.makedirs(output_directory, exist_ok=True)

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        report_path = os.path.join(
            output_directory,
            f"Merchant_Intelligence_Report_{timestamp}.xlsx"
        )

        summary_table = pd.DataFrame(
            list(summary.items()),
            columns=["Metric", "Value"]
        )

        with pd.ExcelWriter(
            report_path,
            engine="openpyxl"
        ) as writer:

            summary_table.to_excel(
                writer,
                sheet_name="Dashboard Summary",
                index=False
            )

            make_excel_safe(priority_review).to_excel(
                writer,
                sheet_name="Priority Review",
                index=False
            )

            make_excel_safe(outreach_list).to_excel(
                writer,
                sheet_name="Proactive Outreach",
                index=False
            )

            make_excel_safe(manual_review_list).to_excel(
                writer,
                sheet_name="Manual Review",
                index=False
            )

            make_excel_safe(monitor_list).to_excel(
                writer,
                sheet_name="Monitor Only",
                index=False
            )

            make_excel_safe(all_actions).to_excel(
                writer,
                sheet_name="All Merchants",
                index=False
            )

            playbook_summary.to_excel(
                writer,
                sheet_name="Playbook Summary",
                index=False
            )

        # ---------------------------------------------
        # RETURN RESULTS TO GRADIO
        # ---------------------------------------------

        return (
            dashboard_summary,
            priority_display,
            outreach_display,
            manual_display,
            monitor_display,
            all_display,
            playbook_summary,
            report_path
        )

    except Exception as error:

        raise gr.Error(
            f"Merchant analysis failed: {str(error)}"
        )


# =====================================================
# 5. BUILD GRADIO USER INTERFACE
# =====================================================

def build_app():

    with gr.Blocks(
        title="Merchant Churn Intelligence"
    ) as demo:

        gr.Markdown("""
        # Merchant Churn Intelligence

        ### AI-powered merchant behaviour analysis and retention decision support

        Upload your original merchant CSV or Excel file to generate
        merchant segments, anomaly insights, value rankings and CSM recommendations.
        """)

        with gr.Row():

            file_input = gr.File(
                label="Upload Merchant Dataset",
                file_types=[".csv", ".xlsx"],
                file_count="single",
                type="filepath"
            )

            analyze_button = gr.Button(
                "Analyse Merchants",
                variant="primary"
            )

        summary_output = gr.Markdown(
            label="Portfolio Dashboard"
        )

        with gr.Tabs():

            with gr.Tab("Top 15 Priority Review"):

                gr.Markdown(
                    "Highest-value merchants requiring priority review."
                )

                priority_output = gr.Dataframe(
                    label="Priority Review",
                    interactive=False,
                    wrap=True
                )

            with gr.Tab("Proactive Outreach"):

                gr.Markdown(
                    "Up to 15 merchants selected by annualised revenue exposure "
                    "among those with an outreach recommendation."
                )

                outreach_output = gr.Dataframe(
                    label="Outreach Recommendations",
                    interactive=False,
                    wrap=True
                )

            with gr.Tab("Manual Review"):

                gr.Markdown(
                    "Unusual or deteriorating merchant cases requiring analyst investigation."
                )

                manual_output = gr.Dataframe(
                    label="Manual Review Queue",
                    interactive=False,
                    wrap=True
                )

            with gr.Tab("Monitor Only"):

                monitor_output = gr.Dataframe(
                    label="Monitor-Only Merchants",
                    interactive=False,
                    wrap=True
                )

            with gr.Tab("All Merchants"):

                all_output = gr.Dataframe(
                    label="Complete Merchant Results",
                    interactive=False,
                    wrap=True
                )

            with gr.Tab("Playbook Summary"):

                playbook_output = gr.Dataframe(
                    label="CSM Playbook Distribution",
                    interactive=False,
                    wrap=True
                )

        download_output = gr.File(
            label="Download Complete Excel Report"
        )

        analyze_button.click(
            fn=analyze_uploaded_file,
            inputs=[file_input],
            outputs=[
                summary_output,
                priority_output,
                outreach_output,
                manual_output,
                monitor_output,
                all_output,
                playbook_output,
                download_output
            ],
            api_name="analyse_merchants"
        )

        gr.Markdown("""
        ---
        **Model information:** The system uses unsupervised behavioural
        clustering and anomaly detection. It does not provide validated
        churn probabilities because historical churn labels are not available.
        """)

    return demo


# Build the application
demo = build_app()


if __name__ == "__main__":
    demo.launch()
