"""
Streamlit Web Dashboard for Loan Default Risk Prediction & Underwriting Portal.
Features interactive scoring, portfolio analytics, model explainability,
and MySQL decision audit persistence.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import (
    FIGURES_DIR,
    METRICS_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_PATH,
)
from src.predict import LoanRiskPredictor
from src.database import db_manager

# Conditional import for seaborn and streamlit
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_theme(style="whitegrid")
except ImportError:
    HAS_SEABORN = False

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def run_streamlit_dashboard():
    if not HAS_STREAMLIT:
        print("Streamlit is not installed in the current environment.")
        print("Please run: pip install streamlit seaborn")
        return

    st.set_page_config(
        page_title="CreditIQ | Loan Default Prediction & Underwriting Intelligence",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom Styling
    st.markdown("""
        <style>
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .risk-badge {
            font-size: 1.2rem;
            font-weight: 700;
            padding: 8px 16px;
            border-radius: 20px;
            display: inline-block;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.image("https://img.icons8.com/color/96/bank-building.png", width=70)
    st.sidebar.title("CreditIQ Risk Engine")
    st.sidebar.caption("Machine Learning Credit Decisioning System")

    nav_selection = st.sidebar.radio(
        "Navigation Menu",
        [
            "📊 Portfolio Risk Analytics",
            "🔍 Applicant Underwriting Simulator",
            "🧠 Model Explainability & ROI",
            "🗄️ Database & Audit Log (MySQL)",
        ]
    )

    predictor = LoanRiskPredictor()

    # Load metrics summary
    metrics_summary = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_summary = json.load(f)

    # =========================================================================
    # PAGE 1: PORTFOLIO RISK ANALYTICS
    # =========================================================================
    if nav_selection == "📊 Portfolio Risk Analytics":
        st.title("📊 Loan Portfolio & Risk Intelligence Overview")
        st.markdown(
            "Executive dashboard analyzing historical credit default signals, "
            "financial ratio distributions, and applicant demographics."
        )

        db_kpis = db_manager.get_portfolio_kpis()

        # Top KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Loan Applicants", f"{db_kpis['total_applicants']:,}")
        col2.metric("Average Credit Size", f"${db_kpis['avg_credit_amount']:,.2f}")
        col3.metric("Average Annual Income", f"${db_kpis['avg_income']:,.2f}")
        col4.metric("Historical Default Rate", f"{db_kpis['historical_default_rate_pct']:.1f}%")

        st.markdown("---")

        # Load sample data for charts
        sample_path = RAW_DATA_PATH
        if sample_path.exists():
            df_sample = pd.read_csv(sample_path, nrows=3000)

            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Credit Amount Distribution by Default Status")
                fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
                if HAS_SEABORN:
                    sns.kdeplot(
                        data=df_sample,
                        x="AMT_CREDIT",
                        hue="TARGET",
                        common_norm=False,
                        fill=True,
                        palette={0: "#2ca02c", 1: "#d62728"},
                        ax=ax
                    )
                else:
                    df_sample[df_sample["TARGET"] == 0]["AMT_CREDIT"].plot(kind="kde", ax=ax, label="Repaid", color="green")
                    df_sample[df_sample["TARGET"] == 1]["AMT_CREDIT"].plot(kind="kde", ax=ax, label="Defaulted", color="red")
                ax.set_xlabel("Credit Amount ($)", fontweight="bold")
                ax.set_ylabel("Density", fontweight="bold")
                ax.legend(["Defaulted (1)", "Repaid (0)"])
                st.pyplot(fig)

            with c2:
                st.subheader("Default Rate by Income Type")
                fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
                income_default = df_sample.groupby("NAME_INCOME_TYPE")["TARGET"].mean().sort_values(ascending=False) * 100
                if HAS_SEABORN:
                    sns.barplot(x=income_default.values, y=income_default.index, palette="Blues_r", ax=ax)
                else:
                    income_default.plot(kind="barh", ax=ax, color="#1f77b4")
                ax.set_xlabel("Default Rate (%)", fontweight="bold")
                ax.set_ylabel("Income Category", fontweight="bold")
                st.pyplot(fig)

            # Model ROC Comparison
            st.markdown("---")
            st.subheader("Model Performance: ROC & PR Curves")
            mc1, mc2 = st.columns(2)
            roc_img = FIGURES_DIR / "roc_curves.png"
            pr_img = FIGURES_DIR / "pr_curves.png"

            if roc_img.exists():
                mc1.image(str(roc_img), caption="ROC Curves Benchmark", use_container_width=True)
            if pr_img.exists():
                mc2.image(str(pr_img), caption="Precision-Recall Curves under Imbalance", use_container_width=True)

    # =========================================================================
    # PAGE 2: APPLICANT UNDERWRITING SIMULATOR
    # =========================================================================
    elif nav_selection == "🔍 Applicant Underwriting Simulator":
        st.title("🔍 Automated Credit Risk Underwriter")
        st.markdown(
            "Simulate an applicant credit assessment in real time. Adjust financial inputs "
            "to compute default probability, underwriting recommendation, and risk factor drivers."
        )

        with st.form("underwriting_form"):
            st.subheader("Applicant Financials")
            f1, f2, f3 = st.columns(3)
            income = f1.number_input("Annual Total Income ($)", min_value=10000.0, max_value=2000000.0, value=180000.0, step=5000.0)
            credit = f2.number_input("Requested Credit Amount ($)", min_value=10000.0, max_value=3000000.0, value=450000.0, step=10000.0)
            annuity = f3.number_input("Loan Annuity / Yearly Installment ($)", min_value=1000.0, max_value=200000.0, value=25000.0, step=1000.0)

            f4, f5, f6 = st.columns(3)
            goods_price = f4.number_input("Goods Price ($)", min_value=10000.0, max_value=3000000.0, value=400000.0, step=10000.0)
            age = f5.slider("Applicant Age (Years)", min_value=18, max_value=75, value=38)
            employed_years = f6.slider("Tenure at Current Job (Years)", min_value=0.0, max_value=40.0, value=5.5, step=0.5)

            st.subheader("Credit Bureau Ratings & Demographics")
            b1, b2, b3 = st.columns(3)
            ext_1 = b1.slider("Bureau Score 1 (EXT_SOURCE_1)", min_value=0.01, max_value=0.99, value=0.52, step=0.01)
            ext_2 = b2.slider("Bureau Score 2 (EXT_SOURCE_2)", min_value=0.01, max_value=0.99, value=0.60, step=0.01)
            ext_3 = b3.slider("Bureau Score 3 (EXT_SOURCE_3)", min_value=0.01, max_value=0.99, value=0.58, step=0.01)

            d1, d2, d3, d4 = st.columns(4)
            contract_type = d1.selectbox("Loan Contract Type", ["Cash loans", "Revolving loans"])
            education = d2.selectbox("Education Level", ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary"])
            housing = d3.selectbox("Housing Type", ["House / apartment", "Rented apartment", "With parents", "Municipal apartment"])
            family_status = d4.selectbox("Family Status", ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"])

            underwriter_notes = st.text_input("Underwriter Audit Notes", placeholder="e.g. Verified income documentation via tax filings.")
            submitted = st.form_submit_button("⚡ Evaluate Credit Risk & Decision", use_container_width=True)

        if submitted:
            applicant_data = {
                "SK_ID_CURR": int(np.random.randint(200000, 999999)),
                "AMT_INCOME_TOTAL": income,
                "AMT_CREDIT": credit,
                "AMT_ANNUITY": annuity,
                "AMT_GOODS_PRICE": goods_price,
                "DAYS_BIRTH": -int(age * 365.25),
                "DAYS_EMPLOYED": -int(employed_years * 365.25),
                "DAYS_REGISTRATION": -int(age * 150),
                "DAYS_ID_PUBLISH": -int(age * 80),
                "EXT_SOURCE_1": ext_1,
                "EXT_SOURCE_2": ext_2,
                "EXT_SOURCE_3": ext_3,
                "CNT_CHILDREN": 0,
                "CNT_FAM_MEMBERS": 2,
                "NAME_CONTRACT_TYPE": contract_type,
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "Y",
                "FLAG_OWN_REALTY": "Y",
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": education,
                "NAME_FAMILY_STATUS": family_status,
                "NAME_HOUSING_TYPE": housing,
                "REGION_RATING_CLIENT": 2,
            }

            result = predictor.predict_single(applicant_data)

            st.markdown("---")
            st.subheader("📋 Underwriting Assessment Result")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Predicted Default Risk", result["default_risk_percentage"])
            r2.metric("Assigned Risk Tier", result["risk_tier"])
            r3.metric("Underwriting Decision", result["recommendation"])
            r4.metric("Optimal Risk Cutoff", f"{result['decision_cutoff']:.2f}")

            # Color-coded banner
            if result["recommendation"] == "APPROVE":
                st.success(f"**RECOMMENDATION: {result['recommendation']}** — {result['recommendation_rationale']}")
            elif result["recommendation"] == "MANUAL REVIEW":
                st.warning(f"**RECOMMENDATION: {result['recommendation']}** — {result['recommendation_rationale']}")
            else:
                st.error(f"**RECOMMENDATION: {result['recommendation']}** — {result['recommendation_rationale']}")

            # Key Financial Ratios
            st.markdown("#### Financial Leverage Health Check")
            rc1, rc2, rc3 = st.columns(3)
            dti = credit / income
            ati = (annuity / income) * 100
            ctg = credit / goods_price

            rc1.metric("Debt-to-Income (DTI)", f"{dti:.2f}x", delta="Safe (<3.0x)" if dti < 3.0 else "High Leverage (>4.0x)", delta_color="inverse" if dti >= 3.0 else "normal")
            rc2.metric("Annuity-to-Income", f"{ati:.1f}%", delta="Manageable (<20%)" if ati < 20 else "Heavy Burden", delta_color="inverse" if ati >= 20 else "normal")
            rc3.metric("Credit-to-Goods Ratio", f"{ctg:.2f}x", delta="Normal" if ctg <= 1.1 else "Financing Surplus", delta_color="off")

            # Risk Factor Drivers
            if result["key_risk_factors"]:
                st.markdown("#### Primary Decision Drivers (Explainability)")
                for factor in result["key_risk_factors"]:
                    st.info(f"**{factor['feature']} = {factor['value']}** ({factor['impact']}): {factor['description']}")

            # Persist Decision to MySQL / Database
            log_id = db_manager.log_underwriting_decision(
                sk_id_curr=applicant_data["SK_ID_CURR"],
                default_probability=result["default_probability"],
                risk_tier=result["risk_tier"],
                recommendation=result["recommendation"],
                decision_threshold=result["decision_cutoff"],
                key_drivers=result["key_risk_factors"],
                notes=underwriter_notes or "Evaluated via Streamlit Underwriter Portal"
            )
            st.caption(f"💾 Decision logged to MySQL audit ledger with Decision ID #{log_id} ({db_manager.db_type.upper()}).")

    # =========================================================================
    # PAGE 3: MODEL EXPLAINABILITY & ROI
    # =========================================================================
    elif nav_selection == "🧠 Model Explainability & ROI":
        st.title("🧠 Model Interpretability & Business ROI Analysis")
        st.markdown(
            "Transparency into model decision mechanics, feature importance, "
            "and cost-sensitive threshold optimization."
        )

        col1, col2 = st.columns(2)

        feat_imp_img = FIGURES_DIR / "feature_importance.png"
        cost_img = FIGURES_DIR / "cost_curve.png"

        with col1:
            if feat_imp_img.exists():
                st.image(str(feat_imp_img), caption="Global Top 20 Feature Importances", use_container_width=True)

        with col2:
            if cost_img.exists():
                st.image(str(cost_img), caption="Cost Optimization Curve ($5 FN + $1 FP)", use_container_width=True)

        st.markdown("---")
        st.subheader("Threshold Decision Matrix & Financial Savings")

        if metrics_summary:
            opt_data = metrics_summary.get("cost_optimization", {})
            st.markdown(f"""
            - **Default Policy Cutoff (0.50):** Financial Loss Units = **{opt_data.get('default_0_5_cost', 'N/A')}**
            - **Cost-Optimal Cutoff ({opt_data.get('optimal_cost_threshold', 'N/A')}):** Financial Loss Units = **{opt_data.get('min_financial_loss', 'N/A')}**
            - **Portfolio Loss Reduction:** **{opt_data.get('cost_reduction_pct', 0.0):.1f}%** financial savings achieved by asymmetric loss calibration.
            """)

        cm_img = FIGURES_DIR / "confusion_matrix.png"
        risk_dist_img = FIGURES_DIR / "risk_score_distribution.png"

        c3, c4 = st.columns(2)
        if cm_img.exists():
            c3.image(str(cm_img), caption="Confusion Matrix: Standard vs Cost-Optimal Cutoff", use_container_width=True)
        if risk_dist_img.exists():
            c4.image(str(risk_dist_img), caption="Predicted Probability Density by Actual Repayment", use_container_width=True)

    # =========================================================================
    # PAGE 4: DATABASE & AUDIT LOG (MYSQL)
    # =========================================================================
    elif nav_selection == "🗄️ Database & Audit Log (MySQL)":
        st.title("🗄️ Enterprise Database & Underwriting Audit Log")
        st.markdown(
            f"Query active loan applications and inspect timestamped algorithmic underwriting "
            f"decisions recorded in **{db_manager.db_type.upper()}** (`loan_risk_db`)."
        )

        st.subheader("Recent Automated Underwriting Decisions")
        decisions_df = db_manager.get_recent_decisions(limit=25)

        if not decisions_df.empty:
            # Filters
            rec_filter = st.multiselect(
                "Filter by Recommendation",
                options=["APPROVE", "MANUAL REVIEW", "DECLINE"],
                default=["APPROVE", "MANUAL REVIEW", "DECLINE"]
            )
            filtered_df = decisions_df[decisions_df["recommendation"].isin(rec_filter)]
            st.dataframe(filtered_df, use_container_width=True)
            st.caption(f"Showing {len(filtered_df)} decision records.")
        else:
            st.info("No decision records found yet. Submit an application in the Underwriting Simulator to log records!")

        st.markdown("---")
        st.subheader("MySQL Database Architecture")
        st.code("""
-- Tables:
1. `applicants`             : Core applicant demographics & financial data
2. `credit_bureau_ratings`  : External bureau scores (EXT_SOURCE 1, 2, 3)
3. `financial_ratios`       : Calculated debt burden and leverage metrics
4. `underwriting_decisions` : Auditable log of probabilities, risk tiers & decisions
        """, language="sql")


if __name__ == "__main__":
    run_streamlit_dashboard()
