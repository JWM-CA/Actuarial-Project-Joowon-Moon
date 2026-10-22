import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ==============================================================================
# PHASE 1: DATA LOADING & ACTUARIAL ASSUMPTIONS (SOA 2015 VBT WORKBOOKS)
# ==============================================================================

@dataclass
class ActuarialAssumptions:
    """Stores policyholder profile, SOA 2015 VBT table basis, economic rates, and expenses."""
    issue_age: int = 45
    gender: str = "Male"               # "Male" or "Female"
    smoker_status: str = "Non-Smoker"  # "Non-Smoker" or "Smoker"
    age_basis: str = "ANB"             # "ANB" (Age Nearest Birthday) or "ALB" (Age Last Birthday)
    term_years: int = 20
    face_amount: float = 500_000.0

    # Economic & Discount Rate Assumptions
    valuation_rate: float = 0.035      # 3.5% statutory valuation interest rate
    earned_rate: float = 0.045         # 4.5% earned investment return on reserves/cash flows
    hurdle_rate: float = 0.080         # 8.0% risk discount rate (hurdle rate for PVFP)

    # Underwriting Preferred Discount (initial ratio to Standard VBT mortality at Duration 1)
    preferred_discount: float = 0.75   # 25% initial preferred discount that wears off via VBT factors

    # Expense Assumptions
    acq_expense_fixed: float = 300.0   # Year 1 fixed acquisition cost per policy ($)
    acq_expense_pct: float = 0.55      # Year 1 % of gross premium (commissions & underwriting)
    maint_expense_fixed: float = 55.0  # Renewal annual maintenance cost per policy ($)
    maint_expense_pct: float = 0.04    # Renewal % of gross premium (premium tax & servicing)
    inflation_rate: float = 0.020      # 2.0% annual inflation on fixed maintenance expense

    # Target Profit Loading & Valuation PfAD (Provision for Adverse Deviations)
    target_profit_loading: float = 0.12 # 12% profit/contingency loading in gross premium
    valuation_mort_pad: float = 0.10    # 10% mortality PfAD margin for Dynamic Reserve


class SOA2015VBTDataLoader:
    """Loads SOA 2015 VBT Smoker-Distinct ALB/ANB tables and Preferred Wear-off Factors."""

    SHEET_MAP = {
        ("male", "non-smoker", "anb"): "2015 MNS ANB",
        ("female", "non-smoker", "anb"): "2015 FNS ANB",
        ("male", "smoker", "anb"): "2015 MSM ANB",
        ("female", "smoker", "anb"): "2015 FSM ANB",
        ("male", "non-smoker", "alb"): "2015 MNS ALB",
        ("female", "non-smoker", "alb"): "2015 FNS ALB",
        ("male", "smoker", "alb"): "2015 MSM ALB",
        ("female", "smoker", "alb"): "2015 FSM ALB",
    }

    def __init__(
        self,
        vbt_filepath: str = "2015-vbt-smoker-distinct-alb-anb.xlsx",
        wearoff_filepath: str = "2015-vbt-preferred-wearoff-factors_2.xlsx"
    ):
        self.vbt_filepath = vbt_filepath
        if not os.path.exists(wearoff_filepath) and os.path.exists("2015-vbt-preferred-wearoff-factors.xlsx"):
            self.wearoff_filepath = "2015-vbt-preferred-wearoff-factors.xlsx"
        else:
            self.wearoff_filepath = wearoff_filepath

        self.vbt_tables = self._load_all_vbt_sheets()
        self.wearoff_df = self._load_wearoff_table()

    def _load_all_vbt_sheets(self) -> Dict[str, pd.DataFrame]:
        xls = pd.ExcelFile(self.vbt_filepath)
        tables = {}
        for sheet in xls.sheet_names:
            df = pd.read_excel(self.vbt_filepath, sheet_name=sheet, skiprows=2)
            df = df.dropna(subset=["Iss. Age"]).copy()
            df["Iss. Age"] = df["Iss. Age"].astype(int)
            tables[sheet] = df.set_index("Iss. Age")
        return tables

    def _load_wearoff_table(self) -> pd.DataFrame:
        xls = pd.ExcelFile(self.wearoff_filepath)
        sheet_name = xls.sheet_names[0]
        df = pd.read_excel(self.wearoff_filepath, sheet_name=sheet_name, skiprows=2)
        df = df.dropna(subset=["Issue Age"]).copy()
        df["Issue Age"] = df["Issue Age"].astype(int)
        return df.set_index("Issue Age")

    def get_sheet_name(self, gender: str, smoker_status: str, age_basis: str) -> str:
        key = (gender.strip().lower(), smoker_status.strip().lower(), age_basis.strip().lower())
        if key not in self.SHEET_MAP:
            raise ValueError(f"Invalid table key: {key}")
        return self.SHEET_MAP[key]

    def get_base_select_mortality(self, assumptions: ActuarialAssumptions) -> np.ndarray:
        sheet_name = self.get_sheet_name(assumptions.gender, assumptions.smoker_status, assumptions.age_basis)
        df = self.vbt_tables[sheet_name]
        if assumptions.issue_age not in df.index:
            raise ValueError(f"Issue Age {assumptions.issue_age} not found in {sheet_name} (18-95).")
        row = df.loc[assumptions.issue_age]
        rates_per_1000 = [float(row[d if d <= 25 else "Ult."]) for d in range(1, assumptions.term_years + 1)]
        return np.clip(np.array(rates_per_1000, dtype=float) / 1000.0, 0.0, 1.0)

    def get_wearoff_factors(self, issue_age: int, term_years: int) -> np.ndarray:
        row = self.wearoff_df.loc[issue_age]
        factors = [float(row[d if d <= 25 else "Ultimate"]) for d in range(1, term_years + 1)]
        return np.array(factors, dtype=float)

    def get_effective_mortality(
        self,
        assumptions: ActuarialAssumptions,
        mortality_mult: float = 1.0,
        apply_wearoff: bool = True
    ) -> pd.DataFrame:
        durations = np.arange(1, assumptions.term_years + 1)
        attained_ages = assumptions.issue_age + durations - 1
        q_base = self.get_base_select_mortality(assumptions)
        w_factors = self.get_wearoff_factors(assumptions.issue_age, assumptions.term_years)

        if apply_wearoff:
            pref_ratio = assumptions.preferred_discount + w_factors * (1.0 - assumptions.preferred_discount)
        else:
            pref_ratio = np.full_like(w_factors, assumptions.preferred_discount)

        q_eff = np.clip(q_base * pref_ratio * mortality_mult, 0.0, 1.0)

        return pd.DataFrame({
            "Duration": durations,
            "Attained_Age": attained_ages,
            "VBT_Base_q_Per_1000": q_base * 1000.0,
            "Base_q": q_base,
            "Wearoff_Factor": w_factors,
            "Effective_Pref_Ratio": pref_ratio,
            "Effective_q": q_eff
        })

    @staticmethod
    def get_lapse_schedule(term_years: int, lapse_mult: float = 1.0) -> np.ndarray:
        base_lapse = np.array(
            [0.08, 0.07, 0.06] + [0.045] * max(0, term_years - 3), dtype=float
        )[:term_years]
        q_w = np.clip(base_lapse * lapse_mult, 0.0, 0.95)
        q_w[-1] = 0.0
        return q_w


# ==============================================================================
# PHASE 2: CORE ACTUARIAL MATH & DYNAMIC RESERVE ENGINE
# ==============================================================================

def compute_irr(cf_series: np.ndarray) -> float:
    """Computes Internal Rate of Return (IRR) for an annual profit vector via bisection."""
    if cf_series[0] >= 0 or np.sum(cf_series) <= 0:
        return np.nan
    low, high = -0.50, 2.00
    periods = np.arange(len(cf_series))
    for _ in range(100):
        mid = (low + high) / 2.0
        npv = np.sum(cf_series / ((1.0 + mid) ** periods))
        if npv > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


class TermLifeEngine:
    """Calculates Net/Gross Premiums, Traditional Prospective Reserves, and Dynamic Reserves."""

    def __init__(self, assumptions: ActuarialAssumptions, loader: SOA2015VBTDataLoader):
        self.assumptions = assumptions
        self.loader = loader

    def solve_goal_seek_gross_premium(self, target_margin: float) -> float:
        """Solves via bisection for the Gross Premium that achieves exact target Profit Margin."""
        low, high = 10.0, self.assumptions.face_amount * 0.25
        for _ in range(60):
            mid = (low + high) / 2.0
            res = self.run_valuation(fixed_gross_premium=mid)
            if res["profit_margin"] < target_margin:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    def run_valuation(
        self,
        interest_shock: float = 0.0,
        mortality_mult: float = 1.0,
        lapse_mult: float = 1.0,
        apply_wearoff: bool = True,
        fixed_gross_premium: Optional[float] = None
    ) -> Dict[str, object]:
        a = self.assumptions
        n = a.term_years
        val_i = max(0.0001, a.valuation_rate + interest_shock)
        earn_i = max(0.0001, a.earned_rate + interest_shock)
        v_val = 1.0 / (1.0 + val_i)
        v_earn = 1.0 / (1.0 + earn_i)

        mort_df = self.loader.get_effective_mortality(
            a, mortality_mult=mortality_mult, apply_wearoff=apply_wearoff
        )
        q_d = mort_df["Effective_q"].values
        q_w = self.loader.get_lapse_schedule(n, lapse_mult=lapse_mult)

        # 1. Traditional Net Level Prospective Reserve (Single-Decrement: Mortality Only)
        p_d = 1.0 - q_d
        t_p_x_mort = np.ones(n + 1)
        for t in range(1, n + 1):
            t_p_x_mort[t] = t_p_x_mort[t - 1] * p_d[t - 1]

        disc_beg = v_val ** np.arange(n)
        disc_end = v_val ** np.arange(1, n + 1)

        epv_annuity_due = np.sum(t_p_x_mort[:-1] * disc_beg)
        epv_unit_benefit = np.sum(t_p_x_mort[:-1] * q_d * disc_end)
        net_premium = a.face_amount * (epv_unit_benefit / epv_annuity_due)

        net_reserves = np.zeros(n + 1)
        for t in range(1, n):
            rem = n - t
            surv_t = t_p_x_mort[t:n] / t_p_x_mort[t]
            a_xt = np.sum(surv_t * (v_val ** np.arange(rem)))
            A_xt = np.sum(surv_t * q_d[t:n] * (v_val ** np.arange(1, rem + 1)))
            net_reserves[t] = a.face_amount * A_xt - net_premium * a_xt
        net_reserves[n] = 0.0

        # 2. Multiple-Decrement Survival (Mortality + Lapse) & Gross Premium
        p_tau = (1.0 - q_d) * (1.0 - q_w)
        t_p_x_active = np.ones(n + 1)
        for t in range(1, n + 1):
            t_p_x_active[t] = t_p_x_active[t - 1] * p_tau[t - 1]

        disc_earn_beg = v_earn ** np.arange(n)
        disc_earn_end = v_earn ** np.arange(1, n + 1)
        epv_claims = np.sum(t_p_x_active[:-1] * q_d * a.face_amount * disc_earn_end)
        active_pure_premium = epv_claims / np.sum(t_p_x_active[:-1] * disc_earn_beg)

        durations = np.arange(1, n + 1)
        fixed_expenses = np.where(
            durations == 1,
            a.acq_expense_fixed,
            a.maint_expense_fixed * ((1.0 + a.inflation_rate) ** (durations - 1))
        )
        pct_expenses = np.where(durations == 1, a.acq_expense_pct, a.maint_expense_pct)

        if fixed_gross_premium is None:
            epv_fixed_exp = np.sum(t_p_x_active[:-1] * fixed_expenses * disc_earn_beg)
            epv_prem_factor = np.sum(
                t_p_x_active[:-1] * (1.0 - pct_expenses - a.target_profit_loading) * disc_earn_beg
            )
            gross_premium = (epv_claims + epv_fixed_exp) / epv_prem_factor
        else:
            gross_premium = fixed_gross_premium

        total_expenses_per_pol = fixed_expenses + pct_expenses * gross_premium

        # 3. Dynamic Reserve (Multi-Decrement Recursive Valuation with PfAD)
        q_d_val = np.clip(q_d * (1.0 + a.valuation_mort_pad), 0.0, 1.0)
        p_tau_val = (1.0 - q_d_val) * (1.0 - q_w)
        t_p_val = np.ones(n + 1)
        for t in range(1, n + 1):
            t_p_val[t] = t_p_val[t - 1] * p_tau_val[t - 1]

        renewal_exp = (
            a.maint_expense_fixed * ((1.0 + a.inflation_rate) ** (durations - 1))
            + a.maint_expense_pct * gross_premium
        )
        epv_val_ann = np.sum(t_p_val[:-1] * disc_beg)
        epv_val_outflows = np.sum(
            t_p_val[:-1] * (q_d_val * a.face_amount * v_val + renewal_exp) * disc_beg
        )
        valuation_premium = epv_val_outflows / epv_val_ann

        dynamic_reserves = np.zeros(n + 1)
        for t in range(n - 1, 0, -1):
            cf_start = renewal_exp[t] - valuation_premium
            cf_end = q_d_val[t] * a.face_amount + p_tau_val[t] * dynamic_reserves[t + 1]
            dynamic_reserves[t] = max(0.0, cf_start + v_val * cf_end)
        dynamic_reserves[0] = 0.0

        # 4. Cash Flow Projection (per 1.0 Initial Issued Policy)
        inforce_beg = t_p_x_active[:-1]
        inforce_end = t_p_x_active[1:]

        prem_income = inforce_beg * gross_premium
        expense_outflow = inforce_beg * total_expenses_per_pol
        death_claims = inforce_beg * q_d * a.face_amount

        res_beg = inforce_beg * dynamic_reserves[:-1]
        res_end = inforce_end * dynamic_reserves[1:]
        delta_reserve = res_end - res_beg

        invested_assets = res_beg + prem_income - expense_outflow
        investment_income = invested_assets * earn_i
        net_profit = prem_income - expense_outflow + investment_income - death_claims - delta_reserve

        proj_df = pd.DataFrame({
            "Year": durations,
            "Attained_Age": mort_df["Attained_Age"].values,
            "VBT_Base_q_Per_1000": mort_df["VBT_Base_q_Per_1000"].values,
            "Wearoff_Factor": mort_df["Wearoff_Factor"].values,
            "Mortality_q": q_d,
            "Lapse_q": q_w,
            "Inforce_Beg": inforce_beg,
            "Inforce_End": inforce_end,
            "Net_Reserve_Per_Pol": net_reserves[1:],
            "Dynamic_Reserve_Per_Pol": dynamic_reserves[1:],
            "Premium_Income": prem_income,
            "Expenses": expense_outflow,
            "Investment_Income": investment_income,
            "Death_Claims": death_claims,
            "Reserve_Change": delta_reserve,
            "Net_Profit": net_profit
        })

        # 5. Pricing Key Performance Indicators (KPIs)
        disc_hurdle_beg = (1.0 + a.hurdle_rate) ** (-np.arange(n))
        disc_hurdle_end = (1.0 + a.hurdle_rate) ** (-np.arange(1, n + 1))

        pv_profits = float(np.sum(net_profit * disc_hurdle_end))
        pv_gross_prems = float(np.sum(prem_income * disc_hurdle_beg))
        profit_margin = pv_profits / pv_gross_prems if pv_gross_prems > 0 else 0.0
        irr = compute_irr(net_profit)

        return {
            "table_used": self.loader.get_sheet_name(a.gender, a.smoker_status, a.age_basis),
            "net_premium": float(net_premium),
            "active_pure_premium": float(active_pure_premium),
            "valuation_premium": float(valuation_premium),
            "gross_premium": float(gross_premium),
            "pvfp": pv_profits,
            "pv_premiums": pv_gross_prems,
            "profit_margin": float(profit_margin),
            "irr": float(irr) if not np.isnan(irr) else np.nan,
            "peak_net_reserve": float(np.max(net_reserves)),
            "peak_dynamic_reserve": float(np.max(dynamic_reserves)),
            "projection_df": proj_df
        }


# ==============================================================================
# PHASE 3: STRESS TESTING & SCENARIO SIMULATION ENGINE
# ==============================================================================

class PricingSimulationEngine:
    """Executes sensitivity and stress-testing scenarios holding Gross Premium constant."""

    def __init__(self, term_engine: TermLifeEngine):
        self.engine = term_engine

    def run_stress_suite(self, locked_gp: Optional[float] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        baseline_res = self.engine.run_valuation(fixed_gross_premium=locked_gp)
        contract_gp = baseline_res["gross_premium"]
        sheet_label = baseline_res["table_used"]

        scenarios = [
            (f"1. Baseline ({sheet_label})", {"interest_shock": 0.00, "mortality_mult": 1.00, "lapse_mult": 1.00, "apply_wearoff": True}),
            ("2. Interest Rate -100 bps",    {"interest_shock": -0.01, "mortality_mult": 1.00, "lapse_mult": 1.00, "apply_wearoff": True}),
            ("3. Interest Rate +100 bps",    {"interest_shock": +0.01, "mortality_mult": 1.00, "lapse_mult": 1.00, "apply_wearoff": True}),
            ("4. Mortality +15% Shock",      {"interest_shock": 0.00, "mortality_mult": 1.15, "lapse_mult": 1.00, "apply_wearoff": True}),
            ("5. Lapse +25% Shock",          {"interest_shock": 0.00, "mortality_mult": 1.00, "lapse_mult": 1.25, "apply_wearoff": True}),
            ("6. Lapse -25% Shock (Tail)",   {"interest_shock": 0.00, "mortality_mult": 1.00, "lapse_mult": 0.75, "apply_wearoff": True}),
            ("7. No Preferred Wear-off",     {"interest_shock": 0.00, "mortality_mult": 1.00, "lapse_mult": 1.00, "apply_wearoff": False}),
        ]

        summary_rows = []
        scenario_projections = {}

        for name, params in scenarios:
            res = self.engine.run_valuation(fixed_gross_premium=contract_gp, **params)
            scenario_projections[name] = res["projection_df"]
            summary_rows.append({
                "Scenario": name,
                "Net_Premium ($)": round(res["net_premium"], 2),
                "Active_Pure_Prem ($)": round(res["active_pure_premium"], 2),
                "Locked_Gross_Prem ($)": round(res["gross_premium"], 2),
                "PVFP ($)": round(res["pvfp"], 2),
                "Profit_Margin (%)": round(res["profit_margin"] * 100, 2),
                "IRR (%)": round(res["irr"] * 100, 2) if not np.isnan(res["irr"]) else np.nan,
                "Peak_Net_Reserve ($)": round(res["peak_net_reserve"], 2),
                "Peak_Dynamic_Reserve ($)": round(res["peak_dynamic_reserve"], 2),
            })

        return pd.DataFrame(summary_rows), scenario_projections

    def compare_all_vbt_cohorts(self, use_goal_seek: bool = False) -> pd.DataFrame:
        rows = []
        a = self.engine.assumptions
        for gender in ["Male", "Female"]:
            for smoker in ["Non-Smoker", "Smoker"]:
                for basis in ["ANB", "ALB"]:
                    cohort_assump = ActuarialAssumptions(
                        issue_age=a.issue_age,
                        gender=gender,
                        smoker_status=smoker,
                        age_basis=basis,
                        term_years=a.term_years,
                        face_amount=a.face_amount,
                        valuation_rate=a.valuation_rate,
                        earned_rate=a.earned_rate,
                        hurdle_rate=a.hurdle_rate,
                        preferred_discount=a.preferred_discount,
                        acq_expense_fixed=a.acq_expense_fixed,
                        acq_expense_pct=a.acq_expense_pct,
                        maint_expense_fixed=a.maint_expense_fixed,
                        maint_expense_pct=a.maint_expense_pct,
                        inflation_rate=a.inflation_rate,
                        target_profit_loading=a.target_profit_loading,
                        valuation_mort_pad=a.valuation_mort_pad
                    )
                    temp_engine = TermLifeEngine(cohort_assump, self.engine.loader)
                    gp = temp_engine.solve_goal_seek_gross_premium(a.target_profit_loading) if use_goal_seek else None
                    res = temp_engine.run_valuation(fixed_gross_premium=gp)
                    rows.append({
                        "VBT_Sheet": res["table_used"],
                        "Gender": gender,
                        "Smoker_Status": smoker,
                        "Age_Basis": basis,
                        "Net_Premium ($)": round(res["net_premium"], 2),
                        "Active_Pure_Prem ($)": round(res["active_pure_premium"], 2),
                        "Gross_Premium ($)": round(res["gross_premium"], 2),
                        "PVFP ($)": round(res["pvfp"], 2),
                        "Profit_Margin (%)": round(res["profit_margin"] * 100, 2),
                        "IRR (%)": round(res["irr"] * 100, 2) if not np.isnan(res["irr"]) else np.nan,
                        "Peak_Net_Reserve ($)": round(res["peak_net_reserve"], 2),
                        "Peak_Dynamic_Reserve ($)": round(res["peak_dynamic_reserve"], 2),
                    })
        return pd.DataFrame(rows)


# ==============================================================================
# PHASE 4: INSTITUTIONAL STREAMLIT WORKBENCH (DEEP NAVY + TEAL + IVORY)
# ==============================================================================

st.set_page_config(
    page_title="Term Life Pricing & Dynamic Reserve Engine | SOA 2015 VBT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep Navy (#0B192C) + Teal (#14B8A6) + Ivory (#FAF8F5) Institutional Styling
# Works identically across both macOS/Browser Dark Mode and Light Mode
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    h1, h2, h3, h4, h5 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        letter-spacing: -0.01em;
    }
    /* KPI Metric Cards: Deep Navy Surface + Teal Left Accent + Crisp Ivory Typography */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #0B192C 0%, #112240 100%) !important;
        border: 1px solid rgba(20, 184, 166, 0.35) !important;
        border-left: 4px solid #14B8A6 !important;
        border-radius: 8px !important;
        padding: 14px 16px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.14) !important;
    }
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #E6DEC8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #FAF8F5 !important;
    }
    div[data-testid="stMetricDelta"] svg {
        display: none !important;
    }
    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] > div {
        color: #2DD4BF !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
    }
    /* Active Tab Styling with Teal Underline */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
    }
    /* Prominent Institutional Styling for Download Buttons */
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(145deg, #0B192C 0%, #112240 100%) !important;
        color: #FAF8F5 !important;
        border: 1.5px solid #14B8A6 !important;
        border-left: 5px solid #14B8A6 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.1rem !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.12) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        border-color: #2DD4BF !important;
        color: #2DD4BF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_cached_vbt_data() -> SOA2015VBTDataLoader:
    return SOA2015VBTDataLoader(
        vbt_filepath="2015-vbt-smoker-distinct-alb-anb.xlsx",
        wearoff_filepath="2015-vbt-preferred-wearoff-factors_2.xlsx"
    )


def get_theme_palette(canvas_mode: str) -> Dict[str, object]:
    """
    Returns coordinated Deep Navy + Teal + Ivory color tokens for:
    1) 'Auto (Light & Dark Adaptive)' - transparent bg, universal high-contrast traces
    2) 'Deep Navy Canvas (Dark)'      - locked #0B192C bg with Ivory & Vibrant Teal traces
    3) 'Warm Ivory Canvas (Light)'    - locked #FAF8F5 bg with Deep Navy & Rich Teal traces
    """
    if canvas_mode == "Deep Navy Canvas (Dark)":
        return {
            "plotly_theme": None,
            "paper_bg": "#0B192C",
            "plot_bg": "#112240",
            "font_color": "#FAF8F5",
            "grid_color": "rgba(230, 222, 200, 0.14)",
            "axis_line": "#334E68",
            "legend_bg": "rgba(17, 34, 64, 0.92)",
            "legend_border": "#14B8A6",
            "navy_primary": "#38BDF8",       # Luminous Sky-Navy on Dark Navy canvas
            "teal_primary": "#2DD4BF",       # Electric Teal
            "teal_secondary": "#0D9488",     # Deep Teal
            "ivory_primary": "#FAF8F5",      # Crisp Ivory line/marker
            "ivory_fill": "rgba(230, 222, 200, 0.25)",
            "ivory_bar": "#E6DEC8",          # Warm Ivory Bar
            "slate_muted": "#94A3B8",        # Silver Slate for dotted baseline
            "coral_alert": "#FB7185",        # High-visibility Coral Rose for claims/shocks
            "amber_accent": "#FBBF24",       # Warm Gold-Ivory accent
            "stress_palette": [
                "#2DD4BF", "#FAF8F5", "#38BDF8", "#FB7185", "#FBBF24", "#A78BFA", "#94A3B8"
            ]
        }
    elif canvas_mode == "Warm Ivory Canvas (Light)":
        return {
            "plotly_theme": None,
            "paper_bg": "#FAF8F5",
            "plot_bg": "#F4EFE6",
            "font_color": "#0B192C",
            "grid_color": "rgba(11, 25, 44, 0.10)",
            "axis_line": "#94A3B8",
            "legend_bg": "rgba(250, 248, 245, 0.94)",
            "legend_border": "#0D9488",
            "navy_primary": "#0B192C",       # Deep Institutional Navy
            "teal_primary": "#0D9488",       # Rich Teal
            "teal_secondary": "#14B8A6",     # Bright Teal
            "ivory_primary": "#1E3A8A",      # Royal Navy contrast line on Ivory canvas
            "ivory_fill": "rgba(212, 180, 131, 0.30)",
            "ivory_bar": "#D8C8B0",          # Warm Ivory-Sand Bar
            "slate_muted": "#64748B",        # Slate for dotted baseline
            "coral_alert": "#E11D48",        # Crimson Rose for claims/shocks
            "amber_accent": "#D97706",       # Rich Amber
            "stress_palette": [
                "#0B192C", "#0D9488", "#2563EB", "#E11D48", "#D97706", "#7C3AED", "#64748B"
            ]
        }
    else:
        # Auto (Light & Dark Adaptive): works natively with Streamlit Light Mode & Dark Mode
        return {
            "plotly_theme": "streamlit",
            "paper_bg": "rgba(0,0,0,0)",
            "plot_bg": "rgba(20, 184, 166, 0.04)",
            "font_color": None,              # Inherit automatically from Streamlit Light/Dark theme
            "grid_color": "rgba(148, 163, 184, 0.22)",
            "axis_line": "rgba(148, 163, 184, 0.45)",
            "legend_bg": "rgba(148, 163, 184, 0.10)",
            "legend_border": "#14B8A6",
            "navy_primary": "#2563EB",       # Luminous Cobalt-Navy (high contrast on both dark & white)
            "teal_primary": "#14B8A6",       # Vibrant Teal (high contrast on both dark & white)
            "teal_secondary": "#0D9488",     # Deep Teal
            "ivory_primary": "#00B4D8",      # Cyan-Teal contrast line with Ivory markers
            "ivory_fill": "rgba(212, 180, 131, 0.24)",
            "ivory_bar": "#D4B483",          # Warm Ivory-Sand (visible on white AND dark)
            "slate_muted": "#94A3B8",        # Neutral Slate
            "coral_alert": "#F43F5E",        # Vibrant Rose-Crimson
            "amber_accent": "#F59E0B",       # Vivid Amber
            "stress_palette": [
                "#14B8A6", "#2563EB", "#D4B483", "#F43F5E", "#38BDF8", "#F59E0B", "#A78BFA"
            ]
        }


def apply_institutional_chart_layout(
    fig: go.Figure,
    title: str,
    pal: Dict[str, object],
    height: int = 460,
    bottom_margin: int = 105,
    legend_y: float = -0.22
) -> go.Figure:
    """
    Applies a non-overlapping layout:
    - Title is anchored cleanly at the top-left (y=0.96).
    - Legend is placed horizontally BELOW the x-axis (y=-0.22) in a bordered pill box,
      guaranteeing zero collision between the chart title and legend across all screen widths.
    """
    layout_kwargs = dict(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.0,
            y=0.96,
            xanchor="left",
            yanchor="top",
            font=dict(size=15, family="Arial, sans-serif")
        ),
        font=dict(family="Arial, sans-serif", size=12),
        paper_bgcolor=pal["paper_bg"],
        plot_bgcolor=pal["plot_bg"],
        hovermode="x unified",
        height=height,
        margin=dict(l=58, r=52, t=58, b=bottom_margin),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="left",
            x=0.0,
            bgcolor=pal["legend_bg"],
            bordercolor=pal["legend_border"],
            borderwidth=1,
            font=dict(size=11)
        )
    )

    if pal["font_color"] is not None:
        layout_kwargs["font"]["color"] = pal["font_color"]
        layout_kwargs["title"]["font"]["color"] = pal["font_color"]
        layout_kwargs["legend"]["font"]["color"] = pal["font_color"]

    fig.update_layout(**layout_kwargs)

    axis_common = dict(
        showgrid=True,
        gridcolor=pal["grid_color"],
        zeroline=False,
        linecolor=pal["axis_line"],
        tickfont=dict(color=pal["font_color"]) if pal["font_color"] else None,
        title_font=dict(color=pal["font_color"]) if pal["font_color"] else None
    )
    fig.update_xaxes(**axis_common)
    fig.update_yaxes(**axis_common)
    return fig


def render_chart(fig: go.Figure, pal: Dict[str, object]) -> None:
    """Renders Plotly chart with exact theme control."""
    st.plotly_chart(fig, use_container_width=True, theme=pal["plotly_theme"])


def generate_excel_report(
    assumptions: ActuarialAssumptions,
    pricing_mode: str,
    base_res: Dict[str, object],
    proj_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    cohort_df: pd.DataFrame
) -> bytes:
    """
    Generates an in-memory multi-sheet Excel actuarial report (.xlsx) using pandas and io.BytesIO
    reusing the exact valuation outputs, schedules, and stress tables already computed in the app.
    """
    output = io.BytesIO()

    # 1. Build Executive Summary & Assumptions table
    summary_rows = [
        ("Policyholder & Contract Profile", "Mortality Table Used", str(base_res["table_used"])),
        ("Policyholder & Contract Profile", "Issue Age (x)", assumptions.issue_age),
        ("Policyholder & Contract Profile", "Gender", assumptions.gender),
        ("Policyholder & Contract Profile", "Smoker Status", assumptions.smoker_status),
        ("Policyholder & Contract Profile", "Age Basis", assumptions.age_basis),
        ("Policyholder & Contract Profile", "Policy Term (Years)", assumptions.term_years),
        ("Policyholder & Contract Profile", "Face Amount ($)", round(assumptions.face_amount, 2)),
        ("Economic & Underwriting Assumptions", "Pricing Mode", pricing_mode),
        ("Economic & Underwriting Assumptions", "Valuation Interest Rate (%)", round(assumptions.valuation_rate * 100, 2)),
        ("Economic & Underwriting Assumptions", "Earned Asset Yield (%)", round(assumptions.earned_rate * 100, 2)),
        ("Economic & Underwriting Assumptions", "Risk Discount / Hurdle Rate (%)", round(assumptions.hurdle_rate * 100, 2)),
        ("Economic & Underwriting Assumptions", "Initial Preferred Mortality Factor", round(assumptions.preferred_discount, 4)),
        ("Economic & Underwriting Assumptions", "Valuation Mortality PfAD (%)", round(assumptions.valuation_mort_pad * 100, 2)),
        ("Economic & Underwriting Assumptions", "Target Profit Margin / Loading (%)", round(assumptions.target_profit_loading * 100, 2)),
        ("Expense Structure", "Year 1 Fixed Acquisition Expense ($)", round(assumptions.acq_expense_fixed, 2)),
        ("Expense Structure", "Year 1 Acquisition Commission (% Prem)", round(assumptions.acq_expense_pct * 100, 2)),
        ("Expense Structure", "Annual Maintenance Expense ($)", round(assumptions.maint_expense_fixed, 2)),
        ("Expense Structure", "Renewal Expense (% Prem)", round(assumptions.maint_expense_pct * 100, 2)),
        ("Expense Structure", "Annual Expense Inflation Rate (%)", round(assumptions.inflation_rate * 100, 2)),
        ("Executive Valuation Results", "Contractual Gross Premium ($)", round(base_res["gross_premium"], 2)),
        ("Executive Valuation Results", "Statutory Net Premium - Single Decrement ($)", round(base_res["net_premium"], 2)),
        ("Executive Valuation Results", "Active Pure Premium - Multi Decrement ($)", round(base_res["active_pure_premium"], 2)),
        ("Executive Valuation Results", "Dynamic Valuation Premium ($)", round(base_res["valuation_premium"], 2)),
        ("Executive Valuation Results", "PV of Future Profits - PVFP ($)", round(base_res["pvfp"], 2)),
        ("Executive Valuation Results", "PV of Gross Premiums ($)", round(base_res["pv_premiums"], 2)),
        ("Executive Valuation Results", "Profit Margin (%)", round(base_res["profit_margin"] * 100, 2)),
        ("Executive Valuation Results", "Internal Rate of Return - IRR (%)", round(base_res["irr"] * 100, 2) if not np.isnan(base_res["irr"]) else np.nan),
        ("Executive Valuation Results", "Peak Net Level Reserve ($)", round(base_res["peak_net_reserve"], 2)),
        ("Executive Valuation Results", "Peak Dynamic Reserve ($)", round(base_res["peak_dynamic_reserve"], 2)),
    ]
    exec_summary_df = pd.DataFrame(summary_rows, columns=["Category", "Parameter / Metric", "Value"])

    # 2. Format Cash Flow Projection schedule with descriptive column names
    schedule_export_df = proj_df.rename(columns={
        "Year": "Policy Year",
        "Attained_Age": "Attained Age",
        "VBT_Base_q_Per_1000": "VBT Base q (per 1,000)",
        "Wearoff_Factor": "Preferred Wear-off Factor",
        "Mortality_q": "Effective Mortality Rate (q_d)",
        "Lapse_q": "Annual Lapse Rate (q_w)",
        "Inforce_Beg": "In-Force (Beginning)",
        "Inforce_End": "In-Force (Ending)",
        "Net_Reserve_Per_Pol": "Net Level Reserve per Policy ($)",
        "Dynamic_Reserve_Per_Pol": "Dynamic Reserve per Policy ($)",
        "Premium_Income": "Gross Premium Income ($)",
        "Expenses": "Policy Expenses ($)",
        "Investment_Income": "Investment Income ($)",
        "Death_Claims": "Expected Death Claims ($)",
        "Reserve_Change": "Reserve Change dV ($)",
        "Net_Profit": "Net Distributable Profit ($)",
    })

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        exec_summary_df.to_excel(writer, sheet_name="Executive Summary", index=False)
        schedule_export_df.to_excel(writer, sheet_name="Cash Flow Schedule", index=False)
        stress_df.to_excel(writer, sheet_name="Stress Testing", index=False)
        cohort_df.to_excel(writer, sheet_name="Cohort Comparison", index=False)

        # Apply readable numerical formatting & auto-adjusted column widths across all sheets
        for sheet_name, ws in writer.sheets.items():
            ws.views.sheetView[0].showGridLines = True
            ws.freeze_panes = "A2"

            headers = [cell.value for cell in ws[1]]
            for col_idx, header in enumerate(headers, start=1):
                col_letter = ws.cell(row=1, column=col_idx).column_letter
                header_str = str(header or "")

                # Apply cell number formatting for data rows
                if sheet_name != "Executive Summary":
                    for row_idx in range(2, ws.max_row + 1):
                        cell = ws.cell(row=row_idx, column=col_idx)
                        if not isinstance(cell.value, (int, float)):
                            continue
                        if "($)" in header_str:
                            cell.number_format = "$#,##0.00;($#,##0.00);\"-\""
                        elif "(%)" in header_str:
                            cell.number_format = "0.00\"%\""
                        elif "per 1,000" in header_str:
                            cell.number_format = "0.00"
                        elif "q_d" in header_str:
                            cell.number_format = "0.00000"
                        elif "In-Force" in header_str:
                            cell.number_format = "0.0000"
                        elif "Factor" in header_str or "q_w" in header_str:
                            cell.number_format = "0.000"
                elif col_idx == 3:
                    for row_idx in range(2, ws.max_row + 1):
                        metric_label = str(ws.cell(row=row_idx, column=2).value or "")
                        cell = ws.cell(row=row_idx, column=3)
                        if isinstance(cell.value, (int, float)):
                            if "($)" in metric_label:
                                cell.number_format = "$#,##0.00;($#,##0.00);\"-\""
                            elif "(%)" in metric_label or "Prem)" in metric_label:
                                cell.number_format = "0.00\"%\""
                            elif "Factor" in metric_label:
                                cell.number_format = "0.0000"

                # Auto-fit column width
                max_len = max(
                    [len(str(ws.cell(row=r, column=col_idx).value or "")) for r in range(1, ws.max_row + 1)]
                    + [len(header_str)]
                )
                ws.column_dimensions[col_letter].width = min(max(max_len + 4, 14), 42)

    output.seek(0)
    return output.getvalue()


def main():
    st.title("Term Life Insurance Pricing & Dynamic Reserve Engine")
    st.caption(
        "Actuarial Valuation, Cash Flow Projection, and Sensitivity Workbench | "
        "Mortality Basis: SOA 2015 VBT Smoker-Distinct (ANB/ALB) & Preferred Wear-off Factors"
    )

    try:
        loader = load_cached_vbt_data()
    except Exception as e:
        st.error(f"Error loading SOA 2015 VBT Excel workbooks: {e}")
        st.stop()

    # --------------------------------------------------------------------------
    # Sidebar Controls
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.subheader("Display & Theme Settings")
        canvas_mode = st.selectbox(
            "Chart Canvas Mode",
            [
                "Auto (Light & Dark Adaptive)",
                "Deep Navy Canvas (Dark)",
                "Warm Ivory Canvas (Light)"
            ],
            index=0,
            help=(
                "Auto adapts dynamically to Streamlit Light/Dark mode. "
                "Deep Navy Canvas and Warm Ivory Canvas lock the chart surface to the "
                "exact institutional Deep Navy (#0B192C) or Warm Ivory (#FAF8F5) background."
            )
        )

        st.divider()
        st.subheader("1. Policyholder & Table Basis")
        issue_age = st.slider("Issue Age (x)", min_value=18, max_value=75, value=45, step=1)
        col_s1, col_s2 = st.columns(2)
        gender = col_s1.selectbox("Gender", ["Male", "Female"], index=0)
        smoker_status = col_s2.selectbox("Smoker Status", ["Non-Smoker", "Smoker"], index=0)
        col_s3, col_s4 = st.columns(2)
        age_basis = col_s3.selectbox("Age Basis", ["ANB", "ALB"], index=0)
        term_years = col_s4.selectbox("Term (Years)", [10, 15, 20, 25], index=2)
        face_amount = st.number_input(
            "Face Amount ($)", min_value=50_000.0, max_value=10_000_000.0, value=500_000.0, step=50_000.0
        )

        st.divider()
        st.subheader("2. Economic & Valuation Assumptions")
        valuation_rate = st.slider("Valuation Interest Rate (%)", 1.0, 7.0, 3.5, 0.25) / 100.0
        earned_rate = st.slider("Earned Asset Yield (%)", 1.0, 8.0, 4.5, 0.25) / 100.0
        hurdle_rate = st.slider("Risk Discount / Hurdle Rate (%)", 4.0, 15.0, 8.0, 0.50) / 100.0
        preferred_discount = st.slider("Initial Preferred Mortality Factor", 0.50, 1.00, 0.75, 0.05)
        valuation_mort_pad = st.slider("Valuation Mortality PfAD (%)", 0.0, 25.0, 10.0, 1.0) / 100.0

        st.divider()
        st.subheader("3. Expense Structure & Pricing Mode")
        pricing_mode = st.radio(
            "Gross Premium Calculation Mode",
            ["EPV Loading Formula", "Goal-Seek Exact Target Margin"],
            index=0
        )
        target_profit_loading = st.slider("Target Profit Margin / Loading (%)", 2.0, 25.0, 12.0, 0.5) / 100.0
        acq_expense_fixed = st.number_input("Year 1 Fixed Acquisition Expense ($)", 0.0, 2000.0, 300.0, 25.0)
        acq_expense_pct = st.slider("Year 1 Acquisition Commission (% Prem)", 0.0, 100.0, 55.0, 5.0) / 100.0
        maint_expense_fixed = st.number_input("Annual Maintenance Expense ($)", 0.0, 500.0, 55.0, 5.0)
        maint_expense_pct = st.slider("Renewal Expense (% Prem)", 0.0, 20.0, 4.0, 0.5) / 100.0

    pal = get_theme_palette(canvas_mode)

    assumptions = ActuarialAssumptions(
        issue_age=issue_age,
        gender=gender,
        smoker_status=smoker_status,
        age_basis=age_basis,
        term_years=term_years,
        face_amount=face_amount,
        valuation_rate=valuation_rate,
        earned_rate=earned_rate,
        hurdle_rate=hurdle_rate,
        preferred_discount=preferred_discount,
        acq_expense_fixed=acq_expense_fixed,
        acq_expense_pct=acq_expense_pct,
        maint_expense_fixed=maint_expense_fixed,
        maint_expense_pct=maint_expense_pct,
        target_profit_loading=target_profit_loading,
        valuation_mort_pad=valuation_mort_pad
    )

    engine = TermLifeEngine(assumptions, loader)
    use_goal_seek = (pricing_mode == "Goal-Seek Exact Target Margin")
    locked_gp = engine.solve_goal_seek_gross_premium(target_profit_loading) if use_goal_seek else None

    base_res = engine.run_valuation(fixed_gross_premium=locked_gp)
    sim_engine = PricingSimulationEngine(engine)
    stress_df, scenario_dfs = sim_engine.run_stress_suite(locked_gp=base_res["gross_premium"])
    cohort_df = sim_engine.compare_all_vbt_cohorts(use_goal_seek=use_goal_seek)
    proj_df = base_res["projection_df"]

    # --------------------------------------------------------------------------
    # Prominent Excel Report Export & Executive Summary Header
    # --------------------------------------------------------------------------
    excel_bytes = generate_excel_report(
        assumptions=assumptions,
        pricing_mode=pricing_mode,
        base_res=base_res,
        proj_df=proj_df,
        stress_df=stress_df,
        cohort_df=cohort_df
    )
    excel_filename = f"Actuarial_Report_{base_res['table_used'].replace(' ', '_')}_Age{issue_age}.xlsx"

    hdr_col, dl_col = st.columns([3.2, 1.3])
    with hdr_col:
        st.markdown(f"#### Executive Valuation Summary — Mortality Table: `{base_res['table_used']}`")
    with dl_col:
        st.download_button(
            label="Download Actuarial Report (Excel)",
            data=excel_bytes,
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(
        "Contract Gross Premium",
        f"${base_res['gross_premium']:,.2f}",
        f"Basis: {'Goal-Seek' if use_goal_seek else 'EPV Formula'}"
    )
    k2.metric(
        "Statutory Net Premium",
        f"${base_res['net_premium']:,.2f}",
        f"Active Pure: ${base_res['active_pure_premium']:,.2f}"
    )
    k3.metric(
        "PV of Future Profits (PVFP)",
        f"${base_res['pvfp']:,.2f}",
        f"Discount Rate: {hurdle_rate*100:.1f}%"
    )
    k4.metric(
        "Profit Margin (%)",
        f"{base_res['profit_margin']*100:.2f}%",
        f"{(base_res['profit_margin'] - target_profit_loading)*100:+.2f}%p vs Target"
    )
    k5.metric(
        "Internal Rate of Return",
        f"{base_res['irr']*100:.2f}%" if not np.isnan(base_res["irr"]) else "N/A",
        f"{(base_res['irr'] - hurdle_rate)*100:+.2f}%p vs Hurdle" if not np.isnan(base_res["irr"]) else "N/A"
    )
    k6.metric(
        "Peak Dynamic Reserve",
        f"${base_res['peak_dynamic_reserve']:,.2f}",
        f"Net Peak: ${base_res['peak_net_reserve']:,.2f}"
    )

    st.divider()

    # --------------------------------------------------------------------------
    # Analytical Navigation Tabs
    # --------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Reserve & Mortality Trajectory",
        "2. Cash Flow & Profit Emergence",
        "3. Stress Testing & Sensitivity",
        "4. Cohort Valuation Comparison"
    ])

    # === TAB 1: RESERVE & MORTALITY TRAJECTORY ===
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig_res = make_subplots(specs=[[{"secondary_y": True}]])
            fig_res.add_trace(
                go.Scatter(
                    x=proj_df["Year"],
                    y=proj_df["Net_Reserve_Per_Pol"],
                    mode="lines+markers",
                    name="Net Level Reserve (Single-Dec)",
                    line=dict(color=pal["navy_primary"], width=2.8),
                    marker=dict(size=6, symbol="square", color="#FAF8F5", line=dict(color=pal["navy_primary"], width=2))
                ),
                secondary_y=False
            )
            fig_res.add_trace(
                go.Scatter(
                    x=proj_df["Year"],
                    y=proj_df["Dynamic_Reserve_Per_Pol"],
                    mode="lines+markers",
                    name="Dynamic Reserve (Multi-Dec + PfAD)",
                    line=dict(color=pal["teal_primary"], width=3.0),
                    marker=dict(size=7, symbol="circle", color=pal["teal_primary"], line=dict(color="#0B192C", width=1.2))
                ),
                secondary_y=False
            )
            fig_res.add_trace(
                go.Scatter(
                    x=proj_df["Year"],
                    y=proj_df["Inforce_Beg"] * 100,
                    mode="lines",
                    fill="tozeroy",
                    name="Active In-Force Survival (%)",
                    line=dict(color=pal["ivory_bar"], width=1.8),
                    fillcolor=pal["ivory_fill"]
                ),
                secondary_y=True
            )
            fig_res = apply_institutional_chart_layout(
                fig_res, "Prospective Net Level vs. Dynamic Reserve Trajectory ($)", pal, height=470, bottom_margin=105
            )
            fig_res.update_xaxes(title_text="Policy Duration (Year)")
            fig_res.update_yaxes(title_text="Reserve per Active Policy ($)", secondary_y=False)
            fig_res.update_yaxes(title_text="Active In-Force (%)", range=[0, 105], showgrid=False, secondary_y=True)
            render_chart(fig_res, pal)

        with c2:
            no_wear_df = scenario_dfs["7. No Preferred Wear-off"]
            fig_mort = make_subplots(specs=[[{"secondary_y": True}]])
            fig_mort.add_trace(
                go.Scatter(
                    x=proj_df["Year"],
                    y=proj_df["VBT_Base_q_Per_1000"],
                    mode="lines",
                    name="2015 VBT Standard Rate (per 1,000)",
                    line=dict(color=pal["slate_muted"], dash="dot", width=2.3)
                ),
                secondary_y=False
            )
            fig_mort.add_trace(
                go.Scatter(
                    x=proj_df["Year"],
                    y=proj_df["Mortality_q"] * 1000,
                    mode="lines+markers",
                    name="Effective Preferred Rate (With Wear-off)",
                    line=dict(color=pal["navy_primary"], width=2.8),
                    marker=dict(size=6, color="#FAF8F5", line=dict(color=pal["navy_primary"], width=2))
                ),
                secondary_y=False
            )
            fig_mort.add_trace(
                go.Scatter(
                    x=no_wear_df["Year"],
                    y=no_wear_df["Mortality_q"] * 1000,
                    mode="lines",
                    name="Constant Preferred (No Wear-off)",
                    line=dict(color=pal["teal_primary"], dash="dash", width=2.6)
                ),
                secondary_y=False
            )
            fig_mort.add_trace(
                go.Bar(
                    x=proj_df["Year"],
                    y=proj_df["Wearoff_Factor"] * 100,
                    name="2015 VBT Wear-off Factor (%)",
                    marker=dict(
                        color=pal["ivory_bar"],
                        opacity=0.45,
                        line=dict(color=pal["teal_primary"], width=1.2)
                    )
                ),
                secondary_y=True
            )
            fig_mort = apply_institutional_chart_layout(
                fig_mort, "SOA 2015 VBT Select Mortality Curve (per 1,000) & Preferred Wear-off", pal, height=470, bottom_margin=105
            )
            fig_mort.update_xaxes(title_text="Policy Duration (Year)")
            fig_mort.update_yaxes(title_text="Mortality Rate per 1,000", secondary_y=False)
            fig_mort.update_yaxes(title_text="Wear-off Factor (%)", showgrid=False, secondary_y=True)
            render_chart(fig_mort, pal)

    # === TAB 2: CASH FLOW & PROFIT EMERGENCE ===
    with tab2:
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(
            x=proj_df["Year"], y=proj_df["Premium_Income"],
            name="Gross Premium Income (+)",
            marker=dict(color=pal["navy_primary"], line=dict(color="#0B192C", width=0.6))
        ))
        fig_cf.add_trace(go.Bar(
            x=proj_df["Year"], y=proj_df["Investment_Income"],
            name="Investment Income (+)",
            marker=dict(color=pal["teal_primary"], line=dict(color="#0B192C", width=0.6))
        ))
        fig_cf.add_trace(go.Bar(
            x=proj_df["Year"], y=-proj_df["Death_Claims"],
            name="Expected Death Claims (-)",
            marker=dict(color=pal["coral_alert"])
        ))
        fig_cf.add_trace(go.Bar(
            x=proj_df["Year"], y=-proj_df["Expenses"],
            name="Policy Expenses (-)",
            marker=dict(color=pal["ivory_bar"], line=dict(color=pal["teal_secondary"], width=1.0))
        ))
        fig_cf.add_trace(go.Bar(
            x=proj_df["Year"], y=-proj_df["Reserve_Change"],
            name="Reserve Change -dV (-)",
            marker=dict(color=pal["slate_muted"])
        ))
        fig_cf.add_trace(go.Scatter(
            x=proj_df["Year"], y=proj_df["Net_Profit"],
            mode="lines+markers",
            name="Net Distributable Profit ($)",
            line=dict(color=pal["amber_accent"], width=3.2),
            marker=dict(size=7, color="#FAF8F5", line=dict(color="#0B192C", width=2.0))
        ))
        fig_cf = apply_institutional_chart_layout(
            fig_cf,
            "Annual Cash Flow Decomposition & Distributable Profit Vector (per Issued Policy)",
            pal,
            height=490,
            bottom_margin=110
        )
        fig_cf.update_layout(barmode="relative")
        fig_cf.update_xaxes(title_text="Policy Duration (Year)")
        fig_cf.update_yaxes(title_text="Annual Cash Flow ($)")
        render_chart(fig_cf, pal)

        st.markdown("##### Actuarial Cash Flow & Reserve Schedule")
        st.dataframe(proj_df.style.format({
            "VBT_Base_q_Per_1000": "{:.2f}",
            "Wearoff_Factor": "{:.3f}",
            "Mortality_q": "{:.5f}",
            "Lapse_q": "{:.3f}",
            "Inforce_Beg": "{:.4f}",
            "Inforce_End": "{:.4f}",
            "Net_Reserve_Per_Pol": "${:,.2f}",
            "Dynamic_Reserve_Per_Pol": "${:,.2f}",
            "Premium_Income": "${:,.2f}",
            "Expenses": "${:,.2f}",
            "Investment_Income": "${:,.2f}",
            "Death_Claims": "${:,.2f}",
            "Reserve_Change": "${:,.2f}",
            "Net_Profit": "${:,.2f}",
        }), use_container_width=True)

        csv_bytes = proj_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export Cash Flow Schedule (CSV)",
            data=csv_bytes,
            file_name=f"term_life_projection_{base_res['table_used'].replace(' ', '_')}_age{issue_age}.csv",
            mime="text/csv"
        )

    # === TAB 3: STRESS TESTING & SENSITIVITY ===
    with tab3:
        st.markdown("##### Deterministic Sensitivity & Shock Analysis (Contractual Gross Premium Locked)")
        st.dataframe(stress_df.style.format({
            "Net_Premium ($)": "${:,.2f}",
            "Active_Pure_Prem ($)": "${:,.2f}",
            "Locked_Gross_Prem ($)": "${:,.2f}",
            "PVFP ($)": "${:,.2f}",
            "Profit_Margin (%)": "{:.2f}%",
            "IRR (%)": "{:.2f}%",
            "Peak_Net_Reserve ($)": "${:,.2f}",
            "Peak_Dynamic_Reserve ($)": "${:,.2f}",
        }), use_container_width=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            fig_stress_res = go.Figure()
            s_palette = pal["stress_palette"]
            for idx, (scen_name, df_s) in enumerate(scenario_dfs.items()):
                fig_stress_res.add_trace(go.Scatter(
                    x=df_s["Year"],
                    y=df_s["Dynamic_Reserve_Per_Pol"],
                    mode="lines",
                    name=scen_name,
                    line=dict(width=2.6 if idx == 0 else 2.1, color=s_palette[idx % len(s_palette)])
                ))
            fig_stress_res = apply_institutional_chart_layout(
                fig_stress_res,
                "Dynamic Reserve Sensitivity Across 7 Stress Scenarios ($)",
                pal,
                height=480,
                bottom_margin=130,
                legend_y=-0.23
            )
            fig_stress_res.update_xaxes(title_text="Policy Duration (Year)")
            fig_stress_res.update_yaxes(title_text="Dynamic Reserve per Active Policy ($)")
            render_chart(fig_stress_res, pal)

        with sc2:
            bar_colors = [pal["teal_primary"] if m >= 0 else pal["coral_alert"] for m in stress_df["Profit_Margin (%)"]]
            border_colors = ["#FAF8F5" if m >= 0 else "#0B192C" for m in stress_df["Profit_Margin (%)"]]
            fig_margin = go.Figure(go.Bar(
                x=stress_df["Profit_Margin (%)"],
                y=stress_df["Scenario"],
                orientation="h",
                name="Profit Margin (%)",
                marker=dict(color=bar_colors, line=dict(color=border_colors, width=1.0)),
                text=[f"{m:.2f}%" for m in stress_df["Profit_Margin (%)"]],
                textposition="auto"
            ))
            fig_margin = apply_institutional_chart_layout(
                fig_margin,
                "Profit Margin (%) Impact Under Shock Scenarios",
                pal,
                height=480,
                bottom_margin=85
            )
            fig_margin.update_layout(showlegend=False)
            fig_margin.update_xaxes(title_text="Profit Margin (% of PV Gross Premiums)")
            fig_margin.update_yaxes(autorange="reversed", showgrid=False)
            render_chart(fig_margin, pal)

    # === TAB 4: COHORT VALUATION COMPARISON ===
    with tab4:
        st.markdown(f"##### Cross-Sectional Valuation Across All 8 SOA 2015 VBT Tables (Issue Age {issue_age})")
        st.dataframe(cohort_df.style.format({
            "Net_Premium ($)": "${:,.2f}",
            "Active_Pure_Prem ($)": "${:,.2f}",
            "Gross_Premium ($)": "${:,.2f}",
            "PVFP ($)": "${:,.2f}",
            "Profit_Margin (%)": "{:.2f}%",
            "IRR (%)": "{:.2f}%",
            "Peak_Net_Reserve ($)": "${:,.2f}",
            "Peak_Dynamic_Reserve ($)": "${:,.2f}",
        }), use_container_width=True)

        fig_cohort = go.Figure()
        fig_cohort.add_trace(go.Bar(
            x=cohort_df["VBT_Sheet"],
            y=cohort_df["Active_Pure_Prem ($)"],
            name="Active Pure Premium (Multi-Decrement)",
            marker=dict(color=pal["teal_primary"], line=dict(color="#0B192C", width=0.8))
        ))
        fig_cohort.add_trace(go.Bar(
            x=cohort_df["VBT_Sheet"],
            y=cohort_df["Net_Premium ($)"],
            name="Statutory Net Premium (Single-Decrement)",
            marker=dict(color=pal["ivory_bar"], line=dict(color=pal["teal_secondary"], width=1.4))
        ))
        fig_cohort.add_trace(go.Bar(
            x=cohort_df["VBT_Sheet"],
            y=cohort_df["Gross_Premium ($)"],
            name="Contractual Gross Premium",
            marker=dict(color=pal["navy_primary"], line=dict(color="#FAF8F5", width=0.8))
        ))
        fig_cohort = apply_institutional_chart_layout(
            fig_cohort,
            f"Annual Premium Structure Across All 8 SOA 2015 VBT Tables (Issue Age {issue_age})",
            pal,
            height=470,
            bottom_margin=105
        )
        fig_cohort.update_layout(barmode="group")
        fig_cohort.update_xaxes(title_text="SOA 2015 VBT Table")
        fig_cohort.update_yaxes(title_text="Annual Premium ($)")
        render_chart(fig_cohort, pal)


if __name__ == "__main__":
    main()