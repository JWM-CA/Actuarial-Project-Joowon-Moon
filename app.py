import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ==============================================================================
# PHASE 1: DATA LOADING & ACTUARIAL ASSUMPTIONS (DUAL SOA 2015 VBT WORKBOOKS)
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
    """Loads both SOA 2015 VBT Excel workbooks (Smoker-Distinct ALB/ANB & Preferred Wear-off)."""

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
# PHASE 4: STREAMLIT INTERACTIVE WEB APPLICATION & PLOTLY DASHBOARD
# ==============================================================================

st.set_page_config(
    page_title="SOA 2015 VBT Term Life Pricing & Dynamic Reserve Engine",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def load_cached_vbt_data() -> SOA2015VBTDataLoader:
    return SOA2015VBTDataLoader(
        vbt_filepath="2015-vbt-smoker-distinct-alb-anb.xlsx",
        wearoff_filepath="2015-vbt-preferred-wearoff-factors_2.xlsx"
    )


def main():
    st.title("📊 20-Year Term Life Insurance Pricing & Dynamic Reserve Engine")
    st.markdown(
        "**Actuarial Valuation & Stress-Testing Suite** powered by official **SOA 2015 VBT Smoker-Distinct (ANB/ALB)** "
        "select mortality tables and **SOA 2015 VBT Preferred Wear-off Factors**."
    )

    try:
        loader = load_cached_vbt_data()
    except Exception as e:
        st.error(f"Error loading SOA 2015 VBT Excel files: {e}")
        st.stop()

    # --------------------------------------------------------------------------
    # Sidebar: Interactive Policyholder, Economic, Expense, and Pricing Controls
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ 1. Policyholder & Table Basis")
        issue_age = st.slider("Issue Age (x)", min_value=18, max_value=75, value=45, step=1)
        col_s1, col_s2 = st.columns(2)
        gender = col_s1.selectbox("Gender", ["Male", "Female"], index=0)
        smoker_status = col_s2.selectbox("Smoker Status", ["Non-Smoker", "Smoker"], index=0)
        col_s3, col_s4 = st.columns(2)
        age_basis = col_s3.selectbox("Age Basis", ["ANB", "ALB"], index=0)
        term_years = col_s4.selectbox("Term (Years)", [10, 15, 20, 25], index=2)
        face_amount = st.number_input("Face Amount ($)", min_value=50_000.0, max_value=10_000_000.0, value=500_000.0, step=50_000.0)

        st.header("📈 2. Economic & Underwriting Rates")
        valuation_rate = st.slider("Valuation Interest Rate (%)", 1.0, 7.0, 3.5, 0.25) / 100.0
        earned_rate = st.slider("Earned Asset Yield (%)", 1.0, 8.0, 4.5, 0.25) / 100.0
        hurdle_rate = st.slider("Risk Discount / Hurdle Rate (%)", 4.0, 15.0, 8.0, 0.50) / 100.0
        preferred_discount = st.slider("Initial Preferred Mortality Ratio", 0.50, 1.00, 0.75, 0.05)
        valuation_mort_pad = st.slider("Dynamic Reserve Mortality PfAD (%)", 0.0, 25.0, 10.0, 1.0) / 100.0

        st.header("💼 3. Expenses & Pricing Methodology")
        pricing_mode = st.radio(
            "Gross Premium Calculation Mode",
            ["EPV Loading Formula", "Goal-Seek Exact Target Margin"],
            index=0,
            help="EPV Loading adds target margin before reserve strain. Goal-Seek solves for the exact Gross Premium achieving Target Profit Margin after reserve strain."
        )
        target_profit_loading = st.slider("Target Profit Margin / Loading (%)", 2.0, 25.0, 12.0, 0.5) / 100.0
        acq_expense_fixed = st.number_input("Year 1 Fixed Acquisition Cost ($)", 0.0, 2000.0, 300.0, 25.0)
        acq_expense_pct = st.slider("Year 1 Acquisition Commission (% Prem)", 0.0, 100.0, 55.0, 5.0) / 100.0
        maint_expense_fixed = st.number_input("Annual Maintenance Expense ($)", 0.0, 500.0, 55.0, 5.0)
        maint_expense_pct = st.slider("Renewal Expense (% Prem)", 0.0, 20.0, 4.0, 0.5) / 100.0

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
    # Top Executive KPI Metrics Row
    # --------------------------------------------------------------------------
    st.subheader(f"📌 Executive Pricing Summary — Table Used: `{base_res['table_used']}`")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Contract Gross Premium", f"${base_res['gross_premium']:,.2f}", f"Mode: {'Goal-Seek' if use_goal_seek else 'EPV Load'}")
    k2.metric("Statutory Net Premium", f"${base_res['net_premium']:,.2f}", f"Active Pure: ${base_res['active_pure_premium']:,.2f}")
    k3.metric("PV of Future Profits (PVFP)", f"${base_res['pvfp']:,.2f}", f"Hurdle: {hurdle_rate*100:.1f}%")
    k4.metric("Profit Margin (%)", f"{base_res['profit_margin']*100:.2f}%", f"{(base_res['profit_margin'] - target_profit_loading)*100:+.2f}%p vs Target")
    k5.metric("Internal Rate of Return (IRR)", f"{base_res['irr']*100:.2f}%" if not np.isnan(base_res["irr"]) else "N/A", f"Spread: {(base_res['irr']-hurdle_rate)*100:+.2f}%p")
    k6.metric("Peak Dynamic Reserve", f"${base_res['peak_dynamic_reserve']:,.2f}", f"Net Peak: ${base_res['peak_net_reserve']:,.2f}")

    st.divider()

    # --------------------------------------------------------------------------
    # Interactive Analytical Tabs
    # --------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Tab 1: Reserve & Mortality Trajectory",
        "💵 Tab 2: Cash Flow & Profit Emergence",
        "⚡ Tab 3: 7-Scenario Stress Testing",
        "🧬 Tab 4: 8 SOA 2015 VBT Cohort Comparison"
    ])

    # === TAB 1: RESERVE & MORTALITY TRAJECTORY ===
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig_res = make_subplots(specs=[[{"secondary_y": True}]])
            fig_res.add_trace(
                go.Scatter(x=proj_df["Year"], y=proj_df["Net_Reserve_Per_Pol"], mode="lines+markers",
                           name="Net Level Reserve (V^Net)", line=dict(color="#2ca02c", width=3)),
                secondary_y=False
            )
            fig_res.add_trace(
                go.Scatter(x=proj_df["Year"], y=proj_df["Dynamic_Reserve_Per_Pol"], mode="lines+markers",
                           name="Dynamic Reserve (V^Dyn with PfAD)", line=dict(color="#d62728", width=3)),
                secondary_y=False
            )
            fig_res.add_trace(
                go.Scatter(x=proj_df["Year"], y=proj_df["Inforce_Beg"] * 100, mode="lines",
                           fill="tozeroy", name="Active In-Force Survival (%)",
                           line=dict(color="rgba(31, 119, 180, 0.35)", width=1.5)),
                secondary_y=True
            )
            fig_res.update_layout(
                title="Prospective Net Level vs. Dynamic Reserve Trajectory ($)",
                xaxis_title="Policy Duration (Year)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                height=440
            )
            fig_res.update_yaxes(title_text="Reserve per Active Policy ($)", secondary_y=False)
            fig_res.update_yaxes(title_text="Active In-Force (%)", range=[0, 105], secondary_y=True)
            st.plotly_chart(fig_res, use_container_width=True)

        with c2:
            no_wear_df = scenario_dfs["7. No Preferred Wear-off"]
            fig_mort = make_subplots(specs=[[{"secondary_y": True}]])
            fig_mort.add_trace(
                go.Scatter(x=proj_df["Year"], y=proj_df["VBT_Base_q_Per_1000"], mode="lines",
                           name="SOA 2015 VBT Base Rate (per 1,000)", line=dict(color="#7f7f7f", dash="dot", width=2.5)),
                secondary_y=False
            )
            fig_mort.add_trace(
                go.Scatter(x=proj_df["Year"], y=proj_df["Mortality_q"] * 1000, mode="lines+markers",
                           name="Effective Preferred Rate (With Wear-off)", line=dict(color="#1f77b4", width=3)),
                secondary_y=False
            )
            fig_mort.add_trace(
                go.Scatter(x=no_wear_df["Year"], y=no_wear_df["Mortality_q"] * 1000, mode="lines",
                           name="Constant Preferred (No Wear-off)", line=dict(color="#ff7f0e", dash="dash", width=2.5)),
                secondary_y=False
            )
            fig_mort.add_trace(
                go.Bar(x=proj_df["Year"], y=proj_df["Wearoff_Factor"] * 100,
                       name="2015 VBT Wear-off Factor (%)", marker_color="rgba(44, 160, 44, 0.28)"),
                secondary_y=True
            )
            fig_mort.update_layout(
                title="SOA 2015 VBT Mortality Curve (per 1,000) & Preferred Wear-off",
                xaxis_title="Policy Duration (Year)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                height=440
            )
            fig_mort.update_yaxes(title_text="Mortality Rate per 1,000", secondary_y=False)
            fig_mort.update_yaxes(title_text="Wear-off Factor (%)", secondary_y=True)
            st.plotly_chart(fig_mort, use_container_width=True)

    # === TAB 2: CASH FLOW & PROFIT EMERGENCE ===
    with tab2:
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(x=proj_df["Year"], y=proj_df["Premium_Income"], name="Gross Premium Income", marker_color="#4c78a8"))
        fig_cf.add_trace(go.Bar(x=proj_df["Year"], y=proj_df["Investment_Income"], name="Investment Income", marker_color="#72b7b2"))
        fig_cf.add_trace(go.Bar(x=proj_df["Year"], y=-proj_df["Death_Claims"], name="Expected Death Claims (-)", marker_color="#e45756"))
        fig_cf.add_trace(go.Bar(x=proj_df["Year"], y=-proj_df["Expenses"], name="Policy Expenses (-)", marker_color="#f58518"))
        fig_cf.add_trace(go.Bar(x=proj_df["Year"], y=-proj_df["Reserve_Change"], name="Reserve Increase -dV (-)", marker_color="#b279a2"))
        fig_cf.add_trace(go.Scatter(
            x=proj_df["Year"], y=proj_df["Net_Profit"], mode="lines+markers",
            name="Net Distributable Profit ($)", line=dict(color="#000000", width=3.5)
        ))
        fig_cf.update_layout(
            barmode="relative",
            title="Annual Cash Flow Decomposition & Distributable Profit Vector (per Initial Policy Issued)",
            xaxis_title="Policy Duration (Year)",
            yaxis_title="Cash Flow Amount ($)",
            height=460
        )
        st.plotly_chart(fig_cf, use_container_width=True)

        st.markdown("#### 📋 20-Year Actuarial Cash Flow & Reserve Schedule")
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
            label="📥 Download 20-Year Cash Flow Projection (CSV)",
            data=csv_bytes,
            file_name=f"term_life_projection_{base_res['table_used'].replace(' ', '_')}_age{issue_age}.csv",
            mime="text/csv"
        )

    # === TAB 3: 7-SCENARIO STRESS TESTING ===
    with tab3:
        st.markdown("#### ⚡ Phase 3A: Sensitivity & Macro/Biometric Stress Test Summary (Locked-In Gross Premium)")
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
            for scen_name, df_s in scenario_dfs.items():
                fig_stress_res.add_trace(go.Scatter(
                    x=df_s["Year"], y=df_s["Dynamic_Reserve_Per_Pol"],
                    mode="lines", name=scen_name, line=dict(width=2.5)
                ))
            fig_stress_res.update_layout(
                title="Dynamic Reserve Sensitivity Across 7 Stress Scenarios ($)",
                xaxis_title="Policy Duration (Year)",
                yaxis_title="Dynamic Reserve per Active Policy ($)",
                height=420
            )
            st.plotly_chart(fig_stress_res, use_container_width=True)

        with sc2:
            bar_colors = ["#1f77b4" if m >= 0 else "#d62728" for m in stress_df["Profit_Margin (%)"]]
            fig_margin = go.Figure(go.Bar(
                x=stress_df["Profit_Margin (%)"],
                y=stress_df["Scenario"],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{m:.2f}%" for m in stress_df["Profit_Margin (%)"]],
                textposition="auto"
            ))
            fig_margin.update_layout(
                title="Profit Margin (%) Impact Under Shock Scenarios",
                xaxis_title="Profit Margin (% of PV Gross Premiums)",
                yaxis=dict(autorange="reversed"),
                height=420
            )
            st.plotly_chart(fig_margin, use_container_width=True)

    # === TAB 4: 8 SOA 2015 VBT COHORT COMPARISON ===
    with tab4:
        st.markdown(f"#### 🧬 Phase 3B: Cross-Sectional Comparison Across All 8 SOA 2015 VBT Tables (Issue Age {issue_age})")
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
        fig_cohort.add_trace(go.Bar(x=cohort_df["VBT_Sheet"], y=cohort_df["Active_Pure_Prem ($)"], name="Active Pure Prem (Multi-Decrement)", marker_color="#72b7b2"))
        fig_cohort.add_trace(go.Bar(x=cohort_df["VBT_Sheet"], y=cohort_df["Net_Premium ($)"], name="Statutory Net Prem (Single-Decrement)", marker_color="#f58518"))
        fig_cohort.add_trace(go.Bar(x=cohort_df["VBT_Sheet"], y=cohort_df["Gross_Premium ($)"], name="Contractual Gross Premium", marker_color="#4c78a8"))
        fig_cohort.update_layout(
            barmode="group",
            title=f"Premium Structure Across All 8 SOA 2015 VBT Tables (Issue Age {issue_age})",
            xaxis_title="SOA 2015 VBT Table",
            yaxis_title="Annual Premium ($)",
            height=430
        )
        st.plotly_chart(fig_cohort, use_container_width=True)


if __name__ == "__main__":
    main()