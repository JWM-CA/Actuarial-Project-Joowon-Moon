import os
import numpy as np
import pandas as pd
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
    """
    Loads both SOA 2015 VBT Excel workbooks directly:
    1) 2015-vbt-smoker-distinct-alb-anb.xlsx (8 sheets: MNS/FNS/MSM/FSM x ANB/ALB, per 1,000)
    2) 2015-vbt-preferred-wearoff-factors_2.xlsx (1 sheet: 'Wearoff Factors', ages 0-99)
    """

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
        # Fallback if the operating system saved the duplicate file under its base name
        if not os.path.exists(wearoff_filepath) and os.path.exists("2015-vbt-preferred-wearoff-factors.xlsx"):
            self.wearoff_filepath = "2015-vbt-preferred-wearoff-factors.xlsx"
        else:
            self.wearoff_filepath = wearoff_filepath

        self.vbt_tables = self._load_all_vbt_sheets()
        self.wearoff_df = self._load_wearoff_table()

    def _load_all_vbt_sheets(self) -> Dict[str, pd.DataFrame]:
        """Reads all 8 smoker-distinct ANB/ALB sheets from the SOA 2015 VBT workbook."""
        xls = pd.ExcelFile(self.vbt_filepath)
        tables = {}
        for sheet in xls.sheet_names:
            df = pd.read_excel(self.vbt_filepath, sheet_name=sheet, skiprows=2)
            df = df.dropna(subset=["Iss. Age"]).copy()
            df["Iss. Age"] = df["Iss. Age"].astype(int)
            df = df.set_index("Iss. Age")
            tables[sheet] = df
        return tables

    def _load_wearoff_table(self) -> pd.DataFrame:
        """Reads the 'Wearoff Factors' sheet from the SOA 2015 VBT Wearoff workbook."""
        xls = pd.ExcelFile(self.wearoff_filepath)
        sheet_name = xls.sheet_names[0]  # 'Wearoff Factors'
        df = pd.read_excel(self.wearoff_filepath, sheet_name=sheet_name, skiprows=2)
        df = df.dropna(subset=["Issue Age"]).copy()
        df["Issue Age"] = df["Issue Age"].astype(int)
        df = df.set_index("Issue Age")
        return df

    def get_sheet_name(self, gender: str, smoker_status: str, age_basis: str) -> str:
        key = (gender.strip().lower(), smoker_status.strip().lower(), age_basis.strip().lower())
        if key not in self.SHEET_MAP:
            raise ValueError(f"Invalid combination {key}. Valid options: Male/Female, Non-Smoker/Smoker, ANB/ALB.")
        return self.SHEET_MAP[key]

    def get_base_select_mortality(self, assumptions: ActuarialAssumptions) -> np.ndarray:
        """
        Extracts exact SOA 2015 VBT Select mortality rates q_{[x]+t-1}^{Base}
        (converting per-1,000 table values to decimal probabilities) for t = 1..term_years.
        """
        sheet_name = self.get_sheet_name(assumptions.gender, assumptions.smoker_status, assumptions.age_basis)
        df = self.vbt_tables[sheet_name]

        if assumptions.issue_age not in df.index:
            raise ValueError(f"Issue Age {assumptions.issue_age} not found in {sheet_name} (valid range: 18-95).")

        row = df.loc[assumptions.issue_age]
        rates_per_1000 = []
        for d in range(1, assumptions.term_years + 1):
            col = d if d <= 25 else "Ult."
            rates_per_1000.append(float(row[col]))

        return np.clip(np.array(rates_per_1000, dtype=float) / 1000.0, 0.0, 1.0)

    def get_wearoff_factors(self, issue_age: int, term_years: int) -> np.ndarray:
        """Extracts durational wear-off factors w_{[x]+t-1} for t = 1..term_years."""
        if issue_age not in self.wearoff_df.index:
            raise ValueError(f"Issue Age {issue_age} not found in Wearoff table (valid range: 0-99).")
        row = self.wearoff_df.loc[issue_age]
        factors = []
        for d in range(1, term_years + 1):
            col = d if d <= 25 else "Ultimate"
            factors.append(float(row[col]))
        return np.array(factors, dtype=float)

    def get_effective_mortality(
        self,
        assumptions: ActuarialAssumptions,
        mortality_mult: float = 1.0,
        apply_wearoff: bool = True
    ) -> pd.DataFrame:
        """
        Combines SOA 2015 VBT base select mortality and SOA 2015 VBT Preferred Wear-off Factors:
        q_{[x]+t-1}^{Pref} = q_{[x]+t-1}^{Base} * [theta + w_{[x]+t-1} * (1 - theta)]
        """
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
        """Returns annual policy lapse rates q_{[x]+t-1}^{(w)} across the policy term."""
        base_lapse = np.array(
            [0.08, 0.07, 0.06] + [0.045] * max(0, term_years - 3), dtype=float
        )[:term_years]
        q_w = np.clip(base_lapse * lapse_mult, 0.0, 0.95)
        q_w[-1] = 0.0  # Policies expire at the end of year n
        return q_w


# ==============================================================================
# PHASE 2: CORE ACTUARIAL MATH & RESERVE ENGINE
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

        # ----------------------------------------------------------------------
        # 1. Traditional Net Level Prospective Reserve (Mortality Only)
        # ----------------------------------------------------------------------
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

        # ----------------------------------------------------------------------
        # 2. Multiple-Decrement Survival (Mortality + Lapse) & Gross Premium
        # ----------------------------------------------------------------------
        p_tau = (1.0 - q_d) * (1.0 - q_w)
        t_p_x_active = np.ones(n + 1)
        for t in range(1, n + 1):
            t_p_x_active[t] = t_p_x_active[t - 1] * p_tau[t - 1]

        durations = np.arange(1, n + 1)
        fixed_expenses = np.where(
            durations == 1,
            a.acq_expense_fixed,
            a.maint_expense_fixed * ((1.0 + a.inflation_rate) ** (durations - 1))
        )
        pct_expenses = np.where(durations == 1, a.acq_expense_pct, a.maint_expense_pct)

        if fixed_gross_premium is None:
            disc_earn_beg = v_earn ** np.arange(n)
            disc_earn_end = v_earn ** np.arange(1, n + 1)
            epv_claims = np.sum(t_p_x_active[:-1] * q_d * a.face_amount * disc_earn_end)
            epv_fixed_exp = np.sum(t_p_x_active[:-1] * fixed_expenses * disc_earn_beg)
            epv_prem_factor = np.sum(
                t_p_x_active[:-1] * (1.0 - pct_expenses - a.target_profit_loading) * disc_earn_beg
            )
            gross_premium = (epv_claims + epv_fixed_exp) / epv_prem_factor
        else:
            gross_premium = fixed_gross_premium

        total_expenses_per_pol = fixed_expenses + pct_expenses * gross_premium

        # ----------------------------------------------------------------------
        # 3. Dynamic Reserve (Multi-Decrement Recursive Valuation with PfAD)
        # ----------------------------------------------------------------------
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

        # ----------------------------------------------------------------------
        # 4. Cash Flow Projection (per 1.0 Initial Issued Policy)
        # ----------------------------------------------------------------------
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

        # ----------------------------------------------------------------------
        # 5. Pricing Key Performance Indicators (KPIs)
        # ----------------------------------------------------------------------
        disc_hurdle_beg = (1.0 + a.hurdle_rate) ** (-np.arange(n))
        disc_hurdle_end = (1.0 + a.hurdle_rate) ** (-np.arange(1, n + 1))

        pv_profits = float(np.sum(net_profit * disc_hurdle_end))
        pv_gross_prems = float(np.sum(prem_income * disc_hurdle_beg))
        profit_margin = pv_profits / pv_gross_prems if pv_gross_prems > 0 else 0.0
        irr = compute_irr(net_profit)

        return {
            "table_used": self.loader.get_sheet_name(a.gender, a.smoker_status, a.age_basis),
            "net_premium": float(net_premium),
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

    def run_stress_suite(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        baseline_res = self.engine.run_valuation()
        locked_gp = baseline_res["gross_premium"]
        sheet_label = baseline_res["table_used"]

        scenarios = [
            (f"1. Baseline ({sheet_label})", {"interest_shock": 0.00, "mortality_mult": 1.00, "lapse_mult": 1.00, "apply_wearoff": True}),
            ("2. Interest Rate -100 bps",    {"interest_shock": -0.01, "mortality_mult": 1.00, "lapse_mult": 1.00, "apply_wearoff": True}),
            ("3. Interest Rate +100 bps",    {"interest_shock": +0.01, "mortality_mult": 1.00, "lapse_mult": 1.00, "apply_wearoff": True}),
            ("4. Mortality +15% Shock",      {"interest_shock": 0.00, "mortality_mult": 1.15, "lapse_mult": 1.00, "apply_wearoff": True}),
            ("5. Lapse +25% Shock",          {"interest_shock": 0.00, "mortality_mult": 1.00, "lapse_mult": 1.25, "apply_wearoff": True}),
            ("6. No Preferred Wear-off",     {"interest_shock": 0.00, "mortality_mult": 1.00, "lapse_mult": 1.00, "apply_wearoff": False}),
        ]

        summary_rows = []
        scenario_projections = {}

        for name, params in scenarios:
            res = self.engine.run_valuation(fixed_gross_premium=locked_gp, **params)
            scenario_projections[name] = res["projection_df"]
            summary_rows.append({
                "Scenario": name,
                "Net_Premium ($)": round(res["net_premium"], 2),
                "Locked_Gross_Prem ($)": round(res["gross_premium"], 2),
                "PVFP ($)": round(res["pvfp"], 2),
                "Profit_Margin (%)": round(res["profit_margin"] * 100, 2),
                "IRR (%)": round(res["irr"] * 100, 2) if not np.isnan(res["irr"]) else np.nan,
                "Peak_Net_Reserve ($)": round(res["peak_net_reserve"], 2),
                "Peak_Dynamic_Reserve ($)": round(res["peak_dynamic_reserve"], 2),
            })

        summary_df = pd.DataFrame(summary_rows)
        return summary_df, scenario_projections

    def compare_all_vbt_cohorts(self) -> pd.DataFrame:
        """Evaluates pricing & reserve metrics across all 8 SOA 2015 VBT sheets for the given Issue Age."""
        rows = []
        for gender in ["Male", "Female"]:
            for smoker in ["Non-Smoker", "Smoker"]:
                for basis in ["ANB", "ALB"]:
                    cohort_assump = ActuarialAssumptions(
                        issue_age=self.engine.assumptions.issue_age,
                        gender=gender,
                        smoker_status=smoker,
                        age_basis=basis,
                        term_years=self.engine.assumptions.term_years,
                        face_amount=self.engine.assumptions.face_amount
                    )
                    temp_engine = TermLifeEngine(cohort_assump, self.engine.loader)
                    res = temp_engine.run_valuation()
                    rows.append({
                        "VBT_Sheet": res["table_used"],
                        "Gender": gender,
                        "Smoker_Status": smoker,
                        "Age_Basis": basis,
                        "Net_Premium ($)": round(res["net_premium"], 2),
                        "Gross_Premium ($)": round(res["gross_premium"], 2),
                        "PVFP ($)": round(res["pvfp"], 2),
                        "Profit_Margin (%)": round(res["profit_margin"] * 100, 2),
                        "IRR (%)": round(res["irr"] * 100, 2),
                        "Peak_Net_Reserve ($)": round(res["peak_net_reserve"], 2),
                        "Peak_Dynamic_Reserve ($)": round(res["peak_dynamic_reserve"], 2),
                    })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    loader = SOA2015VBTDataLoader(
        vbt_filepath="2015-vbt-smoker-distinct-alb-anb.xlsx",
        wearoff_filepath="2015-vbt-preferred-wearoff-factors_2.xlsx"
    )
    assumptions = ActuarialAssumptions(
        issue_age=45,
        gender="Male",
        smoker_status="Non-Smoker",
        age_basis="ANB",
        term_years=20,
        face_amount=500_000.0
    )

    engine = TermLifeEngine(assumptions, loader)
    sim_engine = PricingSimulationEngine(engine)

    stress_summary_df, scenario_dfs = sim_engine.run_stress_suite()
    cohort_comparison_df = sim_engine.compare_all_vbt_cohorts()

    print("=== PHASE 3A: STRESS TEST SUMMARY (AGE 45, 2015 MNS ANB) ===")
    print(stress_summary_df.to_string(index=False))

    print("\n=== PHASE 3B: COHORT PRICING COMPARISON ACROSS ALL 8 SOA 2015 VBT TABLES (AGE 45) ===")
    print(cohort_comparison_df.to_string(index=False))