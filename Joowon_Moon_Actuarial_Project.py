import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple


# ==============================================================================
# PHASE 1: DATA LOADING & ACTUARIAL ASSUMPTIONS (SOA 2015 VBT WORKBOOK)
# ==============================================================================

@dataclass
class ActuarialAssumptions:
    """Stores core policyholder profile and valuation interest rate assumption."""
    issue_age: int = 45
    gender: str = "Male"               # "Male" or "Female"
    smoker_status: str = "Non-Smoker"  # "Non-Smoker" or "Smoker"
    age_basis: str = "ANB"             # "ANB" (Age Nearest Birthday) or "ALB" (Age Last Birthday)
    term_years: int = 20
    face_amount: float = 500_000.0
    valuation_rate: float = 0.035      # 3.5% annual statutory valuation interest rate


class SOA2015VBTDataLoader:
    """
    Loads the official SOA 2015 VBT Smoker-Distinct Excel workbook:
    - 2015-vbt-smoker-distinct-alb-anb.xlsx (8 sheets: MNS/FNS/MSM/FSM x ANB/ALB, per 1,000)
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

    def __init__(self, vbt_filepath: str = "2015-vbt-smoker-distinct-alb-anb.xlsx"):
        self.vbt_filepath = vbt_filepath
        self.vbt_tables = self._load_all_vbt_sheets()

    def _load_all_vbt_sheets(self) -> Dict[str, pd.DataFrame]:
        """Reads all 8 smoker-distinct ANB/ALB sheets from the SOA 2015 VBT workbook."""
        xls = pd.ExcelFile(self.vbt_filepath)
        tables = {}
        for sheet in xls.sheet_names:
            df = pd.read_excel(self.vbt_filepath, sheet_name=sheet, skiprows=2)
            df = df.dropna(subset=["Iss. Age"]).copy()
            df["Iss. Age"] = df["Iss. Age"].astype(int)
            tables[sheet] = df.set_index("Iss. Age")
        return tables

    def get_sheet_name(self, gender: str, smoker_status: str, age_basis: str) -> str:
        key = (gender.strip().lower(), smoker_status.strip().lower(), age_basis.strip().lower())
        if key not in self.SHEET_MAP:
            raise ValueError(f"Invalid combination {key}. Valid options: Male/Female, Non-Smoker/Smoker, ANB/ALB.")
        return self.SHEET_MAP[key]

    def get_select_mortality_rates(
        self, assumptions: ActuarialAssumptions, mortality_mult: float = 1.0
    ) -> np.ndarray:
        """
        Extracts SOA 2015 VBT Select mortality rates q_{[x]+t-1}
        (converting per-1,000 table values to decimal probabilities) for t = 1..term_years.
        """
        sheet_name = self.get_sheet_name(assumptions.gender, assumptions.smoker_status, assumptions.age_basis)
        df = self.vbt_tables[sheet_name]

        if assumptions.issue_age not in df.index:
            raise ValueError(f"Issue Age {assumptions.issue_age} not found in {sheet_name} (valid range: 18-95).")

        row = df.loc[assumptions.issue_age]
        rates_per_1000 = [float(row[d if d <= 25 else "Ult."]) for d in range(1, assumptions.term_years + 1)]
        q_base = np.array(rates_per_1000, dtype=float) / 1000.0
        return np.clip(q_base * mortality_mult, 0.0, 1.0)


# ==============================================================================
# PHASE 2: CORE ACTUARIAL MATH (NET PREMIUM & PROSPECTIVE RESERVE ENGINE)
# ==============================================================================

class TermLifeEngine:
    """Calculates Statutory Net Premium (P) and Prospective Terminal Reserves (tV)."""

    def __init__(self, assumptions: ActuarialAssumptions, loader: SOA2015VBTDataLoader):
        self.assumptions = assumptions
        self.loader = loader

    def run_valuation(
        self,
        interest_shock: float = 0.0,
        mortality_mult: float = 1.0
    ) -> Dict[str, object]:
        a = self.assumptions
        n = a.term_years
        val_i = max(0.0001, a.valuation_rate + interest_shock)
        v = 1.0 / (1.0 + val_i)

        # 1. Single-decrement mortality rates (q_x) and survival probabilities (t_p_x)
        q_d = self.loader.get_select_mortality_rates(a, mortality_mult=mortality_mult)
        p_d = 1.0 - q_d

        t_p_x = np.ones(n + 1)
        for t in range(1, n + 1):
            t_p_x[t] = t_p_x[t - 1] * p_d[t - 1]

        # 2. Equivalence Principle: Net Premium (P) = Face * A_{x:n}^1 / a_due_{x:n}
        disc_beg = v ** np.arange(n)
        disc_end = v ** np.arange(1, n + 1)

        epv_annuity_due = np.sum(t_p_x[:-1] * disc_beg)
        epv_unit_benefit = np.sum(t_p_x[:-1] * q_d * disc_end)
        net_premium = a.face_amount * (epv_unit_benefit / epv_annuity_due)

        # 3. Prospective Terminal Reserve (tV) at the end of policy year t (t = 0..n)
        terminal_reserves = np.zeros(n + 1)
        for t in range(1, n):
            rem = n - t
            surv_t = t_p_x[t:n] / t_p_x[t]
            a_xt = np.sum(surv_t * (v ** np.arange(rem)))
            A_xt = np.sum(surv_t * q_d[t:n] * (v ** np.arange(1, rem + 1)))
            terminal_reserves[t] = a.face_amount * A_xt - net_premium * a_xt
        terminal_reserves[n] = 0.0

        # 4. Year 0 to Term Cash Flow & Terminal Reserve Schedule
        years = np.arange(0, n + 1)
        q_x_schedule = np.concatenate([[0.0], q_d])
        net_prem_inflow = np.concatenate([[0.0], t_p_x[:-1] * net_premium])
        claim_outflow = np.concatenate([[0.0], t_p_x[:-1] * q_d * a.face_amount])

        schedule_df = pd.DataFrame({
            "Year": years,
            "q_x": q_x_schedule,
            "t_p_x": t_p_x,
            "Net Premium Inflow": net_prem_inflow,
            "Claim Outflow": claim_outflow,
            "Terminal Reserve tV": terminal_reserves
        })

        peak_idx = int(np.argmax(terminal_reserves))

        return {
            "table_used": self.loader.get_sheet_name(a.gender, a.smoker_status, a.age_basis),
            "valuation_rate": float(val_i),
            "net_premium": float(net_premium),
            "peak_reserve": float(terminal_reserves[peak_idx]),
            "peak_year": peak_idx,
            "schedule_df": schedule_df
        }


# ==============================================================================
# PHASE 3: SENSITIVITY ANALYSIS & COHORT COMPARISON ENGINE
# ==============================================================================

class PricingSimulationEngine:
    """Runs interest/mortality sensitivity checks and compares all 8 SOA 2015 VBT tables."""

    def __init__(self, term_engine: TermLifeEngine):
        self.engine = term_engine

    def run_sensitivity_suite(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        sheet_label = self.engine.loader.get_sheet_name(
            self.engine.assumptions.gender,
            self.engine.assumptions.smoker_status,
            self.engine.assumptions.age_basis
        )

        scenarios = [
            (f"1. Baseline ({sheet_label})", {"interest_shock": 0.00, "mortality_mult": 1.00}),
            ("2. Interest Rate -100 bps",    {"interest_shock": -0.01, "mortality_mult": 1.00}),
            ("3. Interest Rate +100 bps",    {"interest_shock": +0.01, "mortality_mult": 1.00}),
            ("4. Mortality +15% Shock",      {"interest_shock": 0.00, "mortality_mult": 1.15}),
        ]

        summary_rows = []
        scenario_schedules = {}

        for name, params in scenarios:
            res = self.engine.run_valuation(**params)
            scenario_schedules[name] = res["schedule_df"]
            summary_rows.append({
                "Scenario": name,
                "Valuation_Rate (%)": round(res["valuation_rate"] * 100, 2),
                "Statutory_Net_Premium ($)": round(res["net_premium"], 2),
                "Peak_Reserve ($)": round(res["peak_reserve"], 2),
                "Peak_Year": res["peak_year"],
            })

        return pd.DataFrame(summary_rows), scenario_schedules

    def compare_all_vbt_cohorts(self) -> pd.DataFrame:
        """Evaluates Net Premium and Peak Reserve across all 8 SOA 2015 VBT sheets."""
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
                        valuation_rate=a.valuation_rate
                    )
                    temp_engine = TermLifeEngine(cohort_assump, self.engine.loader)
                    res = temp_engine.run_valuation()
                    rows.append({
                        "VBT_Sheet": res["table_used"],
                        "Gender": gender,
                        "Smoker_Status": smoker,
                        "Age_Basis": basis,
                        "Statutory_Net_Premium ($)": round(res["net_premium"], 2),
                        "Peak_Reserve ($)": round(res["peak_reserve"], 2),
                        "Peak_Year": res["peak_year"],
                    })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    loader = SOA2015VBTDataLoader(vbt_filepath="2015-vbt-smoker-distinct-alb-anb.xlsx")
    assumptions = ActuarialAssumptions(
        issue_age=45,
        gender="Male",
        smoker_status="Non-Smoker",
        age_basis="ANB",
        term_years=20,
        face_amount=500_000.0,
        valuation_rate=0.035
    )

    engine = TermLifeEngine(assumptions, loader)
    sim_engine = PricingSimulationEngine(engine)

    sensitivity_df, scenario_schedules = sim_engine.run_sensitivity_suite()
    cohort_comparison_df = sim_engine.compare_all_vbt_cohorts()

    print("=== PHASE 2: BASELINE CASH FLOW & TERMINAL RESERVE SCHEDULE (YEAR 0 TO 20) ===")
    baseline_key = list(scenario_schedules.keys())[0]
    print(scenario_schedules[baseline_key].to_string(index=False))

    print("\n=== PHASE 3A: SENSITIVITY SUMMARY (AGE 45, 2015 MNS ANB) ===")
    print(sensitivity_df.to_string(index=False))

    print("\n=== PHASE 3B: COHORT COMPARISON ACROSS ALL 8 SOA 2015 VBT TABLES (AGE 45) ===")
    print(cohort_comparison_df.to_string(index=False))