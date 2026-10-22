import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass
from typing import Dict


# ==============================================================================
# PHASE 1: DATA LOADING & CORE ACTUARIAL ASSUMPTIONS (SOA 2015 VBT)
# ==============================================================================

@dataclass
class PolicyInputs:
    """Core policyholder and valuation inputs for the Net Premium & Reserve model."""
    issue_age: int = 45
    gender: str = "Male"               # "Male" or "Female"
    smoker_status: str = "Non-Smoker"  # "Non-Smoker" or "Smoker"
    term_years: int = 20
    face_amount: float = 500_000.0
    valuation_rate: float = 0.035      # Annual valuation interest rate (i)
    age_basis: str = "ANB"             # SOA 2015 VBT Age Nearest Birthday basis


class SOA2015VBTLoader:
    """Loads official SOA 2015 VBT Smoker-Distinct mortality tables."""

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
        xls = pd.ExcelFile(self.vbt_filepath)
        tables = {}
        for sheet in xls.sheet_names:
            df = pd.read_excel(self.vbt_filepath, sheet_name=sheet, skiprows=2)
            df = df.dropna(subset=["Iss. Age"]).copy()
            df["Iss. Age"] = df["Iss. Age"].astype(int)
            tables[sheet] = df.set_index("Iss. Age")
        return tables

    def get_sheet_name(self, gender: str, smoker_status: str, age_basis: str = "ANB") -> str:
        key = (gender.strip().lower(), smoker_status.strip().lower(), age_basis.strip().lower())
        if key not in self.SHEET_MAP:
            raise ValueError(f"Invalid table key: {key}")
        return self.SHEET_MAP[key]

    def get_select_mortality_rates(self, inputs: PolicyInputs) -> np.ndarray:
        """Extracts annual mortality rates q_{[x]+t-1} for policy years t = 1..term_years."""
        sheet_name = self.get_sheet_name(inputs.gender, inputs.smoker_status, inputs.age_basis)
        df = self.vbt_tables[sheet_name]
        if inputs.issue_age not in df.index:
            raise ValueError(f"Issue Age {inputs.issue_age} not found in {sheet_name} (valid range: 18-95).")

        row = df.loc[inputs.issue_age]
        rates_per_1000 = [float(row[d if d <= 25 else "Ult."]) for d in range(1, inputs.term_years + 1)]
        return np.clip(np.array(rates_per_1000, dtype=float) / 1000.0, 0.0, 1.0)


# ==============================================================================
# PHASE 2: EQUIVALENCE PRINCIPLE NET PREMIUM & PROSPECTIVE RESERVE ENGINE
# ==============================================================================

class NetPremiumReserveEngine:
    """
    Computes Level Net Premium (P) via the Equivalence Principle and
    Prospective Terminal Reserves (tV) from Year 0 to Maturity (Term).
    """

    def __init__(self, inputs: PolicyInputs, loader: SOA2015VBTLoader):
        self.inputs = inputs
        self.loader = loader

    def run_valuation(self) -> Dict[str, object]:
        inp = self.inputs
        n = inp.term_years
        i = max(0.0001, inp.valuation_rate)
        v = 1.0 / (1.0 + i)

        # 1. Mortality rates q_x for policy years 1..n and survival probabilities t_p_x for t = 0..n
        q_d = self.loader.get_select_mortality_rates(inp)
        p_d = 1.0 - q_d

        t_p_x = np.ones(n + 1)
        for t in range(1, n + 1):
            t_p_x[t] = t_p_x[t - 1] * p_d[t - 1]

        # 2. Equivalence Principle: EPV(Net Premiums) = EPV(Death Benefits)
        disc_beg = v ** np.arange(n)
        disc_end = v ** np.arange(1, n + 1)

        epv_annuity_due = np.sum(t_p_x[:-1] * disc_beg)
        epv_unit_benefit = np.sum(t_p_x[:-1] * q_d * disc_end)
        net_premium = inp.face_amount * (epv_unit_benefit / epv_annuity_due)

        # 3. Prospective Terminal Reserve (tV) at the end of Year t (t = 0..n)
        terminal_reserves = np.zeros(n + 1)
        for t in range(1, n):
            rem = n - t
            surv_from_t = t_p_x[t:n] / t_p_x[t]
            a_xt = np.sum(surv_from_t * (v ** np.arange(rem)))
            A_xt = np.sum(surv_from_t * q_d[t:n] * (v ** np.arange(1, rem + 1)))
            terminal_reserves[t] = inp.face_amount * A_xt - net_premium * a_xt
        terminal_reserves[n] = 0.0

        # 4. Build Year 0 to Term Cash Flow & Reserve Schedule
        years = np.arange(0, n + 1)
        q_x_schedule = np.concatenate([[0.0], q_d])
        net_prem_inflow = np.concatenate([[0.0], t_p_x[:-1] * net_premium])
        claim_outflow = np.concatenate([[0.0], t_p_x[:-1] * q_d * inp.face_amount])

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
            "table_used": self.loader.get_sheet_name(inp.gender, inp.smoker_status, inp.age_basis),
            "net_premium": float(net_premium),
            "peak_reserve": float(terminal_reserves[peak_idx]),
            "peak_year": peak_idx,
            "schedule_df": schedule_df
        }


# ==============================================================================
# PHASE 3: SIMPLIFIED 2-SHEET EXCEL EXPORT ('1_Summary', '2_Cashflows')
# ==============================================================================

def generate_excel_report(inputs: PolicyInputs, val_res: Dict[str, object]) -> bytes:
    """
    Exports a clean 2-sheet Excel workbook in memory:
    - Sheet 1 ('1_Summary'): Basic policyholder inputs, Statutory Net Premium, and Peak Reserve.
    - Sheet 2 ('2_Cashflows'): Year 0 to Term schedule (Year, q_x, t_p_x, Net Premium Inflow, Claim Outflow, Terminal Reserve tV).
    """
    output = io.BytesIO()

    summary_df = pd.DataFrame([
        ("Policyholder Input", "Issue Age", inputs.issue_age),
        ("Policyholder Input", "Gender", inputs.gender),
        ("Policyholder Input", "Smoker Status", inputs.smoker_status),
        ("Policyholder Input", "Policy Term (Years)", inputs.term_years),
        ("Policyholder Input", "Face Amount ($)", round(inputs.face_amount, 2)),
        ("Valuation Assumption", "Valuation Interest Rate (%)", round(inputs.valuation_rate * 100, 2)),
        ("Valuation Assumption", "SOA 2015 VBT Table", str(val_res["table_used"])),
        ("Key Valuation Output", "Statutory Net Premium ($)", round(val_res["net_premium"], 2)),
        ("Key Valuation Output", "Peak Dynamic Reserve ($)", round(val_res["peak_reserve"], 2)),
        ("Key Valuation Output", "Peak Reserve Policy Year", int(val_res["peak_year"])),
    ], columns=["Category", "Item", "Value"])

    cashflows_df = val_res["schedule_df"].copy()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="1_Summary", index=False)
        cashflows_df.to_excel(writer, sheet_name="2_Cashflows", index=False)

        for sheet_name, ws in writer.sheets.items():
            ws.views.sheetView[0].showGridLines = True
            ws.freeze_panes = "A2"

            headers = [cell.value for cell in ws[1]]
            for col_idx, header in enumerate(headers, start=1):
                col_letter = ws.cell(row=1, column=col_idx).column_letter
                header_str = str(header or "")

                if sheet_name == "2_Cashflows":
                    for row_idx in range(2, ws.max_row + 1):
                        cell = ws.cell(row=row_idx, column=col_idx)
                        if not isinstance(cell.value, (int, float)):
                            continue
                        if header_str in ("Net Premium Inflow", "Claim Outflow", "Terminal Reserve tV"):
                            cell.number_format = "$#,##0.00"
                        elif header_str == "q_x":
                            cell.number_format = "0.00000"
                        elif header_str == "t_p_x":
                            cell.number_format = "0.0000"
                elif sheet_name == "1_Summary" and col_idx == 3:
                    for row_idx in range(2, ws.max_row + 1):
                        item_label = str(ws.cell(row=row_idx, column=2).value or "")
                        cell = ws.cell(row=row_idx, column=3)
                        if isinstance(cell.value, (int, float)):
                            if "($)" in item_label:
                                cell.number_format = "$#,##0.00"
                            elif "(%)" in item_label:
                                cell.number_format = "0.00\"%\""

                max_len = max(
                    [len(str(ws.cell(row=r, column=col_idx).value or "")) for r in range(1, ws.max_row + 1)]
                    + [len(header_str)]
                )
                ws.column_dimensions[col_letter].width = min(max(max_len + 4, 14), 38)

    output.seek(0)
    return output.getvalue()


# ==============================================================================
# PHASE 4: STREAMLIT UI & HUMP-SHAPED RESERVE TRAJECTORY DASHBOARD
# ==============================================================================

st.set_page_config(
    page_title="Term Life Net Premium & Reserve Model | SOA 2015 VBT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep Navy (#0B192C) + Teal (#14B8A6) + Ivory (#FAF8F5) Institutional Styling
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
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #0B192C 0%, #112240 100%) !important;
        border: 1px solid rgba(20, 184, 166, 0.35) !important;
        border-left: 4px solid #14B8A6 !important;
        border-radius: 8px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.14) !important;
    }
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #E6DEC8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #FAF8F5 !important;
    }
    div[data-testid="stMetricDelta"] svg {
        display: none !important;
    }
    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] > div {
        color: #2DD4BF !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(145deg, #0B192C 0%, #112240 100%) !important;
        color: #FAF8F5 !important;
        border: 1.5px solid #14B8A6 !important;
        border-left: 5px solid #14B8A6 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.2rem !important;
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
def load_cached_vbt_data() -> SOA2015VBTLoader:
    return SOA2015VBTLoader(vbt_filepath="2015-vbt-smoker-distinct-alb-anb.xlsx")


def build_reserve_trajectory_chart(schedule_df: pd.DataFrame, peak_year: int, peak_reserve: float) -> go.Figure:
    """Builds the Hump-shaped Terminal Reserve (tV) trajectory chart from Year 0 to Maturity."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=schedule_df["Year"],
            y=schedule_df["Terminal Reserve tV"],
            mode="lines+markers",
            name="Terminal Reserve (tV)",
            fill="tozeroy",
            fillcolor="rgba(20, 184, 166, 0.14)",
            line=dict(color="#14B8A6", width=3.2),
            marker=dict(size=7, symbol="circle", color="#FAF8F5", line=dict(color="#0B192C", width=2.0)),
            hovertemplate="<b>Year %{x}</b><br>Terminal Reserve (tV): $%{y:,.2f}<extra></extra>"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[peak_year],
            y=[peak_reserve],
            mode="markers+text",
            name=f"Peak Reserve (Year {peak_year})",
            marker=dict(size=12, symbol="diamond", color="#2563EB", line=dict(color="#FAF8F5", width=2.0)),
            text=[f"Peak: ${peak_reserve:,.2f} (Yr {peak_year})"],
            textposition="top center",
            hovertemplate="<b>Peak Year %{x}</b><br>Peak Reserve: $%{y:,.2f}<extra></extra>"
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Prospective Reserve Trajectory (tV) — Hump-Shaped Path to Maturity</b>",
            x=0.0,
            y=0.96,
            xanchor="left",
            yanchor="top",
            font=dict(size=16, family="Arial, sans-serif")
        ),
        font=dict(family="Arial, sans-serif", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20, 184, 166, 0.04)",
        hovermode="x unified",
        height=460,
        margin=dict(l=60, r=40, t=60, b=95),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(148, 163, 184, 0.10)",
            bordercolor="#14B8A6",
            borderwidth=1,
            font=dict(size=11)
        )
    )

    fig.update_xaxes(
        title_text="Policy Duration (Year t)",
        dtick=1,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.22)",
        linecolor="rgba(148, 163, 184, 0.45)"
    )
    fig.update_yaxes(
        title_text="Terminal Reserve per Surviving Policy ($)",
        range=[0, peak_reserve * 1.18 if peak_reserve > 0 else 100],
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.22)",
        linecolor="rgba(148, 163, 184, 0.45)"
    )
    return fig


def main():
    st.title("Term Life Insurance Net Premium & Reserve Engine")
    st.caption(
        "Actuarial Equivalence Principle & Prospective Reserve Valuation Model | "
        "Mortality Basis: Official SOA 2015 Valuation Basic Table (VBT)"
    )

    try:
        loader = load_cached_vbt_data()
    except Exception as e:
        st.error(f"Error loading SOA 2015 VBT Excel workbook: {e}")
        st.stop()

    # --------------------------------------------------------------------------
    # Sidebar: 6 Core Actuarial Inputs
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.subheader("Policyholder & Valuation Inputs")
        issue_age = st.slider("Issue Age (x)", min_value=18, max_value=75, value=45, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        smoker_status = st.selectbox("Smoker Status", ["Non-Smoker", "Smoker"], index=0)
        term_years = st.selectbox("Policy Term (Years)", [10, 15, 20, 25], index=2)
        face_amount = st.number_input(
            "Face Amount ($)", min_value=50_000.0, max_value=10_000_000.0, value=500_000.0, step=50_000.0
        )
        valuation_rate = st.slider("Valuation Interest Rate (%)", 1.0, 7.0, 3.5, 0.25) / 100.0

    inputs = PolicyInputs(
        issue_age=issue_age,
        gender=gender,
        smoker_status=smoker_status,
        term_years=term_years,
        face_amount=face_amount,
        valuation_rate=valuation_rate
    )

    engine = NetPremiumReserveEngine(inputs, loader)
    val_res = engine.run_valuation()
    schedule_df = val_res["schedule_df"]

    # --------------------------------------------------------------------------
    # Top Header & Prominent Excel Export Button
    # --------------------------------------------------------------------------
    excel_bytes = generate_excel_report(inputs, val_res)
    excel_filename = f"Actuarial_Report_{val_res['table_used'].replace(' ', '_')}_Age{issue_age}.xlsx"

    hdr_col, dl_col = st.columns([3.2, 1.3])
    with hdr_col:
        st.markdown(f"#### Actuarial Valuation Summary — Mortality Table: `{val_res['table_used']}`")
    with dl_col:
        st.download_button(
            label="Download Actuarial Report (Excel)",
            data=excel_bytes,
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # --------------------------------------------------------------------------
    # Top KPI Metrics (Strictly 2 Core Metrics)
    # --------------------------------------------------------------------------
    k1, k2 = st.columns(2)
    k1.metric(
        "Statutory Net Premium (Annual)",
        f"${val_res['net_premium']:,.2f}",
        f"Equivalence Principle | i = {valuation_rate*100:.2f}%"
    )
    k2.metric(
        "Peak Dynamic Reserve",
        f"${val_res['peak_reserve']:,.2f}",
        f"Reached at Policy Year {val_res['peak_year']} (Age {issue_age + val_res['peak_year']})"
    )

    st.divider()

    # --------------------------------------------------------------------------
    # Main Chart: Hump-Shaped Reserve Trajectory
    # --------------------------------------------------------------------------
    fig_reserve = build_reserve_trajectory_chart(
        schedule_df=schedule_df,
        peak_year=val_res["peak_year"],
        peak_reserve=val_res["peak_reserve"]
    )
    st.plotly_chart(fig_reserve, use_container_width=True, theme="streamlit")

    # --------------------------------------------------------------------------
    # Year 0 to Term Cash Flow & Reserve Schedule Table
    # --------------------------------------------------------------------------
    st.markdown("##### Annual Actuarial Cash Flow & Terminal Reserve Schedule (Year 0 to Maturity)")
    st.dataframe(
        schedule_df.style.format({
            "q_x": "{:.5f}",
            "t_p_x": "{:.4f}",
            "Net Premium Inflow": "${:,.2f}",
            "Claim Outflow": "${:,.2f}",
            "Terminal Reserve tV": "${:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )


if __name__ == "__main__":
    main()