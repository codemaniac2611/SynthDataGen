import json
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from synth_core import (
    generate_employee_dataset,
    validate_employee_dataset,
    repair_employee_missing_cost_centres,
    generate_retail_dataset,
    validate_retail_dataset,
    generate_iot_dataset,
    validate_iot_dataset,
)

load_dotenv(".env")

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
BASE_URL = bytes([104,116,116,112,115,58,47,47,97,112,105,46,116,111,107,101,110,102,97,99,116,111,114,121,46,110,101,98,105,117,115,46,99,111,109,47,118,49,47]).decode()

def interpret_transfer_with_nemotron(prompt):
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["NEBIUS_API_KEY"]
        except (KeyError, FileNotFoundError):
            pass
    if not api_key:
        raise RuntimeError("NEBIUS_API_KEY is not configured")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract transfer requirements from the user prompt. "
                    "Return only JSON with keys row_count, payroll_delay_pct, "
                    "and access_failure_pct. Use defaults 1000, 10, and 5 "
                    "only when a value is not specified. Do not add markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=300,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Nemotron returned an empty response")

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\\{.*\\}", content, re.DOTALL)
        if not match:
            raise ValueError("Nemotron did not return valid JSON")
        data = json.loads(match.group(0))

    row_count = int(data["row_count"])
    payroll_pct = float(data["payroll_delay_pct"])
    access_pct = float(data["access_failure_pct"])

    if not 1 <= row_count <= 100000:
        raise ValueError("Invalid employee count from Nemotron")
    if not 0 <= payroll_pct <= 100 or not 0 <= access_pct <= 100:
        raise ValueError("Invalid failure percentage from Nemotron")

    return row_count, payroll_pct, access_pct


st.set_page_config(page_title="SynthDataGen", layout="wide")

DEFAULT_PROMPT = (
    "Generate 1000 synthetic employee records. Ages 21 to 65. "
    "Salary should positively correlate with age. Exactly 2% of cost centres "
    "should be missing. Include US, UK, India, active and terminated employees."
)


def interpret_request(prompt):
    """Interpret supported employee-data requirements locally; no API call."""
    text = prompt.lower()

    def number(pattern, default):
        match = re.search(pattern, text, re.IGNORECASE)
        return int(match.group(1).replace(",", "")) if match else default

    row_count = number(
        r"\b(\d[\d,]*)\s*(?:synthetic\s+)?(?:employee\s+)?(?:records|rows|employees)\b",
        1000,
    )
    row_count = int(np.clip(row_count, 100, 100000))

    age_range = re.search(
        r"\bages?\s*(\d{1,3})\s*(?:to|-|–)\s*(\d{1,3})\b",
        text,
    )
    if age_range:
        min_age, max_age = map(int, age_range.groups())
    else:
        min_age = number(r"\bminimum age\s*(?:of|is|=|:)?\s*(\d{1,3})", 21)
        max_age = number(r"\bmaximum age\s*(?:of|is|=|:)?\s*(\d{1,3})", 65)

    missing_pct = 2.0
    missing_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*%\s*(?:of\s*)?"
        r"(?:cost[\s_-]*cent(?:er|re)s?).{0,35}?\bmissing\b",
        text,
    )
    if not missing_match:
        missing_match = re.search(
            r"\bmissing\b.{0,35}?\b(\d+(?:\.\d+)?)\s*%\s*"
            r"(?:of\s*)?(?:cost[\s_-]*cent(?:er|re)s?)",
            text,
        )
    if missing_match:
        missing_pct = float(missing_match.group(1))
    missing_pct = float(np.clip(missing_pct, 0, 30))

    countries = []
    for name, code in (
        ("united states", "US"), ("usa", "US"), ("us", "US"),
        ("united kingdom", "UK"), ("britain", "UK"), ("uk", "UK"),
        ("india", "India"),
    ):
        if re.search(r"\b" + re.escape(name) + r"\b", text) and code not in countries:
            countries.append(code)
    if not countries:
        countries = ["US", "UK", "India"]

    include_terminated = not bool(
        re.search(r"\bactive employees only\b|\bonly active employees\b", text)
        or re.search(r"\bexclude terminated\b|\bno terminated\b", text)
    )

    correlation = 0.55
    if re.search(r"\b(?:negatively|inversely)\s+correlat", text):
        correlation = -0.55
    elif re.search(r"\b(?:no|zero|uncorrelated)\s+correlation\b", text):
        correlation = 0.0
    elif re.search(r"\b(?:strongly|highly)\s+positively\s+correlat", text):
        correlation = 0.75
    elif re.search(r"\bpositively\s+correlat", text):
        correlation = 0.55

    min_age = int(np.clip(min_age, 18, 80))
    max_age = int(np.clip(max_age, 18, 100))
    if min_age > max_age:
        min_age, max_age = max_age, min_age

    return {
        "row_count": row_count,
        "min_age": min_age,
        "max_age": max_age,
        "missing_cost_center_pct": missing_pct,
        "salary_age_correlation": correlation,
        "countries": countries,
        "include_terminated": include_terminated,
    }

def generate_dataset(plan, seed=42):
    rng = np.random.default_rng(int(seed))
    n = plan["row_count"]
    ages = rng.integers(plan["min_age"], plan["max_age"] + 1, n)

    age_std = (ages - ages.mean()) / max(ages.std(), 1)
    target = plan["salary_age_correlation"]
    noise = rng.normal(size=n)
    signal = target * age_std + np.sqrt(max(0, 1 - target**2)) * noise
    salaries = np.clip(65000 + signal * 18000, 22000, 250000).round().astype(int)

    cost_centres = rng.choice(["CC-100", "CC-200", "CC-300", "CC-400"], n).astype(object)
    missing_count = round(n * plan["missing_cost_center_pct"] / 100)
    if missing_count:
        cost_centres[rng.choice(n, missing_count, replace=False)] = None

    ids = np.array([f"E{i:06d}" for i in range(1, n + 1)])
    managers = [
        None if i == 0 or rng.random() < 0.08 else ids[rng.integers(0, i)]
        for i in range(n)
    ]

    df = pd.DataFrame({
        "employee_id": ids,
        "age": ages,
        "annual_salary": salaries,
        "country": rng.choice(plan["countries"], n),
        "cost_center": cost_centres,
        "employment_status": (
            rng.choice(["Active", "Terminated"], n, p=[0.88, 0.12])
            if plan["include_terminated"] else "Active"
        ),
        "manager_id": managers,
    })
    return df



def repair_missing_cost_centres(df, plan, seed=42):
    """Adjust missing cost centres to match the requested exact target."""
    repaired = df.copy(deep=True)
    target = round(len(repaired) * plan["missing_cost_center_pct"] / 100)
    mask = repaired["cost_center"].isna()
    before = int(mask.sum())
    rng = np.random.default_rng(int(seed))

    if before > target:
        missing_indices = repaired.index[mask].to_numpy()
        fill_indices = rng.choice(missing_indices, size=before - target, replace=False)
        available = repaired.loc[~mask, "cost_center"].dropna()
        choices = available.to_numpy() if not available.empty else np.array(
            ["CC-100", "CC-200", "CC-300", "CC-400"]
        )
        repaired.loc[fill_indices, "cost_center"] = rng.choice(
            choices, size=len(fill_indices), replace=True
        )
    elif before < target:
        present_indices = repaired.index[~mask].to_numpy()
        new_missing = rng.choice(present_indices, size=target - before, replace=False)
        repaired.loc[new_missing, "cost_center"] = None

    after = int(repaired["cost_center"].isna().sum())
    return repaired, {
        "target_missing": target,
        "missing_before": before,
        "missing_after": after,
        "target_met": after == target,
    }


def validate_dataset(df, plan):
    n = len(df)
    expected_missing = round(n * plan["missing_cost_center_pct"] / 100)
    observed_corr = df["age"].corr(df["annual_salary"])
    target_corr = plan["salary_age_correlation"]
    checks = [
        ("Unique employee IDs", df["employee_id"].is_unique,
         f"{df['employee_id'].nunique()} unique IDs", f"{n} unique IDs"),
        ("Age boundaries", bool(df["age"].between(plan["min_age"], plan["max_age"]).all()),
         f"{df['age'].min()}–{df['age'].max()}",
         f"{plan['min_age']}–{plan['max_age']}"),
        ("Positive salaries", bool((df["annual_salary"] > 0).all()),
         int((df["annual_salary"] <= 0).sum()), "0 invalid"),
        ("Missing cost-centre target", int(df["cost_center"].isna().sum()) == expected_missing,
         f"{df['cost_center'].isna().sum()} rows",
         f"{expected_missing} rows"),
        ("Valid manager references",
         bool(df["manager_id"].dropna().isin(df["employee_id"]).all()),
         int((~df["manager_id"].dropna().isin(df["employee_id"])).sum()),
         "0 invalid references"),
        ("Age-salary correlation",
         abs(observed_corr - target_corr) <= 0.18,
         f"{observed_corr:.3f}", f"{target_corr:.2f} ± 0.18"),
        ("Allowed countries", set(df["country"]).issubset(set(plan["countries"])),
         ", ".join(sorted(df["country"].unique())), ", ".join(plan["countries"])),
    ]
    return pd.DataFrame(checks, columns=["Check", "Passed", "Observed", "Requirement"])




from universal_engine import generate_universal_dataset
from scenario_engine import generate_transfer_scenarios

st.markdown(
    """
    <style>
    #MainMenu, footer, [data-testid="stSidebar"] {visibility: hidden;}
    .block-container {
        max-width: 900px;
        padding-top: 5rem;
        padding-bottom: 3rem;
    }
    .hero-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 650;
        letter-spacing: -0.045em;
        margin-bottom: 0.45rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #858585;
        font-size: 1.05rem;
        margin-bottom: 2.5rem;
    }
    div[data-testid="stTextArea"] textarea {
        border-radius: 16px;
        padding: 1.1rem;
        font-size: 1rem;
        min-height: 150px;
    }
    div.stButton > button {
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

page = st.radio(
    "Workspace",
    ["Generate", "Scenario Lab"],
    horizontal=True,
    label_visibility="collapsed",
)

if page == "Scenario Lab":
    st.title("Scenario Lab")
    st.caption("Simulate linked HR, payroll and access-management workflows.")
    from workflow_ui import render_workflow_simulator
    render_workflow_simulator()
    st.stop()

st.markdown('<div class="hero-title">SynthDataGen</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Describe the data you need. Generate and verify it.</div>',
    unsafe_allow_html=True,
)

prompt = st.text_area(
    "Describe your dataset",
    value=st.session_state.get("synth_prompt", ""),
    placeholder=(
        "Generate 10,000 insurance claims with claim IDs, policy types, "
        "claim amounts, incident dates, risk scores and fraud labels."
    ),
    height=155,
    label_visibility="collapsed",
)

generate = st.button(
    "Generate dataset",
    type="primary",
    use_container_width=True,
)

if generate:
    if not prompt.strip():
        st.warning("Describe the dataset you want to generate.")
    else:
        try:
            text = prompt.lower()
            is_transfer = (
                re.search(r"\btransfers?\b", text) is not None
                and re.search(r"\bindia\b", text) is not None
                and re.search(r"\bsingapore\b", text) is not None
                and re.search(r"\bpayroll\b", text) is not None
                and re.search(r"\baccess\b", text) is not None
            )

            with st.spinner("Generating and validating synthetic data locally..."):
                if is_transfer:
                    count_match = re.search(
                        r"\b(?:generate|create)\s+([\d,]+)\b", text
                    )
                    if not count_match:
                        count_match = re.search(
                            r"\b([\d,]+)\s+(?:employee\s+)?transfers?\b", text
                        )
                    row_count = (
                        int(count_match.group(1).replace(",", ""))
                        if count_match else 1000
                    )
                    if not 1 <= row_count <= 100000:
                        raise ValueError("Transfer count must be between 1 and 100,000.")

                    payroll_match = re.search(
                        r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?"
                        r"(?:payroll(?:[\s-]+activation)?\s+)?"
                        r"(?:delays?|delayed)",
                        text,
                    )
                    access_match = re.search(
                        r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?"
                        r"(?:access(?:[\s-]+revocation)?\s+)?"
                        r"(?:failures?|failed)",
                        text,
                    )

                    payroll_pct = float(payroll_match.group(1)) if payroll_match else 10
                    access_pct = float(access_match.group(1)) if access_match else 5
                    interpreter_source = "Local parser fallback"
                    try:
                        row_count, payroll_pct, access_pct = interpret_transfer_with_nemotron(prompt)
                        interpreter_source = "Nebius Nemotron"
                    except Exception as exc:
                        st.warning(
                            "Nemotron interpretation failed; using local parsing instead. "
                            + str(exc)
                        )

                    interpretation = {
                        "Scenario": "India to Singapore employee transfer",
                        "Interpreter": interpreter_source,
                        "Employee transfers": row_count,
                        "Payroll delays (%)": payroll_pct,
                        "Access-revocation failures (%)": access_pct,
                        "Payroll percentage source": (
                            "Nebius Nemotron" if interpreter_source == "Nebius Nemotron" else ("Prompt" if payroll_match else "Default (10%)")
                        ),
                        "Access percentage source": (
                            "Nebius Nemotron" if interpreter_source == "Nebius Nemotron" else ("Prompt" if access_match else "Default (5%)")
                        ),
                    }

                    result = generate_transfer_scenarios(
                        row_count=row_count,
                        payroll_delay_pct=payroll_pct,
                        access_failure_pct=access_pct,
                    )
                    tables = result["tables"]
                    df = pd.concat(
                        [
                            table.assign(record_type=name)
                            for name, table in tables.items()
                        ],
                        ignore_index=True,
                        sort=False,
                    )
                    report = pd.DataFrame([
                        {
                            "Check": item["check"],
                            "Passed": item["passed"],
                            "Observed": str(item["actual"]),
                            "Requirement": str(item["expected"]),
                        }
                        for item in result["validation"]
                    ])
                    plan = {
                        "scenario": result["scenario"],
                        "seed": result["seed"],
                        "summary": result["summary"],
                    }
                    st.session_state["synth_tables"] = tables
                    st.session_state["synth_mode"] = "transfer"
                    st.session_state["synth_scenario_summary"] = result["summary"]
                    st.session_state["synth_interpretation"] = interpretation
                else:
                    df, report, plan = generate_universal_dataset(prompt)
                    st.session_state["synth_tables"] = {}
                    st.session_state["synth_mode"] = "dataset"
                    st.session_state.pop("synth_scenario_summary", None)

            st.session_state["synth_df"] = df
            st.session_state["synth_report"] = report
            st.session_state["synth_plan"] = plan
            st.session_state["synth_prompt"] = prompt
            st.session_state["synth_filename"] = (
                "employee_transfer_scenario.csv"
                if is_transfer else "synthetic_dataset.csv"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Generation failed: {type(exc).__name__}: {exc}")

if "synth_df" in st.session_state:
    df = st.session_state["synth_df"]
    report = st.session_state["synth_report"]
    plan = st.session_state["synth_plan"]
    passed = int(report["Passed"].sum())

    st.divider()
    left, middle, right = st.columns(3)
    left.metric("Records", f"{len(df):,}")
    middle.metric("Columns", f"{len(df.columns):,}")
    right.metric("Validation checks", f"{passed}/{len(report)}")

    if passed == len(report):
        st.success("All implemented validation checks passed.")
    else:
        st.warning(f"{len(report) - passed} validation check(s) failed.")

    if st.session_state.get("synth_mode") == "transfer":
        st.subheader("Interpreted requirements")
        interpretation_df = pd.DataFrame(
            list(st.session_state.get("synth_interpretation", {}).items()),
            columns=["Requirement", "Interpreted value"],
        )
        interpretation_df["Interpreted value"] = (
            interpretation_df["Interpreted value"].astype(str)
        )
        st.dataframe(
            interpretation_df,
            hide_index=True,
            use_container_width=True,
        )
        st.subheader("Linked transfer tables")
        tables = st.session_state["synth_tables"]
        table_tabs = st.tabs(list(tables.keys()))
        for tab, (name, table) in zip(table_tabs, tables.items()):
            with tab:
                st.dataframe(table.head(100), hide_index=True, use_container_width=True)
                st.download_button(
                    f"Download {name} (CSV)",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name=f"{name}.csv",
                    mime="text/csv",
                    key=f"download_{name}",
                )
        st.caption(
            "Injected payroll delays and access failures are intentional test cases. "
            "Validation checks verify the generated scenario and failure counts."
        )
    else:
        st.subheader("Dataset overview")

        metric_cols = st.columns(4)
        metric_cols[0].metric("Total records", f"{len(df):,}")
        metric_cols[1].metric("Features", f"{len(df.columns):,}")
        metric_cols[2].metric("Missing values", f"{int(df.isna().sum().sum()):,}")
        metric_cols[3].metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        date_cols = [
            c for c in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        categorical_cols = [
            c for c in df.select_dtypes(include=["object", "category", "bool"]).columns
            if df[c].nunique(dropna=True) <= 30
        ]

        def style_chart(fig, height=350):
            fig.update_layout(
                template="plotly_dark",
                height=height,
                title_x=0.03,
                margin=dict(l=25, r=20, t=55, b=25),
                font=dict(family="Arial", size=12),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            return fig

        st.divider()
        st.subheader("Feature distributions")

        for i in range(0, min(len(numeric_cols), 4), 2):
            chart_cols = st.columns(2)
            for j, column in enumerate(numeric_cols[i:i + 2]):
                fig = px.histogram(
                    df, x=column, nbins=30,
                    title=f"{column.replace(chr(95), chr(32)).title()} distribution",
                    color_discrete_sequence=["#60A5FA"],
                )
                fig.update_traces(marker_line_width=0)
                chart_cols[j].plotly_chart(
                    style_chart(fig), use_container_width=True
                )

        if categorical_cols:
            st.subheader("Category analysis")
            chart_cols = st.columns(2)
            for i, column in enumerate(categorical_cols[:2]):
                counts = (
                    df[column].astype(str).value_counts().head(12)
                    .rename_axis(column).reset_index(name="Count")
                )
                fig = px.bar(
                    counts, x="Count", y=column, orientation="h",
                    title=f"{column.replace(chr(95), chr(32)).title()} breakdown",
                    color="Count",
                    color_continuous_scale=["#2563EB", "#38BDF8"],
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                chart_cols[i].plotly_chart(
                    style_chart(fig), use_container_width=True
                )

        if date_cols:
            st.subheader("Time-series analysis")
            column = date_cols[0]
            trend = (
                df.dropna(subset=[column])
                .set_index(column).resample("MS").size()
                .rename("Records").reset_index()
            )
            fig = px.line(
                trend, x=column, y="Records",
                title=f"Records over time",
                markers=True,
                color_discrete_sequence=["#38BDF8"],
            )
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(
                style_chart(fig, 400), use_container_width=True
            )

        if len(numeric_cols) >= 2:
            st.subheader("Correlation analysis")
            correlation = df[numeric_cols].corr()
            fig = px.imshow(
                correlation, text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, aspect="auto",
                title="Relationships between numeric features",
            )
            st.plotly_chart(
                style_chart(fig, 480), use_container_width=True
            )

        st.divider()
        st.subheader("Data preview")
        st.dataframe(df.head(100), hide_index=True, use_container_width=True)
        st.download_button(
            "Download dataset (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=st.session_state.get("synth_filename", "synthetic_dataset.csv"),
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Validation details"):
        report_display = report.copy()
        if "Requirement" in report_display.columns:
            report_display["Requirement"] = report_display[
                "Requirement"
            ].map(str)
        st.dataframe(
            report_display,
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Inferred schema and limitations"):
        st.json(plan)
