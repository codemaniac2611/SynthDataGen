"""Streamlit interface for the local SynthDataGen workflow engine."""

import pandas as pd
import streamlit as st

from workflow_engine import run_workflow


def employee_transfer_spec():
    """Return a deterministic example of an India-to-Singapore transfer."""
    return {
        "name": "India to Singapore employee transfer",
        "initial_tables": {
            "employees": [
                {
                    "employee_id": "E001",
                    "name": "Synthetic Employee 001",
                    "country": "India",
                    "cost_center": "CC-IN",
                    "employment_status": "Active",
                }
            ],
            "payroll_assignments": [
                {
                    "assignment_id": "PAY-IN-001",
                    "employee_id": "E001",
                    "country": "India",
                    "currency": "INR",
                    "status": "Active",
                }
            ],
            "access_assignments": [
                {
                    "assignment_id": "ACC-IN-001",
                    "employee_id": "E001",
                    "system": "India-ERP",
                    "status": "Active",
                }
            ],
        },
        "events": [
            {
                "id": "transfer_employee",
                "action": "update",
                "table": "employees",
                "where": {"employee_id": "E001"},
                "set": {
                    "country": "Singapore",
                    "cost_center": "CC-SG",
                },
            },
            {
                "id": "end_india_payroll",
                "action": "update",
                "table": "payroll_assignments",
                "depends_on": ["transfer_employee"],
                "where": {
                    "employee_id": "E001",
                    "country": "India",
                    "status": "Active",
                },
                "set": {"status": "Ended"},
            },
            {
                "id": "create_singapore_payroll",
                "action": "create",
                "table": "payroll_assignments",
                "depends_on": ["transfer_employee"],
                "record": {
                    "assignment_id": "PAY-SG-001",
                    "employee_id": "E001",
                    "country": "Singapore",
                    "currency": "SGD",
                    "status": "Active",
                },
            },
            {
                "id": "revoke_india_access",
                "action": "update",
                "table": "access_assignments",
                "depends_on": ["transfer_employee"],
                "where": {
                    "employee_id": "E001",
                    "system": "India-ERP",
                    "status": "Active",
                },
                "set": {"status": "Revoked"},
            },
            {
                "id": "create_singapore_access",
                "action": "create",
                "table": "access_assignments",
                "depends_on": ["transfer_employee"],
                "record": {
                    "assignment_id": "ACC-SG-001",
                    "employee_id": "E001",
                    "system": "Singapore-ERP",
                    "status": "Active",
                },
            },
        ],
        "invariants": [
            {
                "type": "unique",
                "name": "Unique employee IDs",
                "table": "employees",
                "field": "employee_id",
            },
            {
                "type": "unique",
                "name": "Unique payroll assignment IDs",
                "table": "payroll_assignments",
                "field": "assignment_id",
            },
            {
                "type": "unique",
                "name": "Unique access assignment IDs",
                "table": "access_assignments",
                "field": "assignment_id",
            },
            {
                "type": "foreign_key",
                "name": "Payroll references existing employees",
                "table": "payroll_assignments",
                "field": "employee_id",
                "references_table": "employees",
                "references_field": "employee_id",
            },
            {
                "type": "foreign_key",
                "name": "Access references existing employees",
                "table": "access_assignments",
                "field": "employee_id",
                "references_table": "employees",
                "references_field": "employee_id",
            },
            {
                "type": "required",
                "name": "Payroll currency is present",
                "table": "payroll_assignments",
                "field": "currency",
            },
        ],
    }


def render_workflow_simulator():
    st.divider()
    st.header("Workflow Simulator")
    st.write(
        "Simulate a cross-system employee transfer with linked HR, "
        "payroll and access records. Execution is local and deterministic."
    )

    st.info(
        "Demo scenario: India → Singapore. This simulates business records; "
        "it does not connect to or change real HR, payroll or access systems."
    )

    seed = st.number_input(
        "Scenario seed",
        min_value=0,
        max_value=2_147_483_647,
        value=42,
        step=1,
        key="workflow_seed",
    )

    if st.button(
        "Run employee transfer",
        type="primary",
        use_container_width=True,
        key="run_employee_transfer",
    ):
        try:
            result = run_workflow(employee_transfer_spec(), seed=int(seed))
            st.session_state["workflow_result"] = result
        except Exception as exc:
            st.session_state.pop("workflow_result", None)
            st.error(f"Workflow execution failed: {type(exc).__name__}: {exc}")

    result = st.session_state.get("workflow_result")
    if not result:
        st.caption("Run the scenario to inspect events, records and validation.")
        return

    st.subheader(result["workflow_name"])

    passed_count = sum(
        item["passed"] is True for item in result["validation"]
    )
    failed_count = sum(
        item["passed"] is False for item in result["validation"]
    )
    unknown_count = sum(
        item["passed"] is None for item in result["validation"]
    )

    first, second, third, fourth = st.columns(4)
    first.metric("Events executed", len(result["event_log"]))
    second.metric("Tables", len(result["tables"]))
    third.metric("Checks passed", passed_count)
    fourth.metric("Checks failed", failed_count + unknown_count)

    if result["passed"]:
        st.success("All implemented workflow invariants passed.")
    else:
        st.warning(
            "One or more workflow invariants failed or could not be verified."
        )

    st.subheader("Event timeline")
    st.dataframe(
        pd.DataFrame(result["event_log"]),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Generated records")
    table_tabs = st.tabs(list(result["tables"].keys()))

    for tab, (table_name, records) in zip(
        table_tabs, result["tables"].items()
    ):
        with tab:
            frame = pd.DataFrame(records)
            st.dataframe(frame, hide_index=True, use_container_width=True)
            st.download_button(
                f"Download {table_name}.csv",
                data=frame.to_csv(index=False).encode("utf-8"),
                file_name=f"{table_name}.csv",
                mime="text/csv",
                key=f"download_workflow_{table_name}",
            )

    st.subheader("Validation report")
    report = pd.DataFrame(result["validation"])
    st.dataframe(report, hide_index=True, use_container_width=True)

    st.download_button(
        "Download workflow report (JSON)",
        data=__import__("json").dumps(result, indent=2, default=str),
        file_name="employee_transfer_workflow.json",
        mime="application/json",
        key="download_workflow_report",
    )
