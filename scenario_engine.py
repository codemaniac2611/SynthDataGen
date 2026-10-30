
from copy import deepcopy

import pandas as pd


def generate_transfer_scenarios(
    row_count=1000,
    payroll_delay_pct=10,
    access_failure_pct=5,
    seed=42,
):
    """Generate linked employee-transfer data with labeled test failures."""
    import numpy as np

    if row_count < 1:
        raise ValueError("row_count must be at least 1")

    for name, pct in (
        ("payroll_delay_pct", payroll_delay_pct),
        ("access_failure_pct", access_failure_pct),
    ):
        if not 0 <= pct <= 100:
            raise ValueError(f"{name} must be between 0 and 100")

    rng = np.random.default_rng(seed)
    delayed_count = round(row_count * payroll_delay_pct / 100)
    access_failure_count = round(row_count * access_failure_pct / 100)

    employee_ids = [f"EMP{i:06d}" for i in range(1, row_count + 1)]
    delayed_indices = set(
        rng.choice(row_count, size=delayed_count, replace=False).tolist()
    )
    access_failure_indices = set(
        rng.choice(row_count, size=access_failure_count, replace=False).tolist()
    )

    employees = pd.DataFrame({
        "employee_id": employee_ids,
        "source_country": ["India"] * row_count,
        "destination_country": ["Singapore"] * row_count,
        "transfer_status": ["COMPLETED"] * row_count,
    })

    payroll = pd.DataFrame({
        "payroll_id": [f"PAY{i:06d}" for i in range(1, row_count + 1)],
        "employee_id": employee_ids,
        "country": ["Singapore"] * row_count,
        "currency": ["SGD"] * row_count,
        "activation_status": [
            "DELAYED" if i in delayed_indices else "ACTIVE"
            for i in range(row_count)
        ],
        "injected_failure": [
            i in delayed_indices for i in range(row_count)
        ],
    })

    access = pd.DataFrame({
        "access_id": [f"ACC{i:06d}" for i in range(1, row_count + 1)],
        "employee_id": employee_ids,
        "system": ["Singapore_HRIS"] * row_count,
        "revocation_status": [
            "FAILED" if i in access_failure_indices else "REVOKED"
            for i in range(row_count)
        ],
        "injected_failure": [
            i in access_failure_indices for i in range(row_count)
        ],
    })

    tables = {
        "employees": employees,
        "payroll_assignments": payroll,
        "access_assignments": access,
    }

    validation = validate_transfer_scenarios(
        tables,
        expected_rows=row_count,
        expected_delays=delayed_count,
        expected_access_failures=access_failure_count,
    )

    return {
        "scenario": "India to Singapore employee transfer",
        "seed": seed,
        "tables": tables,
        "validation": validation,
        "passed": all(item["passed"] for item in validation),
        "summary": {
            "employees": row_count,
            "payroll_delays": delayed_count,
            "access_revocation_failures": access_failure_count,
        },
    }


def validate_transfer_scenarios(
    tables,
    expected_rows,
    expected_delays,
    expected_access_failures,
):
    employees = tables["employees"]
    payroll = tables["payroll_assignments"]
    access = tables["access_assignments"]

    employee_ids = set(employees["employee_id"])
    checks = []

    def check(name, passed, actual, expected):
        checks.append({
            "check": name,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
        })

    check(
        "Employee row count",
        len(employees) == expected_rows,
        len(employees),
        expected_rows,
    )
    check(
        "Employee IDs unique",
        employees["employee_id"].is_unique,
        int(employees["employee_id"].nunique()),
        expected_rows,
    )
    check(
        "Payroll employee references valid",
        payroll["employee_id"].isin(employee_ids).all(),
        int(payroll["employee_id"].isin(employee_ids).sum()),
        len(payroll),
    )
    check(
        "Access employee references valid",
        access["employee_id"].isin(employee_ids).all(),
        int(access["employee_id"].isin(employee_ids).sum()),
        len(access),
    )
    check(
        "Payroll delay injection count",
        int(payroll["injected_failure"].sum()) == expected_delays,
        int(payroll["injected_failure"].sum()),
        expected_delays,
    )
    check(
        "Access failure injection count",
        int(access["injected_failure"].sum()) == expected_access_failures,
        int(access["injected_failure"].sum()),
        expected_access_failures,
    )
    check(
        "Payroll failure labels match status",
        (payroll["injected_failure"] == (
            payroll["activation_status"] == "DELAYED"
        )).all(),
        int((payroll["injected_failure"] == (
            payroll["activation_status"] == "DELAYED"
        )).sum()),
        len(payroll),
    )
    check(
        "Access failure labels match status",
        (access["injected_failure"] == (
            access["revocation_status"] == "FAILED"
        )).all(),
        int((access["injected_failure"] == (
            access["revocation_status"] == "FAILED"
        )).sum()),
        len(access),
    )

    return checks
