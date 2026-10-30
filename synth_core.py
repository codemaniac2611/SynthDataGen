"""Local synthetic-data generation and validation for SynthDataGen."""
import numpy as np
import pandas as pd


def generate_employee_dataset(plan, seed=42):
    rng = np.random.default_rng(int(seed))
    n = int(plan["row_count"])
    ages = rng.integers(int(plan["min_age"]), int(plan["max_age"]) + 1, n)

    age_std = (ages - ages.mean()) / max(ages.std(), 1)
    target = float(np.clip(plan["salary_age_correlation"], -0.8, 0.8))
    noise = rng.normal(size=n)
    signal = target * age_std + np.sqrt(max(0, 1 - target**2)) * noise
    salaries = np.clip(65000 + signal * 18000, 22000, 250000).round().astype(int)

    cost_centres = rng.choice(
        ["CC-100", "CC-200", "CC-300", "CC-400"], n
    ).astype(object)
    missing_count = round(n * float(plan["missing_cost_center_pct"]) / 100)
    if missing_count:
        indices = rng.choice(n, missing_count, replace=False)
        cost_centres[indices] = None

    ids = np.array([f"E{i:06d}" for i in range(1, n + 1)])
    managers = [
        None if i == 0 or rng.random() < 0.08 else ids[rng.integers(0, i)]
        for i in range(n)
    ]

    return pd.DataFrame({
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


def validate_employee_dataset(df, plan):
    n = len(df)
    expected_missing = round(n * float(plan["missing_cost_center_pct"]) / 100)
    observed_corr = df["age"].corr(df["annual_salary"])
    target_corr = float(plan["salary_age_correlation"])
    manager_ids = df["manager_id"].dropna()
    checks = [
        ("Unique employee IDs", bool(df["employee_id"].is_unique),
         f"{df['employee_id'].nunique()} unique IDs", f"{n} unique IDs"),
        ("Age boundaries",
         bool(df["age"].between(plan["min_age"], plan["max_age"]).all()),
         f"{df['age'].min()}–{df['age'].max()}",
         f"{plan['min_age']}–{plan['max_age']}"),
        ("Positive salaries", bool((df["annual_salary"] > 0).all()),
         int((df["annual_salary"] <= 0).sum()), "0 invalid"),
        ("Missing cost-centre target",
         int(df["cost_center"].isna().sum()) == expected_missing,
         f"{df['cost_center'].isna().sum()} rows", f"{expected_missing} rows"),
        ("Valid manager references",
         bool(manager_ids.isin(df["employee_id"]).all()),
         int((~manager_ids.isin(df["employee_id"])).sum()),
         "0 invalid references"),
        ("Age-salary correlation",
         bool(pd.notna(observed_corr) and abs(observed_corr - target_corr) <= 0.18),
         f"{observed_corr:.3f}" if pd.notna(observed_corr) else "undefined",
         f"{target_corr:.2f} ± 0.18"),
        ("Allowed countries",
         bool(set(df["country"]).issubset(set(plan["countries"]))),
         ", ".join(sorted(df["country"].unique())),
         ", ".join(plan["countries"])),
    ]
    return pd.DataFrame(checks, columns=["Check", "Passed", "Observed", "Requirement"])


def repair_employee_missing_cost_centres(df, plan, seed=42):
    repaired = df.copy(deep=True)
    target = round(len(repaired) * float(plan["missing_cost_center_pct"]) / 100)
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


def generate_retail_dataset(row_count=1000, seed=42):
    rng = np.random.default_rng(int(seed))
    n = int(row_count)
    products = np.array(["Notebook", "Headphones", "Coffee Mug", "Backpack", "Keyboard"])
    categories = {
        "Notebook": "Stationery",
        "Headphones": "Electronics",
        "Coffee Mug": "Home",
        "Backpack": "Accessories",
        "Keyboard": "Electronics",
    }
    product = rng.choice(products, n)
    unit_price = np.round(rng.uniform(4.0, 250.0, n), 2)
    quantity = rng.integers(1, 11, n)
    order_dates = (
        pd.Timestamp("2025-01-01")
        + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    )
    return pd.DataFrame({
        "order_id": [f"O{i:07d}" for i in range(1, n + 1)],
        "product": product,
        "category": [categories[p] for p in product],
        "unit_price": unit_price,
        "quantity": quantity,
        "order_date": order_dates,
        "order_total": np.round(unit_price * quantity, 2),
    })


def validate_retail_dataset(df):
    checks = [
        ("Unique order IDs", bool(df["order_id"].is_unique)),
        ("Positive prices", bool((df["unit_price"] > 0).all())),
        ("Positive quantities", bool((df["quantity"] > 0).all())),
        ("Order totals consistent",
         bool(np.allclose(df["order_total"], df["unit_price"] * df["quantity"]))),
        ("Valid categories", bool(df["category"].notna().all())),
        ("Valid order dates", bool(pd.to_datetime(df["order_date"], errors="coerce").notna().all())),
    ]
    return pd.DataFrame(checks, columns=["Check", "Passed"])


def generate_iot_dataset(row_count=1000, seed=42):
    rng = np.random.default_rng(int(seed))
    n = int(row_count)
    timestamps = (
        pd.Timestamp("2026-01-01")
        + pd.to_timedelta(np.arange(n) * 5, unit="min")
    )
    device_ids = np.array([f"D{i:04d}" for i in rng.integers(1, 51, n)])
    temperature = np.round(rng.normal(22, 5, n), 2)
    humidity = np.round(rng.uniform(20, 85, n), 2)
    status = np.where(
        (temperature < 5) | (temperature > 40) | (humidity > 80),
        "Alert",
        "Normal",
    )
    return pd.DataFrame({
        "device_id": device_ids,
        "timestamp": timestamps,
        "temperature_c": temperature,
        "humidity_pct": humidity,
        "status": status,
    })


def validate_iot_dataset(df):
    checks = [
        ("Device IDs present", bool(df["device_id"].notna().all())),
        ("Timestamps valid", bool(pd.to_datetime(df["timestamp"], errors="coerce").notna().all())),
        ("Temperature plausible", bool(df["temperature_c"].between(-40, 85).all())),
        ("Humidity in range", bool(df["humidity_pct"].between(0, 100).all())),
        ("Status allowed", bool(df["status"].isin(["Normal", "Alert"]).all())),
    ]
    return pd.DataFrame(checks, columns=["Check", "Passed"])
