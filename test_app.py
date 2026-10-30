import unittest

import pandas as pd

from synth_core import (
    generate_employee_dataset,
    validate_employee_dataset,
    repair_employee_missing_cost_centres,
    generate_retail_dataset,
    validate_retail_dataset,
    generate_iot_dataset,
    validate_iot_dataset,
)


def employee_plan(row_count=1000, missing_pct=2):
    return {
        "row_count": row_count,
        "min_age": 21,
        "max_age": 65,
        "missing_cost_center_pct": missing_pct,
        "salary_age_correlation": 0.55,
        "countries": ["US", "UK", "India"],
        "include_terminated": True,
    }


class TestEmployeeDataset(unittest.TestCase):
    def test_row_count(self):
        df = generate_employee_dataset(employee_plan(500), seed=42)
        self.assertEqual(len(df), 500)

    def test_unique_employee_ids(self):
        df = generate_employee_dataset(employee_plan(), seed=42)
        self.assertTrue(df["employee_id"].is_unique)

    def test_age_bounds(self):
        plan = employee_plan()
        df = generate_employee_dataset(plan, seed=42)
        self.assertTrue(df["age"].between(21, 65).all())

    def test_positive_salaries(self):
        df = generate_employee_dataset(employee_plan(), seed=42)
        self.assertTrue((df["annual_salary"] > 0).all())

    def test_exact_missing_cost_center_target(self):
        plan = employee_plan(1000, 7)
        df = generate_employee_dataset(plan, seed=42)
        self.assertEqual(int(df["cost_center"].isna().sum()), 70)

    def test_reproducible_seed(self):
        plan = employee_plan()
        first = generate_employee_dataset(plan, seed=123)
        second = generate_employee_dataset(plan, seed=123)
        pd.testing.assert_frame_equal(first, second)

    def test_all_employee_validation_checks_pass(self):
        plan = employee_plan()
        df = generate_employee_dataset(plan, seed=42)
        report = validate_employee_dataset(df, plan)
        self.assertTrue(
            report["Passed"].all(),
            msg=report.loc[~report["Passed"]].to_string(index=False),
        )

    def test_missing_cost_center_repair(self):
        plan = employee_plan(1000, 2)
        df = generate_employee_dataset(plan, seed=42)

        # Simulate a coverage gap that needs repair.
        df.loc[df.index[:10], "cost_center"] = None

        repaired, stats = repair_employee_missing_cost_centres(
            df, plan, seed=42
        )

        self.assertEqual(int(repaired["cost_center"].isna().sum()), 20)
        self.assertTrue(stats["target_met"])
        self.assertTrue(validate_employee_dataset(repaired, plan)["Passed"].all())


class TestRetailDataset(unittest.TestCase):
    def test_row_count_and_columns(self):
        df = generate_retail_dataset(500, seed=42)
        self.assertEqual(len(df), 500)
        self.assertTrue(
            {"order_id", "product", "category", "unit_price",
             "quantity", "order_date", "order_total"}.issubset(df.columns)
        )

    def test_reproducible_seed(self):
        first = generate_retail_dataset(100, seed=123)
        second = generate_retail_dataset(100, seed=123)
        pd.testing.assert_frame_equal(first, second)

    def test_all_retail_validation_checks_pass(self):
        df = generate_retail_dataset(1000, seed=42)
        report = validate_retail_dataset(df)
        self.assertTrue(
            report["Passed"].all(),
            msg=report.loc[~report["Passed"]].to_string(index=False),
        )


class TestIoTDataset(unittest.TestCase):
    def test_row_count_and_columns(self):
        df = generate_iot_dataset(500, seed=42)
        self.assertEqual(len(df), 500)
        self.assertTrue(
            {"device_id", "timestamp", "temperature_c",
             "humidity_pct", "status"}.issubset(df.columns)
        )

    def test_reproducible_seed(self):
        first = generate_iot_dataset(100, seed=123)
        second = generate_iot_dataset(100, seed=123)
        pd.testing.assert_frame_equal(first, second)

    def test_all_iot_validation_checks_pass(self):
        df = generate_iot_dataset(1000, seed=42)
        report = validate_iot_dataset(df)
        self.assertTrue(
            report["Passed"].all(),
            msg=report.loc[~report["Passed"]].to_string(index=False),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
