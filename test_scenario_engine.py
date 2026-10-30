
import unittest

from scenario_engine import generate_transfer_scenarios


class TestTransferScenarios(unittest.TestCase):
    def test_default_counts(self):
        result = generate_transfer_scenarios()
        self.assertEqual(result["summary"]["employees"], 1000)
        self.assertEqual(result["summary"]["payroll_delays"], 100)
        self.assertEqual(
            result["summary"]["access_revocation_failures"], 50
        )

    def test_validation_passes(self):
        result = generate_transfer_scenarios()
        self.assertTrue(result["passed"])
        self.assertTrue(all(c["passed"] for c in result["validation"]))

    def test_same_seed_is_reproducible(self):
        a = generate_transfer_scenarios(seed=123)
        b = generate_transfer_scenarios(seed=123)
        for name in a["tables"]:
            self.assertTrue(a["tables"][name].equals(b["tables"][name]))

    def test_different_seeds_change_failure_placement(self):
        a = generate_transfer_scenarios(seed=1)
        b = generate_transfer_scenarios(seed=2)
        self.assertFalse(
            a["tables"]["payroll_assignments"]["injected_failure"].equals(
                b["tables"]["payroll_assignments"]["injected_failure"]
            )
        )

    def test_linked_records(self):
        result = generate_transfer_scenarios(row_count=250)
        employee_ids = set(result["tables"]["employees"]["employee_id"])
        for name in ("payroll_assignments", "access_assignments"):
            table = result["tables"][name]
            self.assertEqual(len(table), 250)
            self.assertTrue(table["employee_id"].isin(employee_ids).all())

    def test_invalid_percentages_rejected(self):
        with self.assertRaises(ValueError):
            generate_transfer_scenarios(payroll_delay_pct=101)
        with self.assertRaises(ValueError):
            generate_transfer_scenarios(access_failure_pct=-1)

    def test_zero_rows_rejected(self):
        with self.assertRaises(ValueError):
            generate_transfer_scenarios(row_count=0)


if __name__ == "__main__":
    unittest.main()
