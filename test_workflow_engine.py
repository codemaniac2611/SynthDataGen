import unittest
from copy import deepcopy

from workflow_engine import WorkflowError, run_workflow


def employee_transfer_spec():
    return {
        "name": "India to Singapore transfer",
        "initial_tables": {
            "employees": [
                {
                    "employee_id": "E001",
                    "country": "India",
                    "cost_center": "CC-IN",
                }
            ],
            "payroll_assignments": [],
            "access_assignments": [],
        },
        "events": [
            {
                "id": "transfer",
                "action": "update",
                "table": "employees",
                "where": {"employee_id": "E001"},
                "set": {"country": "Singapore", "cost_center": "CC-SG"},
            },
            {
                "id": "new_payroll",
                "action": "create",
                "table": "payroll_assignments",
                "depends_on": ["transfer"],
                "record": {
                    "employee_id": "E001",
                    "country": "Singapore",
                    "currency": "SGD",
                },
            },
            {
                "id": "new_access",
                "action": "create",
                "table": "access_assignments",
                "depends_on": ["transfer"],
                "record": {
                    "employee_id": "E001",
                    "country": "Singapore",
                    "system": "Singapore",
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
                "type": "foreign_key",
                "name": "Payroll employee references",
                "table": "payroll_assignments",
                "field": "employee_id",
                "references_table": "employees",
                "references_field": "employee_id",
            },
            {
                "type": "foreign_key",
                "name": "Access employee references",
                "table": "access_assignments",
                "field": "employee_id",
                "references_table": "employees",
                "references_field": "employee_id",
            },
            {
                "type": "required",
                "name": "Payroll currency required",
                "table": "payroll_assignments",
                "field": "currency",
            },
        ],
    }


class TestWorkflowEngine(unittest.TestCase):
    def test_employee_transfer_creates_linked_records(self):
        result = run_workflow(employee_transfer_spec())

        self.assertTrue(result["passed"])
        self.assertEqual(result["tables"]["employees"][0]["country"], "Singapore")
        self.assertEqual(
            result["tables"]["payroll_assignments"][0]["employee_id"], "E001"
        )
        self.assertEqual(
            result["tables"]["access_assignments"][0]["employee_id"], "E001"
        )
        self.assertTrue(all(item["passed"] for item in result["validation"]))

    def test_dependency_order_is_respected(self):
        spec = {
            "events": [
                {
                    "id": "second",
                    "action": "create",
                    "table": "items",
                    "depends_on": ["first"],
                    "record": {"id": 2},
                },
                {
                    "id": "first",
                    "action": "create",
                    "table": "items",
                    "record": {"id": 1},
                },
            ]
        }

        result = run_workflow(spec)
        self.assertEqual(
            [event["event_id"] for event in result["event_log"]],
            ["first", "second"],
        )

    def test_foreign_key_violation_is_reported(self):
        spec = employee_transfer_spec()
        spec["events"][1]["record"]["employee_id"] = "MISSING"

        result = run_workflow(spec)
        check = next(
            item for item in result["validation"]
            if item["name"] == "Payroll employee references"
        )

        self.assertFalse(check["passed"])
        self.assertFalse(result["passed"])

    def test_missing_update_target_fails_loudly(self):
        spec = {
            "events": [{
                "id": "update",
                "action": "update",
                "table": "employees",
                "where": {"employee_id": "MISSING"},
                "set": {"country": "Singapore"},
            }]
        }

        with self.assertRaises(WorkflowError):
            run_workflow(spec)

    def test_dependency_cycle_is_rejected(self):
        spec = {
            "events": [
                {
                    "id": "a",
                    "action": "create",
                    "table": "items",
                    "depends_on": ["b"],
                    "record": {"id": 1},
                },
                {
                    "id": "b",
                    "action": "create",
                    "table": "items",
                    "depends_on": ["a"],
                    "record": {"id": 2},
                },
            ]
        }

        with self.assertRaisesRegex(WorkflowError, "cycle"):
            run_workflow(spec)

    def test_same_spec_is_reproducible(self):
        spec = employee_transfer_spec()
        first = run_workflow(spec, seed=42)
        second = run_workflow(spec, seed=42)

        self.assertEqual(first, second)

    def test_input_spec_is_not_mutated(self):
        spec = employee_transfer_spec()
        original = deepcopy(spec)

        run_workflow(spec)

        self.assertEqual(spec, original)


if __name__ == "__main__":
    unittest.main()
