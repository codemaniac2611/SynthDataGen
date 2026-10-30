import unittest

from universal_engine import generate_universal_dataset


class TestUniversalEngine(unittest.TestCase):

    def test_iot_schema_and_row_count(self):
        prompt = (
            "Generate 20,000 IoT telemetry readings from 500 devices, "
            "including device IDs, timestamps, temperatures, "
            "humidity percentages between 0 and 100, "
            "battery levels between 0 and 100, and alert statuses. "
            "Include occasional anomalous readings."
        )

        df, report, plan = generate_universal_dataset(prompt)

        self.assertEqual(len(df), 20_000)

        expected = {
            "device_id",
            "timestamp",
            "temperature",
            "humidity_pct",
            "battery_level",
            "alert_status",
        }
        self.assertTrue(expected.issubset(set(df.columns)))

    def test_device_cardinality(self):
        prompt = (
            "Generate 20,000 IoT telemetry readings from 500 devices, "
            "including device IDs, timestamps, temperatures, "
            "humidity percentages between 0 and 100, "
            "battery levels between 0 and 100, and alert statuses."
        )

        df, report, plan = generate_universal_dataset(prompt)

        self.assertEqual(df["device_id"].nunique(), 500)

    def test_numeric_bounds(self):
        prompt = (
            "Generate 20,000 IoT telemetry readings from 500 devices, "
            "including device IDs, timestamps, temperatures, "
            "humidity percentages between 0 and 100, "
            "battery levels between 0 and 100, and alert statuses."
        )

        df, report, plan = generate_universal_dataset(prompt)

        self.assertTrue(df["humidity_pct"].between(0, 100).all())
        self.assertTrue(df["battery_level"].between(0, 100).all())

    def test_validation_report_passes_supported_constraints(self):
        prompt = (
            "Generate 20,000 IoT telemetry readings from 500 devices, "
            "including device IDs, timestamps, temperatures, "
            "humidity percentages between 0 and 100, "
            "battery levels between 0 and 100, and alert statuses."
        )

        df, report, plan = generate_universal_dataset(prompt)

        self.assertTrue(
            report["Passed"].all(),
            msg=f"Validation failures:\n{report[~report['Passed']]}",
        )

    def test_reproducible_seed(self):
        prompt = (
            "Generate 100 records including claim IDs, claim amounts, "
            "incident dates, risk scores, and fraud labels."
        )

        first, _, _ = generate_universal_dataset(prompt, seed=42)
        second, _, _ = generate_universal_dataset(prompt, seed=42)

        self.assertTrue(first.equals(second))

    def test_insurance_schema(self):
        prompt = (
            "Generate 100 records including claim IDs, policy types, "
            "claim amounts, incident dates, risk scores, and fraud labels."
        )

        df, report, plan = generate_universal_dataset(prompt)

        expected = {
            "claim_id",
            "policy_type",
            "claim_amount",
            "incident_date",
            "risk_score",
            "fraud_label",
        }
        self.assertTrue(expected.issubset(set(df.columns)))
        self.assertEqual(len(df), 100)


    def test_exact_fraud_percentage(self):
        prompt = (
            "Generate 10,000 claims including claim IDs, claim amounts, "
            "incident dates, risk scores, and fraud labels. "
            "Exactly 5% of claims should be fraud."
        )

        df, report, plan = generate_universal_dataset(prompt)

        self.assertEqual(len(df), 10_000)
        self.assertEqual(int((df["fraud_label"] == "fraud").sum()), 500)
        self.assertTrue(
            report["Passed"].all(),
            msg=f"Validation failures:\n{report[~report['Passed']]}",
        )


    def test_exact_alert_status_proportion(self):
        prompt = (
            "Generate 1,000 records including alert status. "
            "Exactly 10% of records should have alert status critical."
        )
        df, report, plan = generate_universal_dataset(prompt)

        self.assertEqual(int((df["alert_status"] == "critical").sum()), 100)
        self.assertTrue(report["Passed"].all())

    def test_exact_customer_segment_proportion(self):
        prompt = (
            "Generate 1,000 records including customer segment. "
            "Exactly 20% of records should have customer segment premium."
        )
        df, report, plan = generate_universal_dataset(prompt)

        self.assertEqual(
            int((df["customer_segment"] == "premium").sum()), 200
        )
        self.assertTrue(report["Passed"].all())


    def test_injected_anomalies_are_iqr_detectable(self):
        from universal_engine import detect_iqr_outliers

        prompt = (
            "Generate 1000 IoT readings with temperature, humidity_pct "
            "and battery_level. Keep humidity_pct between 0 and 100 "
            "and battery_level between 0 and 100. "
            "Include occasional anomalous readings."
        )

        df, report, plan = generate_universal_dataset(prompt, seed=42)
        injected = set(plan["anomaly_rows"])

        self.assertEqual(len(injected), 20)

        detected = set()
        for column in plan["anomaly_columns"]:
            mask = detect_iqr_outliers(df[column])
            detected.update(df.index[mask].tolist())

        self.assertTrue(
            injected.issubset(detected),
            msg=f"Injected rows not detected: {sorted(injected - detected)}",
        )

    def test_anomaly_generation_preserves_bounds(self):
        prompt = (
            "Generate 1000 IoT readings with temperature, humidity_pct "
            "and battery_level. Keep humidity_pct between 0 and 100 "
            "and battery_level between 0 and 100. "
            "Include occasional anomalous readings."
        )

        df, _, _ = generate_universal_dataset(prompt, seed=42)

        self.assertTrue(df["humidity_pct"].between(0, 100).all())
        self.assertTrue(df["battery_level"].between(0, 100).all())

    def test_anomaly_generation_is_reproducible(self):
        prompt = (
            "Generate 1000 IoT readings with temperature, humidity_pct "
            "and battery_level. Keep humidity_pct between 0 and 100 "
            "and battery_level between 0 and 100. "
            "Include occasional anomalous readings."
        )

        first, _, first_plan = generate_universal_dataset(prompt, seed=42)
        second, _, second_plan = generate_universal_dataset(prompt, seed=42)

        self.assertTrue(first.equals(second))
        self.assertEqual(first_plan["anomaly_rows"], second_plan["anomaly_rows"])


if __name__ == "__main__":
    unittest.main()
