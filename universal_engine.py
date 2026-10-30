import re
import numpy as np
import pandas as pd


ALIASES = {
    "device ids": "device_id",
    "device id": "device_id",
    "claim ids": "claim_id",
    "claim id": "claim_id",
    "timestamps": "timestamp",
    "temperatures": "temperature",
    "humidity percentages": "humidity_pct",
    "humidity percentage": "humidity_pct",
    "humidity percent": "humidity_pct",
    "humidity": "humidity_pct",
    "battery levels": "battery_level",
    "battery level": "battery_level",
    "alert statuses": "alert_status",
    "alert status": "alert_status",
    "policy types": "policy_type",
    "claim amounts": "claim_amount",
    "incident dates": "incident_date",
    "risk scores": "risk_score",
    "fraud labels": "fraud_label",
}


def normalize_column(value):
    name = re.sub(r"[_\-]+", " ", value.lower().strip())
    name = re.sub(r"\s+", " ", name).strip(" .,:;")
    if name in ALIASES:
        return ALIASES[name]
    if name.endswith("ies"):
        name = name[:-3] + "y"
    elif name.endswith("s") and not name.endswith(("ss", "us", "is")):
        name = name[:-1]
    return name.replace(" ", "_") or "field"


def infer_row_count(prompt):
    match = re.search(r"\b([\d,]+)\s+rows?\b", prompt, re.I)
    if match:
        return int(match.group(1).replace(",", ""))

    match = re.search(
        r"\b(?:generate|create|make|produce)\s+([\d,]+)\b",
        prompt,
        re.I,
    )
    if match:
        return int(match.group(1).replace(",", ""))

    match = re.search(r"\b([\d,]+)\s+(?:IoT\s+)?(?:readings|records|entries|claims)\b", prompt, re.I)
    if match:
        return int(match.group(1).replace(",", ""))

    return 1000


def infer_columns(prompt):
    text = prompt.lower().strip()

    # Support explicitly listed columns, including bullet lists.
    markers = (
        "include these columns:",
        "include the following columns:",
        "include these fields:",
        "include the following fields:",
    )

    for marker in markers:
        position = text.find(marker)
        if position == -1:
            continue

        field_text = text[position + len(marker):]
        fields = []

        for line in field_text.splitlines():
            item = line.strip()

            if not item:
                if fields:
                    break
                continue

            if item.endswith(":") and any(
                word in item for word in ("requirements", "constraints", "instructions")
            ):
                break

            if fields and not item.startswith(("-", "*", "•")):
                break

            item = item.lstrip("-*•0123456789.) ").strip()

            for part in item.split(","):
                part = part.strip(" .,:;")
                if not part or len(part) > 60:
                    continue

                column = normalize_column(part)
                if column and column not in fields:
                    fields.append(column)

        if fields:
            return fields


    patterns = [
        r"\bwith columns?\s+(.+)",
        r"\bwith fields?\s+(.+)",
        r"\bincluding\s+(.+)",
        r"\bcontaining\s+(.+)",
        r"\bwith\s+(.+)",
    ]

    field_text = None
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            field_text = match.group(1)
            break

    if not field_text:
        return None

    # Stop at the end of the field-list sentence so later instructions
    # such as "Exactly 5% of claims should be fraud" are not columns.
    field_text = re.split(r"\.\s+", field_text, maxsplit=1)[0]

    # Remove numeric range constraints while retaining later column names.
    field_text = re.sub(
        r"\s+between\s+-?\d+(?:\.\d+)?\s+and\s+-?\d+(?:\.\d+)?",
        "",
        field_text,
    )
    field_text = re.split(
        r"\.\s*(?:include|ensure|keep|make sure)\b|;\s*",
        field_text,
        maxsplit=1,
    )[0]
    field_text = re.sub(r"\band\b", ",", field_text)

    fields = []
    for part in field_text.split(","):
        part = part.strip(" .,:;")
        part = re.sub(
            r"^(?:a|an|the|realistic|synthetic|unique|valid|calculated)\s+",
            "",
            part,
        ).strip()

        if not part or len(part) > 60:
            continue

        column = normalize_column(part)
        if column not in fields:
            fields.append(column)

    return fields or None

def infer_type(column):
    name = column.lower()

    if any(word in name for word in ("timestamp", "date", "time")):
        return "datetime"
    if any(word in name for word in ("amount", "salary", "price", "temperature",
                                      "humidity", "battery", "score", "percentage",
                                      "percent", "rate", "quantity", "age")):
        return "numeric"
    if any(word in name for word in ("id", "identifier")):
        return "id"
    if any(word in name for word in ("label", "status", "type", "category")):
        return "categorical"
    return "text"


def parse_numeric_bounds(prompt, column):
    text = prompt.lower()
    column_words = re.escape(column.replace("_", " "))
    aliases = {
        "humidity_pct": r"humidity(?: percentages?| percent)?",
        "battery_level": r"battery levels?",
        "temperature": r"temperatures?",
        "risk_score": r"risk scores?",
        "claim_amount": r"claim amounts?",
    }
    target = aliases.get(column, column_words)

    match = re.search(
        rf"{target}\s+(?:must be\s+)?between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)",
        text,
    )
    if match:
        return float(match.group(1)), float(match.group(2))

    return None


def parse_device_count(prompt):
    match = re.search(r"\bfrom\s+([\d,]+)\s+devices?\b", prompt, re.I)
    if match:
        return int(match.group(1).replace(",", ""))
    match = re.search(r"\b([\d,]+)\s+devices?\b", prompt, re.I)
    if match:
        return int(match.group(1).replace(",", ""))
    return None



def parse_exact_percentage(prompt, column):
    """Return (percentage, target_label) for a recognized exact-rate request."""
    text = prompt.lower()
    column_text = re.escape(column.replace("_", " ")).replace(r"\ ", r"[\s_-]+")

    # Fraud phrasing: "Exactly 5% of claims should be fraud."
    if "fraud" in column:
        match = re.search(
            r"\bexactly\s+(\d+(?:\.\d+)?)\s*(?:%|percent)\s+"
            r"(?:of\s+(?:the\s+)?(?:rows|records|claims)\s+)?"
            r"(?:should\s+be\s+)?(?:labelled\s+|labeled\s+)?fraud\b",
            text,
        )
        if match:
            percentage = float(match.group(1))
            if 0 <= percentage <= 100:
                return percentage, "fraud"

        match = re.search(
            r"\bexactly\s+(\d+(?:\.\d+)?)\s*(?:%|percent)\s+"
            r"of\s+claims\s+should\s+be\s+fraud\b",
            text,
        )
        if match:
            percentage = float(match.group(1))
            if 0 <= percentage <= 100:
                return percentage, "fraud"

    # Generic examples:
    # "Exactly 10% of records should have alert status critical."
    # "Exactly 20% of records should have customer segment premium."
    patterns = [
        rf"\bexactly\s+(\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?"
        rf"(?:rows|records)\s+(?:should\s+have|must\s+have)\s+"
        rf"(?:the\s+)?{column_text}\s+['\"]?([a-z][a-z _-]*?)['\"]?"
        rf"(?=[.,;]|$)",
        rf"\bexactly\s+(\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?"
        rf"(?:rows|records)\s+(?:should\s+be|must be)\s+"
        rf"['\"]?([a-z][a-z _-]*?)['\"]?"
        rf"(?=[.,;]|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            percentage = float(match.group(1))
            label = match.group(2).strip(" '\".,;")
            if 0 <= percentage <= 100 and label:
                return percentage, label

    return None

def detect_iqr_outliers(values, multiplier=1.5):
    """Return a Boolean mask identifying IQR outliers, including zero-IQR data."""
    series = pd.Series(values, dtype="float64")
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr):
        return pd.Series(False, index=series.index)

    if iqr == 0:
        median = series.median()
        return series != median

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper)


def parse_anomaly_percentage(prompt):
    """Return an explicit anomaly percentage, or 2% for an occasional-anomaly request."""
    text = prompt.lower()

    patterns = [
        r"\bexactly\s+(\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?(?:rows|readings|records|values)\s+should\s+be\s+anomalous\b",
        r"\b(\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?(?:rows|readings|records|values)\s+(?:should\s+be\s+)?anomalous\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            percentage = float(match.group(1))
            return percentage if 0 <= percentage <= 100 else None

    if re.search(r"\b(anomal(?:y|ies|ous)|outliers?)\b", text):
        return 2.0

    return None

def detect_unsupported_requirements(prompt):
    """Identify requested behaviors that this engine does not yet implement."""
    text = prompt.lower()
    unsupported = []

    requirement_patterns = [
        (
            r"\b(anomal(?:y|ies|ous)|outlier|outliers)\b",
            "Anomaly/outlier injection is not implemented.",
        ),
        (
            r"\b(correlated|correlation|correlations)\b",
            "Correlated-column generation is not implemented.",
        ),
        (
            r"\b(real[- ]world|realistic relationships|causal relationships)\b",
            "Real-world or causal relationship modeling is not implemented.",
        ),
        (
            r"\b(privacy[- ]preserving|differential privacy)\b",
            "Privacy-preserving generation is not implemented.",
        ),
    ]

    for pattern, message in requirement_patterns:
        if re.search(pattern, text):
            unsupported.append(message)

    return unsupported

def generate_universal_dataset(prompt, seed=42):
    rng = np.random.default_rng(seed)
    row_count = infer_row_count(prompt)
    columns = infer_columns(prompt)

    if not columns:
        columns = ["record_id", "category", "value"]

    device_count = parse_device_count(prompt)
    data = {}

    for column in columns:
        kind = infer_type(column)

        if column == "device_id" and device_count:
            ids = [f"DEV-{i + 1:05d}" for i in range(device_count)]
            data[column] = [ids[i % device_count] for i in range(row_count)]
            rng.shuffle(data[column])

        elif kind == "id":
            prefix = column.upper().replace("_ID", "")
            data[column] = [f"{prefix}-{i + 1:06d}" for i in range(row_count)]

        elif kind == "datetime":
            start = pd.Timestamp("2025-01-01")
            seconds = np.sort(rng.integers(0, 365 * 24 * 3600, size=row_count))
            data[column] = start + pd.to_timedelta(seconds, unit="s")

        elif kind == "numeric":
            bounds = parse_numeric_bounds(prompt, column)
            if bounds:
                low, high = bounds
            elif "temperature" in column:
                low, high = 15, 35
            elif "humidity" in column or "battery" in column:
                low, high = 0, 100
            elif "score" in column:
                low, high = 0, 1
            elif "amount" in column or "salary" in column or "price" in column:
                low, high = 100, 10000
            else:
                low, high = 0, 100
            data[column] = np.round(rng.uniform(low, high, row_count), 2)

        elif parse_exact_percentage(prompt, column):
            percentage, target_label = parse_exact_percentage(prompt, column)
            target_count = round(row_count * percentage / 100)

            if "fraud" in column:
                alternatives = ["legitimate"]
            elif "status" in column:
                alternatives = [
                    value for value in ["normal", "warning", "critical"]
                    if value != target_label
                ]
            elif "segment" in column:
                alternatives = [
                    value for value in ["standard", "basic", "business"]
                    if value != target_label
                ]
            elif "category" in column or "type" in column:
                alternatives = [
                    value for value in ["A", "B", "C"]
                    if value.lower() != target_label.lower()
                ]
            else:
                alternatives = ["other"]

            remaining = row_count - target_count
            labels = [target_label] * target_count
            if remaining:
                labels.extend(
                    rng.choice(alternatives, size=remaining).tolist()
                )
            rng.shuffle(labels)
            data[column] = labels

        elif "status" in column:
            choices = ["normal", "warning", "critical"]
            data[column] = rng.choice(choices, row_count, p=[0.90, 0.08, 0.02])

        elif "label" in column and "fraud" in column:
            exact_rate = parse_exact_percentage(prompt, column)
            if exact_rate:
                percentage, positive_label = exact_rate
                positive_count = round(row_count * percentage / 100)
                labels = (
                    [positive_label] * positive_count
                    + ["legitimate"] * (row_count - positive_count)
                )
                rng.shuffle(labels)
                data[column] = labels
            else:
                data[column] = rng.choice(
                    ["legitimate", "fraud"], row_count, p=[0.97, 0.03]
                )

        elif any(word in column for word in ("type", "category")):
            data[column] = rng.choice(["A", "B", "C"], row_count)

        else:
            data[column] = [f"{column.upper()}-{i + 1:06d}" for i in range(row_count)]

    df = pd.DataFrame(data)

    # Inject reproducible anomalies into numeric telemetry columns.
    anomaly_percentage = parse_anomaly_percentage(prompt)
    anomaly_columns = [
        column for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and any(word in column for word in (
            "temperature", "humidity", "battery", "pressure", "speed"
        ))
    ]
    anomaly_row_indices = []

    if anomaly_percentage is not None and anomaly_columns:
        anomaly_count = round(row_count * anomaly_percentage / 100)

        if anomaly_count > 0:
            anomaly_row_indices = sorted(
                rng.choice(row_count, size=anomaly_count, replace=False).tolist()
            )
            injected = set(anomaly_row_indices)
            normal_rows = [i for i in range(row_count) if i not in injected]

            for column in anomaly_columns:
                bounds = parse_numeric_bounds(prompt, column)
                if bounds:
                    low, high = bounds
                elif "temperature" in column:
                    low, high = 15, 35
                else:
                    low, high = 0, 100

                if high <= low:
                    continue

                # Concentrate normal readings around the midpoint so extremes
                # can be detected, while preserving the requested bounds.
                midpoint = (low + high) / 2
                spread = (high - low) * 0.08
                if normal_rows:
                    baseline = rng.normal(midpoint, spread, len(normal_rows))
                    baseline = np.clip(baseline, low, high)
                    df.loc[normal_rows, column] = np.round(baseline, 2)

                # Alternate between the allowed lower and upper extremes.
                for position, row_index in enumerate(anomaly_row_indices):
                    value = low if position % 2 == 0 else high
                    df.at[row_index, column] = round(float(value), 2)

    checks = [
        {
            "Check": "Requested row count",
            "Passed": len(df) == row_count,
            "Observed": len(df),
            "Requirement": row_count,
        },
        {
            "Check": "No entirely empty columns",
            "Passed": all(not df[c].isna().all() for c in df.columns),
            "Observed": sum(not df[c].isna().all() for c in df.columns),
            "Requirement": len(df.columns),
        },
        {
            "Check": "Inferred schema created",
            "Passed": len(df.columns) > 0,
            "Observed": len(df.columns),
            "Requirement": "At least 1 column",
        },
    ]

    if "device_id" in df.columns and device_count:
        observed_devices = int(df["device_id"].nunique())
        checks.append({
            "Check": "Requested distinct device count",
            "Passed": observed_devices == device_count,
            "Observed": observed_devices,
            "Requirement": device_count,
        })

    for column in df.columns:
        bounds = parse_numeric_bounds(prompt, column)
        if bounds and pd.api.types.is_numeric_dtype(df[column]):
            low, high = bounds
            checks.append({
                "Check": f"{column} within requested bounds",
                "Passed": bool(df[column].between(low, high).all()),
                "Observed": f"{df[column].min()} to {df[column].max()}",
                "Requirement": f"{low} to {high}",
            })

    for column in df.columns:
        exact_rate = parse_exact_percentage(prompt, column)
        if exact_rate:
            percentage, target_label = exact_rate
            expected_count = round(row_count * percentage / 100)
            observed_count = int((df[column].astype(str).str.lower() == target_label.lower()).sum())
            checks.append({
                "Check": f"Exact {target_label} proportion in {column}",
                "Passed": observed_count == expected_count,
                "Observed": f"{observed_count}/{row_count} ({observed_count / row_count * 100:.4g}%)",
                "Requirement": f"{expected_count}/{row_count} ({percentage}%, rounded to whole rows)",
            })

    unsupported_requirements = detect_unsupported_requirements(prompt)

    if anomaly_percentage is not None:
        if anomaly_columns:
            observed_anomalies = len(anomaly_row_indices)
            checks.append({
                "Check": "Requested anomaly injection",
                "Passed": observed_anomalies == round(row_count * anomaly_percentage / 100),
                "Observed": observed_anomalies,
                "Requirement": (
                    f"{round(row_count * anomaly_percentage / 100)} anomalous rows "
                    f"({anomaly_percentage}%, rounded to whole rows)"
                ),
            })

            detected_rows = set()
            detected_by_column = {}

            for column in anomaly_columns:
                mask = detect_iqr_outliers(df[column])
                detected_by_column[column] = int(mask.sum())
                detected_rows.update(df.index[mask].tolist())

            injected_detected = len(
                set(anomaly_row_indices).intersection(detected_rows)
            )

            checks.append({
                "Check": "IQR-detected outliers",
                "Passed": None,
                "Observed": (
                    f"{len(detected_rows)} distinct rows; "
                    f"{injected_detected}/{len(anomaly_row_indices)} injected rows "
                    f"detected; per-column counts: {detected_by_column}"
                ),
                "Requirement": (
                    "Diagnostic only; IQR detection does not by itself prove "
                    "that all injected anomalies are statistically detectable."
                ),
            })

            unsupported_requirements = [
                item for item in unsupported_requirements
                if "Anomaly/outlier injection" not in item
            ]
        else:
            checks.append({
                "Check": "Requested anomaly injection",
                "Passed": None,
                "Observed": "No eligible numeric telemetry columns",
                "Requirement": "Unsupported; not validated",
            })

    for requirement in unsupported_requirements:
        checks.append({
            "Check": requirement,
            "Passed": None,
            "Observed": "Not implemented",
            "Requirement": "Unsupported; not validated",
        })

    report = pd.DataFrame(checks)
    plan = {
        "row_count": row_count,
        "columns": columns,
        "types": {column: infer_type(column) for column in columns},
        "seed": seed,
        "limitations": [
            "This is a heuristic, local schema-inference engine; it does not understand every natural-language constraint.",
            "Only recognized constraints are enforced and validated.",
        ],
        "unsupported_requirements": unsupported_requirements,
        "anomaly_rows": anomaly_row_indices,
        "anomaly_columns": anomaly_columns,
    }

    return df, report, plan
