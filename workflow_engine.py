"""Deterministic, local-first engine for stateful synthetic workflows."""

from copy import deepcopy


class WorkflowError(ValueError):
    """Raised when a workflow specification cannot be executed."""


def _ordered_events(events):
    """Return a stable topological ordering of workflow events."""
    by_id = {}

    for event in events:
        event_id = event.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise WorkflowError("Every event needs a non-empty string id.")
        if event_id in by_id:
            raise WorkflowError(f"Duplicate event id: {event_id}")
        by_id[event_id] = event

    completed = set()
    ordered = []

    while len(ordered) < len(events):
        progressed = False

        for event in events:
            event_id = event["id"]
            if event_id in completed:
                continue

            dependencies = event.get("depends_on", [])
            if not isinstance(dependencies, list):
                raise WorkflowError(
                    f"Event '{event_id}': depends_on must be a list."
                )

            unknown = set(dependencies) - set(by_id)
            if unknown:
                raise WorkflowError(
                    f"Event '{event_id}' has unknown dependencies: "
                    f"{sorted(unknown)}"
                )

            if set(dependencies).issubset(completed):
                ordered.append(event)
                completed.add(event_id)
                progressed = True

        if not progressed:
            remaining = [
                event["id"] for event in events
                if event["id"] not in completed
            ]
            raise WorkflowError(
                f"Dependency cycle detected among events: {remaining}"
            )

    return ordered


def _matches(record, conditions):
    return all(record.get(key) == value for key, value in conditions.items())


def _validate_invariants(tables, invariants):
    results = []

    for rule in invariants:
        rule_type = rule.get("type")
        name = rule.get("name", rule_type or "Unnamed invariant")

        if rule_type == "unique":
            table = rule["table"]
            field = rule["field"]
            records = tables.get(table, [])
            values = [row.get(field) for row in records]
            passed = all(value is not None for value in values)
            passed = passed and len(values) == len(set(values))
            observed = f"{len(set(values))} distinct values across {len(values)} rows"

        elif rule_type == "foreign_key":
            table = rule["table"]
            field = rule["field"]
            ref_table = rule["references_table"]
            ref_field = rule["references_field"]
            allowed = {
                row.get(ref_field) for row in tables.get(ref_table, [])
            }
            values = [
                row.get(field) for row in tables.get(table, [])
                if row.get(field) is not None
            ]
            invalid = [value for value in values if value not in allowed]
            passed = len(invalid) == 0
            observed = f"{len(invalid)} invalid references"

        elif rule_type == "required":
            table = rule["table"]
            field = rule["field"]
            records = tables.get(table, [])
            missing = sum(
                row.get(field) is None or row.get(field) == ""
                for row in records
            )
            passed = missing == 0
            observed = f"{missing} missing values"

        else:
            results.append({
                "name": name,
                "passed": None,
                "observed": "Unsupported invariant type",
                "requirement": rule_type,
            })
            continue

        results.append({
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "requirement": deepcopy(rule),
        })

    return results


def run_workflow(spec, seed=42):
    """
    Execute a declarative workflow against an isolated in-memory state.

    Supported event actions:
      - create: add a record to a table
      - update: update all records matching a where clause

    Supported invariant types:
      - unique
      - foreign_key
      - required

    The input specification is never mutated. Identical inputs produce
    identical results; seed is recorded for reproducibility and future
    stochastic event generators.
    """
    if not isinstance(spec, dict):
        raise WorkflowError("Workflow specification must be a dictionary.")

    initial_tables = spec.get("initial_tables", {})
    events = spec.get("events", [])
    invariants = spec.get("invariants", [])

    if not isinstance(initial_tables, dict):
        raise WorkflowError("initial_tables must be a dictionary.")
    if not isinstance(events, list):
        raise WorkflowError("events must be a list.")
    if not isinstance(invariants, list):
        raise WorkflowError("invariants must be a list.")

    tables = deepcopy(initial_tables)

    for table, records in tables.items():
        if not isinstance(table, str) or not isinstance(records, list):
            raise WorkflowError(
                "Each initial table must map a string name to a list of records."
            )
        if not all(isinstance(row, dict) for row in records):
            raise WorkflowError(f"Every record in '{table}' must be a dictionary.")

    event_log = []

    for event in _ordered_events(events):
        event_id = event["id"]
        action = event.get("action")
        table = event.get("table")

        if not isinstance(table, str) or not table.strip():
            raise WorkflowError(f"Event '{event_id}' needs a table name.")

        tables.setdefault(table, [])

        if action == "create":
            record = event.get("record")
            if not isinstance(record, dict):
                raise WorkflowError(
                    f"Event '{event_id}': create requires a record dictionary."
                )

            tables[table].append(deepcopy(record))
            affected = 1

        elif action == "update":
            conditions = event.get("where")
            changes = event.get("set")

            if not isinstance(conditions, dict) or not conditions:
                raise WorkflowError(
                    f"Event '{event_id}': update requires a non-empty where clause."
                )
            if not isinstance(changes, dict) or not changes:
                raise WorkflowError(
                    f"Event '{event_id}': update requires a non-empty set clause."
                )

            affected = 0
            for record in tables[table]:
                if _matches(record, conditions):
                    record.update(deepcopy(changes))
                    affected += 1

            if affected == 0:
                raise WorkflowError(
                    f"Event '{event_id}' updated no records in '{table}'."
                )

        else:
            raise WorkflowError(
                f"Event '{event_id}': unsupported action '{action}'."
            )

        event_log.append({
            "event_id": event_id,
            "action": action,
            "table": table,
            "records_affected": affected,
            "status": "completed",
        })

    validation = _validate_invariants(tables, invariants)

    return {
        "workflow_name": spec.get("name", "Unnamed workflow"),
        "seed": seed,
        "tables": tables,
        "event_log": event_log,
        "validation": validation,
        "passed": all(item["passed"] is True for item in validation),
    }
