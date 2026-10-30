"""Nemotron-powered workflow specification interpreter."""

import json
import os
import re

import streamlit as st
from openai import OpenAI

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"

BASE_URL = bytes([
    104, 116, 116, 112, 115, 58, 47, 47, 97, 112, 105, 46,
    116, 111, 107, 101, 110, 102, 97, 99, 116, 111, 114, 121,
    46, 110, 101, 98, 105, 117, 115, 46, 99, 111, 109, 47,
    118, 49, 47
]).decode()


def interpret_workflow_with_nemotron(prompt):
    api_key = os.getenv("NEBIUS_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["NEBIUS_API_KEY"]
        except (KeyError, FileNotFoundError):
            pass

    if not api_key:
        raise RuntimeError("NEBIUS_API_KEY is not configured.")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    system_prompt = """
You convert workflow descriptions into JSON specifications.

Return JSON only, with these keys:
name, initial_tables, events, invariants.

Rules:
- Maximum 10 events.
- Each event must have id, action, table.
- Only use create and update actions.
- Create events must include a record object.
- Update events must include where and values objects.
- Events may include depends_on as a list.
- Initial tables must contain lists of record objects.
- Every invariant must be an object with a type and table.\n- Use only unique, foreign_key, or required invariant types.\n- unique and required need a field.\n- foreign_key needs field, references_table, and references_field.\n- Use an empty invariants list if no checks are needed.
- Do not generate code, SQL, scripts, or explanations.
- Use exactly 3 events and no more than 2 invariants. Keep every record minimal. Return the complete JSON immediately.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=6000,
    )

    message = response.choices[0].message
    content = message.content
    if not content or not content.strip():
        reason = response.choices[0].finish_reason
        reasoning = getattr(message, "reasoning_content", None)
        tools = getattr(message, "tool_calls", None)
        raise ValueError(
            f"Empty model response. Finish reason: {reason}; "
            f"reasoning present: {bool(reasoning)}; "
            f"tool calls present: {bool(tools)}. "
            "Check model output limits or response format."
        )
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)

    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Nemotron did not return a JSON object. Please try again.")
    content = content[start:end + 1]
    try:
        spec = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Nemotron returned malformed JSON near character {exc.pos}."
        ) from exc

    tables = spec.get("initial_tables")
    if isinstance(tables, list):
        converted = {}
        for item in tables:
            if not isinstance(item, dict):
                raise ValueError("AI returned an invalid table definition.")
            table_name = item.get("name", item.get("table"))
            records = item.get("records", item.get("rows", []))
            if not isinstance(table_name, str) or not isinstance(records, list):
                raise ValueError("AI table definitions need a name and records list.")
            converted[table_name] = records
        spec["initial_tables"] = converted
    elif isinstance(tables, str):
        try:
            spec["initial_tables"] = json.loads(tables)
        except json.JSONDecodeError as exc:
            raise ValueError("AI returned invalid initial_tables JSON.") from exc

    validate_workflow_spec(spec)
    return spec


def validate_workflow_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("Workflow must be a JSON object.")

    required = {"name", "initial_tables", "events", "invariants"}
    if not required.issubset(spec):
        raise ValueError("Workflow is missing required fields.")

    tables = spec["initial_tables"]
    events = spec["events"]
    invariants = spec["invariants"]

    if not isinstance(tables, dict):
        raise ValueError("initial_tables must be an object.")
    if not isinstance(events, list) or len(events) > 10:
        raise ValueError("Workflow must contain 0 to 10 events.")
    if not isinstance(invariants, list):
        raise ValueError("invariants must be a list.")

    for table, records in tables.items():
        if not isinstance(table, str) or not isinstance(records, list):
            raise ValueError("Each table must contain a list of records.")
        if len(records) > 100:
            raise ValueError("Initial tables are limited to 100 records.")
        if not all(isinstance(record, dict) for record in records):
            raise ValueError("Table records must be JSON objects.")

    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Each event must be an object.")
        if event.get("action") not in {"create", "update"}:
            raise ValueError("Only create and update actions are supported.")
        if event.get("table") not in tables:
            raise ValueError("An event refers to an unknown table.")

        if event["action"] == "create":
            if not isinstance(event.get("record"), dict):
                raise ValueError("Create events require a record object.")

        if event["action"] == "update":
            if not isinstance(event.get("where"), dict):
                raise ValueError("Update events require a where object.")
            if not isinstance(event.get("values"), dict):
                raise ValueError("Update events require a values object.")

    allowed_rules = {"unique", "foreign_key", "required"}
    for rule in invariants:
        if not isinstance(rule, dict):
            raise ValueError("Each invariant must be an object.")
        if rule.get("type") not in allowed_rules:
            raise ValueError("Unsupported invariant type: " + str(rule.get("type")))
        if not isinstance(rule.get("table"), str) or rule["table"] not in tables:
            raise ValueError("Invariant refers to an unknown table.")
        if rule["type"] in {"unique", "required"} and not isinstance(rule.get("field"), str):
            raise ValueError("Unique and required invariants need a field.")
        if rule["type"] == "foreign_key":
            if not all(isinstance(rule.get(k), str) for k in ("field", "references_table", "references_field")):
                raise ValueError("Foreign-key invariants need field and reference fields.")
            if rule["references_table"] not in tables:
                raise ValueError("Foreign-key invariant refers to an unknown table.")

    return True
