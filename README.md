# SynthDataGen

Synthetic data generation and workflow simulation built with Python and Streamlit.

## Features
- Generate synthetic datasets across supported domains.
- Inject controlled anomalies and failure scenarios.
- Generate linked employee, payroll, and access records.
- Validate row counts, identifiers, relationships, and injected failures.
- Explore workflow events and validation results.
- Export generated datasets.

## Run locally
Install dependencies with `pip install -r requirements.txt`.
Launch with `streamlit run app.py`.

Configure any required API credentials in a local `.env` file. Never commit credentials.

## Tests
Run `python -m unittest -v test_app test_universal_engine test_workflow_engine test_scenario_engine`.

## License
MIT
