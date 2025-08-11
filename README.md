# Daily A-shares Analyzer

This project fetches the highest-turnover A-share stocks each trading day,
computes several technical indicators and produces HTML/CSV reports.

## Run locally

```bash
pip install -r requirements.txt
python run_daily.py
```

Reports will be saved under the `reports/` directory.

## GitHub Actions

The workflow in `.github/workflows/daily.yml` runs every weekday at 09:30 UTC.
It installs dependencies, executes the analyzer and uploads the generated
reports as workflow artifacts. The workflow can also be triggered manually via
the **Run workflow** button.
