# Daily A-shares Analyzer

This project analyses the most active CN A-share stocks through a
**industry → funds → fundamentals → technicals → entry** pipeline.  It
surfaces the strongest industries, scores individual stocks and produces
HTML/CSV reports.

## Pipeline overview

1. **Industry heat** – Eastmoney industry boards are queried via AkShare.
   Turnover growth (Δ5d) and 5‑day return are combined into an
   ``IndustryScore``.  Only stocks belonging to the top‑K industries are kept.
2. **Funds/quant involvement** – northbound flows, institution activity and
   ETF share changes are normalised and summed into ``FlowScore``.  Missing
   values are imputed by the cross‑sectional median.
3. **Fundamentals** – quarterly fundamentals are fetched with disclosure-date
   filters.  TTM/YoY features are industry‑neutralised and form ``FundScore``.
   Raw fundamentals are cached under ``./cache/fundamentals/YYYY-MM-DD``.
4. **Technicals & entry** – moving averages, RSI, volume, ATR and 20‑day high
   conditions produce ``TechScore``.  ``EntryFlag`` checks for a valid entry
   zone (MA20>MA60, close near MA20, RSI14 between 50–65 and a recent high).
5. **Ranking** – a weighted z-score composite of the above factors (and an
   optional ML up-probability) ranks the final universe.  The HTML report shows
   a "Top Industries" section followed by the top stocks table.

## CLI

```bash
pip install -r requirements.txt
python run_daily.py [--top 300] [--ind-top-k 5] [--w-ind 0.30] \
    [--w-flow 0.30] [--w-fund 0.20] [--w-tech 0.20] [--w-ml 0.00] \
    [--no-ml] [--cache-dir cache]
```

- ``--top`` – number of stocks to scan (default ``300``).
- ``--ind-top-k`` – keep stocks in the top‑K industries by ``IndustryScore``
  (default ``5``).
- ``--w-ind``/``--w-flow``/``--w-fund``/``--w-tech``/``--w-ml`` – weights for
  the composite ranking.
- ``--no-ml`` – disable the optional ML up-probability signal.
- ``--cache-dir`` – base directory for caches.

Reports are written under ``reports/`` as both HTML and CSV.  They contain the
industry table and the stock table with ``IndustryScore``, ``FlowScore``,
``FundScore``, ``TechScore``, ``EntryFlag`` and ``Composite`` columns.

## Data policy

All data is fetched with a one‑day lag (T‑1) and is for research purposes only.
Please respect the terms of use of the upstream data providers.

## GitHub Actions

The workflow in `.github/workflows/daily.yml` runs every weekday at 09:30 UTC.
It installs dependencies, executes the analyzer and uploads the generated
reports as workflow artifacts. The workflow can also be triggered manually via
the **Run workflow** button.
