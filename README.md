# DLMDSPWP01 — Programming with Python: Curve-Fitting Assignment

**Author:** Alen Nukovic — IU International University of Applied Sciences, M.Sc. Artificial Intelligence
**Module:** DLMDSPWP01 — Programming with Python
**Submission target:** June 2026

This repository contains the implementation accompanying the written assignment for the module *Programming with Python* (DLMDSPWP01). It selects the four best-fitting ideal functions from a pool of fifty for four training datasets using the least-squares criterion, validates test points against the chosen functions using a √2 deviation factor, persists all data in a SQLite database via SQLAlchemy, and visualises the results with Bokeh.

## Repository Structure

```
dlmdspwp01-curve-fitting/
├── src/                Source code
│   ├── exceptions.py   Custom exception hierarchy
│   ├── data_loader.py  CSV loaders (inheritance hierarchy)
│   ├── database.py     SQLAlchemy ORM models and engine setup
│   ├── analysis.py     Ideal-function selector & test-point mapper
│   ├── visualization.py Bokeh plotting
│   └── main.py         End-to-end orchestrator
├── tests/              pytest unit tests
├── data/               Input CSVs (train.csv, ideal.csv, test.csv)
├── docs/               LaTeX manuscript of the written assignment
├── notebooks/          Exploratory notebooks (not part of submission)
├── pyproject.toml      Project metadata and dependencies
└── README.md           This file
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python -m src.main
```

This produces:
- `output/results.db` — SQLite database with three tables.
- `output/plot.html` — interactive Bokeh visualisation.

## Tests

```bash
pytest
```

## Style

Code follows **PEP 8** (Python Enhancement Proposal 8). Style is enforced via `ruff check src/`.
