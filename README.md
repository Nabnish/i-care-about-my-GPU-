# GPU Bit-Flip Risk Monitor

A lightweight simulator and monitoring pipeline for estimating GPU bit-flip risk
from telemetry while reporting how trustworthy that estimate is.

## Project layout

- `config/` — simulator, fault, and experiment settings.
- `simulator/` — GPU behavior, fault injection, telemetry, and monitoring.
- `preprocessing/` — cleaning, windowing, and feature extraction.
- `models/` — risk estimation, observability, and calibration.
- `alerts/` — risk/confidence alert decisions.
- `evaluation/` — metrics, calibration evaluation, and experiments.
- `data/` — local datasets and generated simulation output.
- `notebooks/` — exploratory analysis.
- `tests/` — automated tests.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and build order.

## Setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the test suite from the repository root:

```bash
python -m pytest
```

This repository is scaffolded for development; simulation and modeling
components will be implemented incrementally.
