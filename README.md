# ExperimentIQ

**Production-grade A/B testing and causal inference engine.**

---

## Features

- **Sequential Testing** — O'Brien–Fleming alpha spending for interim looks
- **CUPED Variance Reduction** — Pre-experiment covariate adjustment for more powerful tests
- **Causal Forest (EconML)** — Heterogeneous treatment effects and ATE via Double ML
- **Synthetic Control** — Counterfactual estimation for single treated unit + donors
- **FastAPI backend** — Typed JSON API with Pydantic validation
- **Streamlit dashboard** — CSV upload, Plotly charts, and plain-English results

---

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────────┐     ┌─────────────┐     ┌─────────────────────┐
│  CSV/Data   │────▶│  Core Modules (stats, cuped, causal, synthetic)   │────▶│  FastAPI    │────▶│  Streamlit Dashboard │
└─────────────┘     └──────────────────────────────────────────────────┘     └─────────────┘     └─────────────────────┘
                              │                            │                         │
                              │  two_sample_ttest           │  causal_forest_ate      │  POST /experiment/*
                              │  sequential_test            │  synthetic_control      │  GET  /health
                              │  apply_cuped                │  chi_square_test        │
                              └────────────────────────────┴─────────────────────────┘
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Causal Forest (EconML)
pip install econml

# Start the API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal: start the dashboard
streamlit run dashboard/app.py
```

- **API:** http://localhost:8000  
- **Docs:** http://localhost:8000/docs  
- **Dashboard:** http://localhost:8501  

---

## API Reference

### POST `/experiment/ttest`

Two-sample t-test (control vs treatment).

```bash
curl -X POST http://localhost:8000/experiment/ttest \
  -H "Content-Type: application/json" \
  -d '{
    "control": [10.1, 9.8, 10.2, 9.5, 10.0],
    "treatment": [10.5, 10.9, 10.7, 10.6, 10.8],
    "alpha": 0.05,
    "equal_var": true
  }'
```

**Example response:**

```json
{
  "method": "ttest",
  "effect_size": 1.234,
  "p_value": 0.012,
  "confidence_interval": [0.15, 0.89],
  "is_significant": true,
  "metadata": { "equal_var": true, "alpha": 0.05 }
}
```

### POST `/experiment/causal`

Causal Forest ATE (covariates **X**, treatment **T**, outcome **Y**).

```bash
curl -X POST http://localhost:8000/experiment/causal \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
    "T": [0, 1, 1],
    "Y": [1.2, 2.8, 3.1],
    "random_state": 42,
    "n_estimators": 28
  }'
```

**Example response:**

```json
{
  "method": "causal",
  "effect_size": 1.95,
  "p_value": null,
  "confidence_interval": null,
  "is_significant": null,
  "metadata": { "ate": 1.95, "random_state": 42, "n_estimators": 28 }
}
```

Other endpoints: `POST /experiment/sequential`, `POST /experiment/cuped`, `POST /experiment/synthetic-control`, `GET /health`.

---

## Why ExperimentIQ?

ExperimentIQ goes beyond basic A/B testing:

- **CUPED** uses pre-experiment data to reduce variance, so you need fewer users to detect the same lift—without changing the treatment effect estimate.
- **Causal inference** (EconML Causal Forest) controls for confounders and estimates the *causal* effect of treatment, not just association, so you can trust results in observational or high-dimensional settings.
- **Sequential testing** lets you peek at results at interim looks while controlling type I error via O'Brien–Fleming bounds.
- **Synthetic control** gives a principled counterfactual for one treated unit (e.g. a region or segment) when you have many donor units and a clear intervention date.

Use ExperimentIQ when you need **production-grade** experiment analysis: reproducible, API-driven, and dashboard-ready.

---

## Tech stack

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)  
[![EconML](https://img.shields.io/badge/EconML-0.16-orange.svg)](https://econml.azurewebsites.net/)  
[![pytest](https://img.shields.io/badge/pytest-7.4+-purple.svg)](https://pytest.org/)

- **Python** — Core runtime  
- **FastAPI** — REST API and OpenAPI docs  
- **Streamlit** — Interactive dashboard  
- **EconML** — Causal Forest / DML  
- **pytest** — Test suite  

---

## Project structure

```
ExperimentIQ/
├── api/              # FastAPI app, schemas
├── core/             # stats, cuped, causal, synthetic_control
├── dashboard/        # Streamlit app
├── tests/            # pytest (test_stats, test_cuped, test_causal)
├── requirements.txt
├── .env.example
└── README.md
```

## License

MIT
