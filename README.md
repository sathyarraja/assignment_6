# Netflix Medallion Pipeline

End-to-end production ELT pipeline implementing the Medallion Architecture
(Bronze → Silver → Gold) for the Netflix Prize dataset.

## Architecture

```
Raw Sources          Bronze              Silver              Gold
─────────────────    ──────────────────  ──────────────────  ──────────────────
Netflix Ratings ───► ratings/           ratings (Delta)     fact_ratings
TMDB Movies     ───► movies/        ──► movies              dim_users
Netflix Users   ───► users/             users (SCD2)        dim_movies
                     quarantine/        genres              dim_dates
                                        enriched_ratings    daily_engagement
                                        dq_metrics          rfm_segmentation
                                                            retention_cohorts
                                                            top_movies_*
                                                            *_kpis
```

## Stack

| Component | Technology |
|---|---|
| Processing | PySpark 3.5.1 + Delta Lake 3.1.0 |
| Orchestration | Apache Airflow 2.9.1 |
| Data Quality | Custom Expectations + YAML rules |
| Storage | Local Parquet + Delta Lake |
| Language | Python 3.9 |

## Project Structure

```
netflix_pipeline/
├── dags/
│   └── netflix_pipeline_dag.py     # Airflow DAG (Q30-Q35)
├── scripts/
│   ├── bronze_ingestion.py         # Part A (Q1-Q5)
│   ├── silver_transforms.py        # Part B (Q6-Q13)
│   ├── gold_transforms.py          # Part C (Q14-Q19)
│   ├── incremental_processing.py   # Part D (Q20-Q24)
│   ├── dq_framework.py             # Part E (Q25-Q29)
│   ├── generate_sample_data.py     # Sample data generator
│   └── run_bronze.py               # Bronze runner
├── utils/
│   └── spark_session.py            # SparkSession factory
├── tests/
│   ├── test_bronze.py              # 13 tests
│   ├── test_silver.py              # 16 tests
│   ├── test_gold.py                # 26 tests
│   ├── test_incremental.py         # 19 tests
│   ├── test_dq_framework.py        # 24 tests
│   └── test_dag.py                 # 22 tests
├── config/
│   ├── pipeline_config.yaml        # Pipeline settings
│   └── dq_rules.yaml               # DQ rules (Q25)
├── expectations/                   # GE-style suite results (Q26)
├── logs/
│   ├── watermarks.json             # Incremental watermarks (Q20)
│   ├── checkpoints.json            # Pipeline checkpoints (Q23)
│   ├── dq_metrics.json             # DQ metrics history (Q12)
│   └── reports/
│       ├── dq_dashboard.html       # DQ dashboard (Q27)
│       ├── monitoring_dashboard.html # Pipeline monitoring (Q35)
│       └── profile_*.html          # Data profiles (Q29)
├── bronze/                         # Raw ingested data
├── silver/                         # Cleaned data
├── gold/                           # Business-ready data
├── data/raw/                       # Source files
├── docker-compose.yaml             # Airflow services
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Setup environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export SPARK_LOCAL_IP=127.0.0.1

# 2. Generate sample data
python3 scripts/generate_sample_data.py

# 3. Run full pipeline
python3 scripts/run_bronze.py
python3 scripts/silver_transforms.py
python3 scripts/gold_transforms.py
python3 scripts/incremental_processing.py
python3 scripts/dq_framework.py

# 4. Run all tests (120 total)
python3 -m pytest tests/ -v

# 5. Start Airflow
docker compose up -d
open http://localhost:8080   # admin / admin
```

## Airflow DAG

The DAG `netflix_medallion_pipeline` runs daily at 2 AM with:
- **Parallel bronze ingestion** (ratings, movies, users run simultaneously)
- **3 retries with exponential backoff** on every task
- **9 AM SLA** on the gold task (7-hour window from 2 AM start)
- **Monitoring dashboard** built after every run

```
start
  ├── ingest_ratings ─┐
  ├── ingest_movies  ─┼── bronze_complete ── run_silver ── run_dq_checks ── run_gold
  └── ingest_users   ─┘                                                  ├── run_incremental
                                                                          ├── build_monitoring
                                                                          └── end
```

## Test Results

```
120 passed in ~90s
  Part A (Bronze):       13/13
  Part B (Silver):       16/16
  Part C (Gold):         26/26
  Part D (Incremental):  19/19
  Part E (DQ Framework): 24/24
  Part F (Orchestration):22/22
```

## Data Quality

Rules defined in `config/dq_rules.yaml`. Checks run automatically after Silver:
- **Null checks** on mandatory columns
- **Range checks** (rating 1.0–5.0)
- **Referential integrity** (user_id, movie_id FK checks)
- **Duplicate detection** by composite key
- **Row count thresholds** per table

Results written to `expectations/` and visualised in `logs/reports/dq_dashboard.html`.
