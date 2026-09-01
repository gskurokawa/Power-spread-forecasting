# Platform & Infrastructure

This scaffold re-platforms the power-spread project onto Docker, Airflow, FastAPI,
and Azure — each doing its real job on the existing system. Stand it up yourself and
you have genuine, documentable experience with all four, tied to one coherent project
rather than four throwaway demos.

## Architecture

```
                         ENTSO-E Transparency API
                                   │
                    ┌──────────────▼───────────────┐
   Airflow DAG ───► │  daily_update.py (pipeline)  │  (Dockerfile.pipeline)
  (scheduler +      │  fetch → build → retrain →   │
   retries + gate)  │  write forecast              │
                    └──────────────┬───────────────┘
                                   │ writes
                          ┌────────▼─────────┐
                          │  Neon Postgres   │  (predictions, modelling_table)
                          └────────┬─────────┘
                                   │ reads
              ┌────────────────────┼─────────────────────┐
              │                                           │
    ┌─────────▼──────────┐                     ┌──────────▼─────────┐
    │  FastAPI service   │  (Dockerfile.api)   │  Streamlit app     │
    │  /forecast /drivers│  → Azure Container  │  (existing UI)     │
    └────────────────────┘     Apps (public)   └────────────────────┘
```

## What each piece is

| Tech | File(s) | Role | What it demonstrates |
|---|---|---|---|
| **Docker** | `Dockerfile.api`, `Dockerfile.pipeline`, `.dockerignore`, `docker-compose.yml` | Containerize the API and the pipeline; local dev stack (Postgres + API) | Reproducible builds, layer caching, multi-service compose, secrets via env |
| **Airflow** | `airflow/dags/power_spread_dag.py`, `airflow/README.md` | Orchestrate the daily run: gate → run → verify, with retries & history | Real scheduling/orchestration, sensors/short-circuit, idempotent tasks |
| **FastAPI** | `api/main.py`, `api/requirements.txt` | Serve forecasts and SHAP drivers over HTTP | Typed REST API, model serving, OpenAPI docs, DB access, CORS |
| **Azure** | `deploy/azure_containerapps.md`, `deploy/deploy_azure.sh` | Deploy the API container to a managed, scale-to-zero service | Cloud deploy, container registry, managed ingress/secrets, cost control |

## Suggested order (fastest first)

1. **Docker + FastAPI** (½ day). `docker compose up --build`, hit `/docs`. This alone
   gives you Docker + FastAPI experience and is fully local/free.
2. **Airflow** (½–1 day). Bring up the official compose, mount the repo, enable the
   DAG, trigger a run. Upgrades your "scheduling was flaky" story into "orchestrated
   with Airflow."
3. **Azure** (½ day). Push the API image to Docker Hub, `deploy_azure.sh`, share the
   public URL. Scale-to-zero keeps it ~free; `az group delete` when done.

## Run locally

```bash
# API + throwaway Postgres
docker compose up --build          # http://localhost:8000/docs

# API against your real Neon DB (no local Postgres)
DATABASE_URL='postgresql+psycopg2://...' uvicorn api.main:app --reload

# Airflow
cd airflow && see README.md
```

## Honesty note

The value is in *running* this, not in the files existing. Once you've done a
`docker build`, watched the DAG succeed in the Airflow UI, and opened the API on a
public Azure URL, every line below is something you did and can talk through.

## CV additions once you've run it

- **Technical Skills line** → "Python, SQL/PostgreSQL, Docker, Airflow, FastAPI, Azure (Container Apps), Git."
- **Power-spread entry** → add a sentence: "Containerised with Docker, orchestrated
  with Airflow, served via a FastAPI REST API, and deployed on Azure Container Apps."
- Only claim what you've actually stood up. If you do Docker + FastAPI + Airflow but
  skip the cloud step, list those three and leave Azure off until it's live.

## "+ extras" (optional standalone pieces)

If a specific job ad wants something this doesn't cover:
- **AWS** variant: the same API deploys to AWS App Runner or Lambda (Mangum) with
  minimal change — a second cloud on the CV if a role names AWS.
- **CI/CD**: a GitHub Actions workflow that builds and pushes the image on every
  commit (you already know Actions) rounds out the DevOps story.
