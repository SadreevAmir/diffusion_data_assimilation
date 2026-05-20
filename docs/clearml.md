# ClearML

`concat_conditioning` experiments always initialize a ClearML task and fail fast if credentials are missing.
ClearML `project_name` and `task_name` always come from the top-level run config fields, not from `.env` and not from the method config.

Create `.env` from `.env.example` and fill in real credentials:

```text
CLEARML_API_ACCESS_KEY=...
CLEARML_API_SECRET_KEY=...
CLEARML_API_HOST=...
```

`docker-compose.yaml` uses `env_file: .env`, so variables from `.env` are propagated into the container. The compose file also explicitly lists the standard ClearML variables under `environment` so they are visible to the service configuration.

Run the ClearML-reported smoke experiment inside the container or another environment with project dependencies installed:

```bash
python -m assim_lib.main --config config/experiments/smoke_concat_conditioning_4f.json
```
