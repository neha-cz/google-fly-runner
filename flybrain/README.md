# flybrain

Python side of FlyRunner: connectome pipeline, substrate model, training. See the repo root
README for setup and the design doc (`../DESIGN.md`) for the rationale.

```
uv sync --extra dev
uv run flybrain bench-backend
export NEUPRINT_APPLICATION_CREDENTIALS=<token>   # from a neuprint.janelia.org account
uv run flybrain fetch-subgraph --dry-run
uv run pytest
```
