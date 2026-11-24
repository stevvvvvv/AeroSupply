#!/bin/sh
set -e
# Daily pipeline once at start-up (generates the synthetic exports on first run, loads PostgreSQL
# and MongoDB, scores everything). Trained models ship in artifacts/; retrain if they are missing.
python -m aerosupply.pipeline.run
exec uvicorn aerosupply.api.main:app --host 0.0.0.0 --port 8000
