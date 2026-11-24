.PHONY: up airflow down data pipeline train api web test

up:        ## full stack (PostgreSQL + MongoDB + API + dashboard) on http://localhost:3000
	docker compose up --build

airflow:   ## same + Airflow (DAG aerosupply_daily_risk) on http://localhost:8080
	docker compose --profile airflow up --build

down:
	docker compose down

data:      ## regenerate the synthetic ERP / SRM exports and news signals
	cd backend && python -m aerosupply.data.generate

pipeline:  ## daily run: load stores, detect, forecast, score, optimise (SQLite + embedded Mongo by default)
	cd backend && python -m aerosupply.pipeline.run

train:     ## retrain and re-evaluate every model -> backend/artifacts (~2-3 min)
	cd backend && python -m aerosupply.pipeline.run --retrain

api:
	cd backend && uvicorn aerosupply.api.main:app --reload

web:
	cd frontend && npm run dev

test:
	cd backend && python -m pytest -q
