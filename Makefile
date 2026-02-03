.PHONY: install install-dev run test lint format typecheck clean docker-up docker-down docker-logs migrate

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

run:
	uvicorn agentflow.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest --cov=agentflow --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down -v

docker-logs:
	docker compose logs -f

migrate:
	alembic upgrade head
