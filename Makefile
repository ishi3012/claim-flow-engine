# Makefile for streamlined developer commands

.PHONY: lint format typecheck test test-fast test-slow check all

# Run Ruff linter
lint:
	pre-commit run ruff --all-files || true

# Format code using Black
format:
	pre-commit run black --all-files || true

# Type check using mypy
typecheck:
	pre-commit run mypy --all-files || true

# Run all tests with coverage
test:
	pytest --cov=claimflowengine --cov-report=term-missing

# Run only fast (non-slow marked) tests
test-fast:
	pytest -m "not slow"

# Run only slow tests
test-slow:
	pytest -m slow

# Run all pre-commit hooks
check:
	pre-commit run --all-files || true

# Run all checks (format → lint → typecheck → test)
all: format lint typecheck test

coverage-html:
	pytest --cov=claimflowengine --cov-report=html
