# Makefile for streamlined developer commands

.PHONY: lint format typecheck test check all

lint:
	pre-commit run ruff --all-files || true

format:
	pre-commit run black --all-files || true

typecheck:
	pre-commit run mypy --all-files || true

test:
	pytest

check:
	pre-commit run --all-files || true

all: format lint typecheck test
