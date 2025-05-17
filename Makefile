# Makefile for streamlined developer commands

.PHONY: lint format typecheck test check all

lint:
	pre-commit run ruff --all-files

format:
	pre-commit run black --all-files

typecheck:
	pre-commit run mypy --all-files

test:
	pytest

check:
	pre-commit run --all-files

all: format lint typecheck test
