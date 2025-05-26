lint:
	black . && ruff check .

typecheck:
	mypy .

test:
	pytest

check: lint typecheck test