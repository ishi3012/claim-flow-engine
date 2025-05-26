# pre_push_checks.sh
#!/bin/bash

set -e

echo "🔍 Running Black..."
black . --check

echo "🔍 Running Ruff..."
ruff check .

echo "🔍 Running mypy..."
mypy .

echo "🧪 Running Pytest..."
pytest
