PYTHON = uv run python
SRC = src

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	$(PYTHON) -m src

debug:
	$(PYTHON) -m pdb -m src

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache data/.mypy_cache
	rm -rf data/processed data/output

lint:
	uv run flake8 $(SRC)
	uv run mypy $(SRC) --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 $(SRC)
	uv run mypy $(SRC) --strict