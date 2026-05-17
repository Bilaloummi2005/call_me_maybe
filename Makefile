install:
	uv sync

run:
	uv run python -m src 

debug:
	uv run python -m pdb -m src \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calls.json

lint:
	mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	flake8 src/

lint-strict:
	mypy src/ --strict
	flake8 src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +

.PHONY: install run debug lint lint-strict clean
