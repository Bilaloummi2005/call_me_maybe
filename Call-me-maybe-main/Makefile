install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ src/__pycache__ .mypy_cache *.egg-info

lint:
	uv run flake8 . --exclude .venv,venv,__pycache__,.mypy_cache,build,dist,*.egg-info,llm_sdk
	uv run mypy . --exclude '(^|.*/)(\.venv|venv|build|dist)/|.*\.egg-info/' \
		--warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --exclude .venv,venv,__pycache__,.mypy_cache,build,dist,*.egg-info,llm_sdk
	uv run mypy . --exclude '(^|.*/)(\.venv|venv|build|dist)/|.*\.egg-info/' --strict

.PHONY: install run debug clean lint lint-strict