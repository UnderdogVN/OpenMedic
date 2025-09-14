lint_check:
	ruff check .
	black --check --diff .


reformat_code:
	ruff check . --fix
	black .
	isort . --profile black


run_tests:
	pytest tests


install_core:
	uv pip install -e .


install_dev:
	uv pip install '.[dev]'