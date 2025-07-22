lint_check:
	ruff check .
	black --check --diff .

reformat_code:
	ruff check . --fix
	black .
	isort . --profile black

run_tests:
	pytest tests