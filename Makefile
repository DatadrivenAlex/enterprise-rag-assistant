.PHONY: install test run

install:
	pip install -r requirements-dev.txt

test:
	pytest

run:
	PYTHONPATH=src uvicorn enterprise_rag.main:app --host 0.0.0.0 --port 8000
