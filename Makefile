VENV?=.venv
PYTHON?=$(VENV)/bin/python
PIP?=$(VENV)/bin/pip

.PHONY: venv install setup seed quality search context l1 l2 l3 viz benchmark test migrate

venv:
	python -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

setup:
	$(PYTHON) main.py setup

seed:
	$(PYTHON) main.py seed

quality:
	$(PYTHON) main.py quality

search:
	$(PYTHON) main.py search-demo

context:
	$(PYTHON) main.py context "Fractal Memory"

l1:
	$(PYTHON) main.py l1 --query "Fractal Memory" --hours 24

l2:
	$(PYTHON) main.py l2 "Sergey"

l3:
	$(PYTHON) main.py l3 "Fractal Memory"

viz:
	$(PYTHON) main.py viz-export --output visualization/graph_data.json

benchmark:
	$(PYTHON) main.py benchmark

test:
	$(PYTHON) -m pytest -q

migrate:
	$(PYTHON) main.py migrate

web:
	$(PYTHON) -m uvicorn app:app --host 0.0.0.0 --port 8000
# Dockerized workflow
.PHONY: dc-build dc-up dc-down dc-logs

dc-build:
	docker compose build

dc-up:
	docker compose up -d

dc-down:
	docker compose down

dc-logs:
	docker compose logs -f

