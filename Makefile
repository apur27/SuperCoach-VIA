PYTHON := /home/abhi/sourceCode/python/coding/.venv/bin/python

.PHONY: refresh predict backtest rank docs pipeline test lint fmt

refresh:
	$(PYTHON) refresh_data.py

predict:
	$(PYTHON) prediction.py

backtest:
	$(PYTHON) backtest.py --start-year 2026 --start-round 1 --end-year 2026 --end-round auto

rank:
	$(PYTHON) top_players_comprehensive.py

docs:
	$(PYTHON) refresh_readme.py

pipeline:
	bash refresh_and_rank.sh

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check .

fmt:
	$(PYTHON) -m ruff format .
