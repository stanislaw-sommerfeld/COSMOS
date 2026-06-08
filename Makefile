.PHONY: all fast smoke test install clean viewer

# Run the entire framework (mission + demos + figures + results + tests)
all: install
	python run_all.py

# Same, but skip the slow multi-seed validation (~1-2 min)
fast: install
	python run_all.py --fast

# Quick harness check (a few steps only)
smoke: install
	python run_all.py --smoke

# Just the test suite (locks the headline invariants)
test: install
	pytest -q

install:
	pip install -e .

# Interactive real-time viewer (needs the .[viz] extra and a display)
viewer:
	python -m cosmos.pygame_orbital_sim

clean:
	rm -f figures/cosmos_mission_*.png figures/cosmos_run.png
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.egg-info" -type d -prune -exec rm -rf {} +
