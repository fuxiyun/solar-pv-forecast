.PHONY: full-run install data proxy train evaluate dashboard report clean help

# ── Configuration ───────────────────────────────────────────────
PYTHON   := python
DATA_DIR := data
OUT_DIR  := output

# ══════════════════════════════════════════════════════════════════
# Main entry point: run entire pipeline end-to-end
# ══════════════════════════════════════════════════════════════════
full-run: install data proxy train evaluate
	@echo "══════════════════════════════════════════════"
	@echo "  Pipeline complete."
	@echo "  Predictions:  $(OUT_DIR)/predictions.parquet"
	@echo "  Runtime log:  $(OUT_DIR)/runtime.log"
	@echo "══════════════════════════════════════════════"

# ── Environment setup ───────────────────────────────────────────
install:
	uv sync
	@echo "[✓] Environment ready."

# ── Phase 1: Data sourcing & engineering ────────────────────────
data: data-weather data-target data-pv data-harmonise
	@echo "[✓] Phase 1 complete: all data sourced and harmonised."

data-weather:
	$(PYTHON) -m solar_pv_forecast.data.fetch_weather

data-target:
	$(PYTHON) -m solar_pv_forecast.data.fetch_target

data-pv:
	$(PYTHON) -m solar_pv_forecast.data.fetch_pv_capacity

data-harmonise:
	$(PYTHON) -m solar_pv_forecast.data.harmonise

# ── Phase 1.5: Synthetic proxy construction ─────────────────────
proxy:
	$(PYTHON) -m solar_pv_forecast.proxy.build_proxy

# ── Phase 2: Modelling ──────────────────────────────────────────
train:
	$(PYTHON) -m solar_pv_forecast.model.train

evaluate:
	$(PYTHON) -m solar_pv_forecast.model.evaluate

# ── Phase 3: Dashboard ──────────────────────────────────────────
dashboard: install
	streamlit run src/solar_pv_forecast/dashboard.py

# ── Phase 4: Report compilation ─────────────────────────────────
report:
	cd report && pdflatex -interaction=nonstopmode technical_note.tex \
		&& pdflatex -interaction=nonstopmode technical_note.tex
	@echo "[✓] Report compiled: report/technical_note.pdf"

# ── Utilities ───────────────────────────────────────────────────
clean:
	rm -rf $(DATA_DIR)/interim/* $(DATA_DIR)/processed/* $(OUT_DIR)/*
	rm -f report/*.aux report/*.log report/*.out report/*.toc
	@echo "[✓] Cleaned intermediate and output files."

help:
	@echo "Usage:"
	@echo "  make full-run    Run entire pipeline (data → proxy → train → evaluate)"
	@echo "  make install     Set up Python environment with uv"
	@echo "  make data        Phase 1: fetch and harmonise all data"
	@echo "  make proxy       Phase 1.5: build synthetic PV proxy"
	@echo "  make train       Phase 2: train baseline + LightGBM models"
	@echo "  make evaluate    Phase 2: evaluate models, produce predictions.parquet"
	@echo "  make dashboard   Phase 3: launch interactive Streamlit dashboard"
	@echo "  make report      Phase 4: compile LaTeX report"
	@echo "  make clean       Remove intermediate files"
