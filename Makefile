# Minima benchmark demo — common workflows.
# Requires `uv` (https://docs.astral.sh/uv). Falls back to python venv if you prefer.

PY  := .venv/bin/python
BIN := .venv/bin/minima-demo
RUN ?= $(shell ls -dt results/*/ 2>/dev/null | head -1)   # most recent run dir (for `make report`)

.PHONY: setup fetch-dataset smoke resolve bench-code bench-hard bench-catalog bench-catalog-dry bench-dataset report all clean

setup:                ## create the venv and install the demo (pinned deps incl. minima-cli[seed])
	uv venv --python 3.13 .venv
	uv pip install --python $(PY) -e .
	@echo "✓ setup complete. Copy .env.example -> .env and fill in your keys."

fetch-dataset:        ## one-time ~1.28GB download of LLMRouterBench (needed only for bench-dataset)
	$(PY) -c "import minima.seeding.llmrouterbench as lr; print('cached at', lr.download_tarball())"

smoke:                ## the gate: health + recommend/feedback round-trip against api.minima.sh
	$(BIN) smoke

resolve:              ## print the live catalog and the resolved candidate pools for both tracks
	$(BIN) resolve

bench-code:           ## HEADLINE: LIVE LiveCodeBench track — REALLY runs each model's code vs tests
	$(BIN) bench-catalog --code --workers 16

bench-hard:           ## LIVE frontier mix: MATH-500 (lvl 1–5) + LLMRouterBench + IFEval, easy→hard
	$(BIN) bench-catalog --hard --workers 16

bench-catalog:        ## LIVE track: route over all real catalog models, calling your keys (prompts to confirm spend)
	$(BIN) bench-catalog

bench-catalog-dry:    ## same pipeline with a simulated matrix — no spend, no provider calls
	$(BIN) bench-catalog --dry-run --yes

bench-dataset:        ## REPLAY track: reproducible LLMRouterBench benchmark (no model spend)
	$(BIN) bench-dataset

report:               ## re-render the dashboard for RUN (defaults to the most recent results/ dir)
	$(BIN) report "$(RUN)"

all: smoke bench-catalog bench-dataset  ## smoke + both tracks end-to-end

clean:                ## remove run outputs (keeps committed fixtures/)
	rm -rf results

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n",$$1,$$2}'
