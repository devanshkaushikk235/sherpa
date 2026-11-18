.PHONY: data check clean-data backtest eval figures test demo dashboard all

data:        ## download raw SteamCharts + Steam News
	python run.py ingest

check:       ## print sampling granularity per game (run this first)
	python run.py check

clean-data:  ## raw -> data/interim/{series,patches}.parquet
	python run.py clean

backtest:    ## run rolling-origin backtest across all games
	python run.py backtest

eval:        ## compute strata metrics, bootstrap CIs, and capacity sensitivity
	python run.py eval

figures:     ## render publication figures in reports/figures/
	python run.py figures

dashboard:   ## launch interactive Streamlit LiveOps dashboard
	streamlit run src/viz/dashboard.py

test:        ## run 36 automated unit & integration tests (leakage, intervals, schemas, DE)
	pytest -q

demo:        ## fast offline end-to-end on synthetic data
	python run.py demo

all: clean-data backtest eval figures test
