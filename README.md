# Fractals and Their Applications

This repository contains Python code used for the computational analysis in my honours thesis on fractals and their applications to financial time series.

## Files

- `src/silver_fractal_analysis.py`  
  Performs fractal and stochastic analysis of silver price dynamics, including Hurst exponent estimation, DFA, MF-DFA, distributional testing, variance scaling, simulated comparisons, and figure/table generation.

- `src/btc_full_analysis.py`  
  Performs a full Bitcoin fractal analysis pipeline, including R/S scaling, global and rolling Hurst exponent estimation, structural breakpoint analysis, regime comparison, and figure/table generation.

## Data

The Bitcoin analysis uses `data/BTC-USD_closing.xlsx`, containing BTC-USD closing price data and log returns.
The silver analysis uses the Yahoo Finance ticker `SI=F` for silver futures price data.

## Running the scripts

```bash
python src/silver_fractal_analysis.py
python src/btc_full_analysis.py
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
