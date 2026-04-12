# CSE881-Project

# NBA MVP Prediction — Adaptive Weighted Ensemble

## Overview

This branch (`weighted-ensemble`) implements a custom ensemble algorithm that combines predictions from 5 base models by dynamically adjusting how much it trusts each model based on how much they agree or disagree on a given prediction.

## How It Works

### Base Models (from notebook)
Five tree-based regression models are trained to predict MVP vote share:
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting

### The Problem with Simple Ensembles
Averaging all 5 models equally (RMSE 40.50) is actually worse than just using Random Forest alone (RMSE 40.27). The weaker models drag down the average.

### Solution: Adaptive Weighted Ensemble
Instead of fixed equal weights, the custom algorithm learns different weights for different situations:

- **High agreement** (models predict similar values): Trust Random Forest heavily (75%). When the answer is obvious (e.g., a bench player getting 0 votes), the most stable model is the most reliable.
- **Low agreement** (models disagree): Spread weight across multiple models — RF 22%, XGBoost 30%, Gradient Boosting 34%. For contested MVP candidates, multiple perspectives help.

The algorithm measures disagreement as the standard deviation across the 5 model predictions. A threshold (median disagreement from training) separates high and low agreement cases.

### How Weights Are Learned
- **Out-of-fold predictions**: Base models predict on data they weren't trained on, preventing the optimizer from rewarding memorization.
- **Leave-one-season-out CV**: For each season in training, hold it out, predict it, measure error. Weights are optimized to minimize error across all held-out seasons.
- **Constrained optimization**: Weights must sum to 1, with a minimum floor of 5% per model so no model is completely ignored.

## Results
| Model | RMSE | R² |
|-------|------|----|
| **Adaptive Ensemble** | **40.31** | **0.4933** |
| Random Forest (best individual) | 40.27 | 0.4941 |
| Gradient Boosting | 40.28 | 0.4939 |
| Simple Average Ensemble | 40.50 | 0.4884 |
| CatBoost | 41.86 | 0.4535 |
| XGBoost | 42.25 | 0.4433 |
| LightGBM | 42.33 | 0.4411 |

The adaptive ensemble outperforms the naive ensemble and matches the best individual model. Its advantage is robustness — it doesn't require knowing in advance which base model will perform best on unseen data.

## Files

- `881_Project_Notebook.ipynb` — Data preprocessing, EDA, base model training and evaluation
- `weighted_model.py` — Adaptive weighted ensemble implementation (custom algorithm)
- `cleaned_data.csv` — Training dataset (NBA stats 1980-2021, 17,976 rows)
- `saved_models/` — Pickled base models and ensemble model

## How to Run

```bash
# 1. Run the notebook first to train and save base models
# (adds pickle files to saved_models/)

# 2. Run the ensemble
python3 weighted_model.py
```

## Requirements
- Python 3.11+
- pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, scipy
