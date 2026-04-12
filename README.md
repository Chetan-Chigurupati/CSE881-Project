# NBA MVP Vote Prediction using a Season Aware Stacked Ensemble

This project predicts NBA MVP voting points using a machine learning pipeline built on player performance, efficiency, availability, and team success features.

## Project Overview

The goal of this project is to predict `Pts Won` in NBA MVP voting and compare multiple regression based approaches. Since MVP prediction is not only a regression task but also a ranking task, the project evaluates both numeric prediction quality and season wise ranking quality.

This project includes:

- season aware train validation test split
- feature engineering for basketball specific metrics
- simple baselines such as Dummy Mean, Linear Regression, and Ridge
- advanced models including Random Forest, XGBoost, LightGBM, Gradient Boosting, and CatBoost
- a team proposed stacked ensemble using out of fold predictions
- regression metrics and ranking based metrics

## Main Notebook

- `881_Project_Notebook_Redesigned_Stacking.ipynb`

## Proposed Method

The main proposed method is a **season aware stacked ensemble**.

### Stacking design

**Base models**
- LightGBM
- Gradient Boosting
- CatBoost

**Meta learner**
- Ridge Regression

**Meta features**
- out of fold predictions from base models
- optional passthrough basketball features such as PPG, APG, RPG, Win Rate, TS%, and Games Played

### Why this method

Different models capture different nonlinear relationships in MVP voting. Instead of selecting a single model, stacking learns how to combine multiple strong regressors while avoiding leakage through out of fold training predictions.

## Evaluation Metrics

The project reports:

- RMSE
- MAE
- R²
- Top1 Accuracy by season
- Top5 Overlap by season

These metrics help evaluate both vote point prediction quality and actual MVP ranking quality.

## Dataset

The notebook expects a cleaned CSV file:

- `cleaned_data.csv`

Make sure this file is placed in the same directory as the notebook, or update the path in the configuration section.

## How to Run

1. Clone the repository
2. Install dependencies
3. Place `cleaned_data.csv` in the project directory
4. Open the notebook and run all cells

## Suggested Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn xgboost lightgbm catboost jupyter
