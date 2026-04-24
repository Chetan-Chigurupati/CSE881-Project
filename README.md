# NBA MVP Prediction System

An end-to-end machine learning project for predicting NBA Most Valuable Player (MVP) vote totals from historical player and team statistics. Built as a class project for CSE 881: Data Mining at Michigan State University.

**Live demo:** https://nba-mvp-predictor-cse881.streamlit.app

## Overview

The project predicts each player's MVP voting points (`Pts Won`) for a given season using regression models trained on 42 seasons of NBA data (1980–2021). It includes:

- A full modeling pipeline with 9 models (dummy baseline, linear baselines, five tree-based ensembles, and a stacked ensemble meta-model)
- An adaptive weighted ensemble with disagreement-based gating as an alternative ensemble approach
- Live 2025–26 season data pulled from the `nba_api` package for out-of-sample inference
- An interactive Streamlit dashboard deployed to Streamlit Community Cloud

## Live Demo

The interactive dashboard is hosted at https://nba-mvp-predictor-cse881.streamlit.app

**First-visit note:** The app retrains all nine models on first load, which can take 5–10 minutes. Subsequent visits are cached and load in seconds. If the app appears stuck on `Running train_all_models()`, please wait — it is training, not frozen.

The dashboard includes:

- Season selector for held-out test seasons
- Model selector across all trained regressors
- MVP leaderboard with predicted and actual rankings side-by-side
- Player comparison panel
- What-if simulator for adjusting player statistics and seeing predicted vote totals update in real time

## Repository Structure

```
.
├── app.py                                         # Streamlit dashboard
├── 881_Project_Notebook.ipynb                     # Main modeling notebook (baselines)
├── 881_Project_Notebook_Stacking_Ensemble.ipynb   # Stacked ensemble pipeline
├── weighted_model.py                              # Adaptive weighted ensemble
├── nba_api.ipynb                                  # 2025–26 data collection script
├── cleaned_data.csv                               # Kaggle training dataset (1980–2021)
├── api_data.csv                                   # Current-season data pulled from nba_api
├── requirements.txt                               # Python dependencies
└── README.md
```

## How to Run Locally

1. Clone the repository
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Launch the Streamlit dashboard:
```bash
   streamlit run app.py
```
4. Or run the notebooks directly in Jupyter to reproduce the modeling results:
```bash
   jupyter notebook 881_Project_Notebook_Stacking_Ensemble.ipynb
```

## Data Sources

- **Training data:** [NBA Stats since 1980](https://www.kaggle.com/datasets/javigallego/stats-nba) on Kaggle (17,976 player-season records, 45 columns, 1980–2021)
- **Current-season data:** [`nba_api`](https://github.com/swar/nba_api) Python package for 2025–26 live inference

## Authors

Adhyan Negi, Arhan Mulay, Chetan Chigurupati, Yash Bhawarkar
Michigan State University — CSE 881, Spring 2026
