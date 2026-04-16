
# NBA MVP Prediction Dashboard Prototype

This repository contains an interactive Streamlit prototype for NBA MVP vote prediction. It is designed to satisfy the prototype development requirement by providing a web-based interface that dynamically updates predictions and visualizations.

## Features

- season selector for held-out test seasons
- model selector for multiple trained regressors
- dynamic MVP leaderboard with predicted and actual rankings
- player comparison panel
- what-if simulator for adjusting player statistics and predicting MVP vote points in real time

## Why this counts as prototype development

This app is not a static notebook visualization. It is an interactive web interface where users can:
- query results by season
- switch between models
- compare players dynamically
- change input features and get new predictions instantly

That makes it a valid prototype under the project requirement.

## Files

- `app.py` — main Streamlit application
- `requirements.txt` — Python dependencies
- `README.md` — project instructions
- `cleaned_data.csv` — expected dataset file placed next to `app.py`

## How to run locally

1. Put `cleaned_data.csv` in the same folder as `app.py`
2. Install dependencies
3. Start Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected columns

The app expects:
- `Pts Won` as the target column
- one season column like `Year` or `Season`
- one player column like `Player`

It also works best if the dataset contains basketball features such as:
- `PPG`
- `APG`
- `RPG`
- `W`
- `L`
- `G`
- `TS%`
- `Age`

## Suggested GitHub repo structure

```text
.
├── app.py
├── requirements.txt
├── README.md
└── cleaned_data.csv
```

## Deployment options

You can deploy this app using:
- Streamlit Community Cloud
- Render
- Hugging Face Spaces
- local demo on your machine

## Author

Yash Bhawarkar
