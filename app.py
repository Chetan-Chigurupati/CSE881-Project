import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import math

from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

st.set_page_config(page_title="NBA MVP Prediction Dashboard", page_icon="🏀", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: white;
        padding: 15px !important;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700;
        color: #1E3A8A;
        white-space: normal !important; 
        word-break: break-word;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        white-space: normal !important;
        color: #4B5563;
    }
    [data-testid="stTooltipIcon"] svg {
        fill: #1E3A8A !important;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #1E3A8A !important; 
        color: white !important; 
    }
    </style>
    """, unsafe_allow_html=True)

DATA_PATH = Path("cleaned_data.csv")
TARGET_COL = "Pts Won"
SEASON_CANDIDATES = ["Year", "Season", "season", "year"]
PLAYER_CANDIDATES = ["Player", "player", "Name", "PlayerName"]
TEAM_CANDIDATES = ["Tm", "Team", "team"]
TEST_SEASONS_N = 2
VAL_SEASONS_N = 2
RANDOM_STATE = 42
RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]

def find_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def season_top1_accuracy(eval_df, pred_col, actual_col, season_col, player_col):
    scores = []
    for _, grp in eval_df.groupby(season_col):
        actual_top = grp.sort_values(actual_col, ascending=False).iloc[0][player_col]
        pred_top = grp.sort_values(pred_col, ascending=False).iloc[0][player_col]
        scores.append(int(actual_top == pred_top))
    return float(np.mean(scores)) if scores else np.nan

def season_topk_overlap(eval_df, pred_col, actual_col, season_col, player_col, k=5):
    overlaps = []
    for _, grp in eval_df.groupby(season_col):
        actual_topk = set(grp.sort_values(actual_col, ascending=False).head(k)[player_col])
        pred_topk = set(grp.sort_values(pred_col, ascending=False).head(k)[player_col])
        overlaps.append(len(actual_topk.intersection(pred_topk)) / k)
    return float(np.mean(overlaps)) if overlaps else np.nan

def evaluate_predictions(model_name, y_true, y_pred, eval_df, season_col, player_col):
    local_df = eval_df.copy()
    local_df["Predicted_Pts_Won"] = np.clip(y_pred, 0, None)
    return {
        "Model": model_name,
        "RMSE": rmse(y_true, local_df["Predicted_Pts_Won"]),
        "MAE": float(mean_absolute_error(y_true, local_df["Predicted_Pts_Won"])),
        "R²": float(r2_score(y_true, local_df["Predicted_Pts_Won"])),
        "Top1 Acc": season_top1_accuracy(local_df, "Predicted_Pts_Won", TARGET_COL, season_col, player_col),
        "Top5 Overlap": season_topk_overlap(local_df, "Predicted_Pts_Won", TARGET_COL, season_col, player_col, k=5),
    }

def engineer_features(df):
    out = df.copy()
    alias_map = {
        "PTS": "PPG",
        "TRB": "RPG",
        "AST": "APG",
        "STL": "SPG",
        "BLK": "BPG",
        "MP": "MPG",
        "TOV": "TPG",
    }
    for src, dst in alias_map.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    if {"W", "L"}.issubset(out.columns):
        denom = (out["W"] + out["L"]).replace(0, np.nan)
        out["Win_Rate"] = (out["W"] / denom).fillna(0)

    if {"PTS", "FGA", "FTA"}.issubset(out.columns):
        denom = (2 * (out["FGA"] + 0.44 * out["FTA"])).replace(0, np.nan)
        out["TS%"] = (out["PTS"] / denom).fillna(0)

    if {"FGA", "FTA", "TOV", "MP"}.issubset(out.columns):
        denom = out["MP"].replace(0, np.nan)
        out["Usage"] = ((out["FGA"] + 0.44 * out["FTA"] + out["TOV"]) / denom).fillna(0)

    if {"G", "W", "L"}.issubset(out.columns):
        team_games = (out["W"] + out["L"]).replace(0, np.nan)
        out["Avail_Rate"] = (out["G"] / team_games).fillna(0)

    if {"PPG", "Win_Rate"}.issubset(out.columns):
        out["Scoring_Team_Impact"] = out["PPG"] * out["Win_Rate"]

    if {"TS%", "Avail_Rate"}.issubset(out.columns):
        out["Efficiency_Availability"] = out["TS%"] * out["Avail_Rate"]

    return out

@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH.resolve()}. Put cleaned_data.csv in the same folder as app.py.")
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    season_col = find_first_existing(df, SEASON_CANDIDATES)
    player_col = find_first_existing(df, PLAYER_CANDIDATES)
    team_col = find_first_existing(df, TEAM_CANDIDATES)

    if season_col is None:
        raise ValueError(f"Could not find a season column. Looked for: {SEASON_CANDIDATES}")
    if player_col is None:
        raise ValueError(f"Could not find a player column. Looked for: {PLAYER_CANDIDATES}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Could not find target column '{TARGET_COL}'")

    return df, season_col, player_col, team_col

def get_feature_cols(df, season_col, player_col, team_col):
    candidate_features = [
        "W", "L", "Win_Rate", "PS/G", "PA/G",
        "PPG", "RPG", "APG", "SPG", "BPG", "MPG", "TPG",
        "FG%", "3P%", "2P%", "eFG%", "FT%", "TS%",
        "Usage", "G", "Age", "Avail_Rate",
        "Scoring_Team_Impact", "Efficiency_Availability",
    ]
    excluded = {TARGET_COL, season_col, player_col}
    if team_col:
        excluded.add(team_col)
    return [c for c in candidate_features if c in df.columns and c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

def build_models():
    return {
        "Dummy Mean": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=800, max_depth=16, min_samples_split=5, min_samples_leaf=2,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.85,
            colsample_bytree=0.85, reg_lambda=1.0, random_state=RANDOM_STATE,
            n_jobs=-1, objective="reg:squarederror",
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=-1, num_leaves=31,
            subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE, verbose=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.03, loss_function="RMSE",
            random_seed=RANDOM_STATE, verbose=False,
        ),
    }

def build_stack_base_models():
    return {
        "LightGBM": LGBMRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=-1, num_leaves=31,
            subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE, verbose=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=RANDOM_STATE,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.03, loss_function="RMSE",
            random_seed=RANDOM_STATE, verbose=False,
        ),
    }

def split_by_recent_seasons(df, season_col):
    all_seasons = sorted(df[season_col].dropna().unique())
    if len(all_seasons) < (VAL_SEASONS_N + TEST_SEASONS_N + 1):
        raise ValueError("Not enough seasons for train/validation/test split.")
    test_seasons = all_seasons[-TEST_SEASONS_N:]
    val_seasons = all_seasons[-(TEST_SEASONS_N + VAL_SEASONS_N):-TEST_SEASONS_N]
    train_seasons = all_seasons[:-(TEST_SEASONS_N + VAL_SEASONS_N)]
    train_df = df[df[season_col].isin(train_seasons)].copy()
    val_df = df[df[season_col].isin(val_seasons)].copy()
    test_df = df[df[season_col].isin(test_seasons)].copy()
    return train_df, val_df, test_df, train_seasons, val_seasons, test_seasons

def build_stacked_predictions(X_trainval, y_trainval, X_test, trainval_df, feature_cols, season_col):
    stack_base_models = build_stack_base_models()
    passthrough_cols = [c for c in ["PPG", "APG", "RPG", "Win_Rate", "TS%", "G"] if c in feature_cols]
    groups = trainval_df[season_col].copy()
    n_splits = min(5, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)

    oof_preds = np.zeros((len(X_trainval), len(stack_base_models)))
    test_preds = np.zeros((len(X_test), len(stack_base_models)))
    model_names = list(stack_base_models.keys())

    for model_idx, (model_name, model) in enumerate(stack_base_models.items()):
        fold_test_preds = np.zeros((len(X_test), n_splits))
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_trainval, y_trainval, groups=groups)):
            X_tr = X_trainval.iloc[tr_idx]
            X_va = X_trainval.iloc[va_idx]
            y_tr = y_trainval.iloc[tr_idx]

            fitted = clone(model)
            fitted.fit(X_tr, y_tr)

            oof_preds[va_idx, model_idx] = fitted.predict(X_va)
            fold_test_preds[:, fold_idx] = fitted.predict(X_test)

        test_preds[:, model_idx] = fold_test_preds.mean(axis=1)

    oof_df = pd.DataFrame(oof_preds, columns=model_names, index=X_trainval.index)
    test_df_meta = pd.DataFrame(test_preds, columns=model_names, index=X_test.index)

    if passthrough_cols:
        X_meta_train = pd.concat([oof_df.reset_index(drop=True), X_trainval[passthrough_cols].reset_index(drop=True)], axis=1)
        X_meta_test = pd.concat([test_df_meta.reset_index(drop=True), X_test[passthrough_cols].reset_index(drop=True)], axis=1)
    else:
        X_meta_train = oof_df.copy()
        X_meta_test = test_df_meta.copy()

    alpha_rows = []
    for alpha in RIDGE_ALPHA_GRID:
        fold_scores = []
        for tr_idx, va_idx in gkf.split(X_meta_train, y_trainval, groups=groups):
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_meta_train.iloc[tr_idx], y_trainval.iloc[tr_idx])
            pred_va = np.clip(ridge.predict(X_meta_train.iloc[va_idx]), 0, None)
            fold_scores.append(rmse(y_trainval.iloc[va_idx], pred_va))
        alpha_rows.append((alpha, np.mean(fold_scores)))

    best_alpha = sorted(alpha_rows, key=lambda x: x[1])[0][0]
    meta_model = Ridge(alpha=best_alpha)
    meta_model.fit(X_meta_train, y_trainval)
    final_pred = np.clip(meta_model.predict(X_meta_test), 0, None)
    return final_pred

@st.cache_resource(show_spinner=True)
def train_all_models():
    df, season_col, player_col, team_col = load_data()
    feature_cols = get_feature_cols(df, season_col, player_col, team_col)
    train_df, val_df, test_df, train_seasons, val_seasons, test_seasons = split_by_recent_seasons(df, season_col)

    trainval_df = pd.concat([train_df, val_df], axis=0)
    train_medians = train_df[feature_cols].median(numeric_only=True)

    def prepare_xy(split_df):
        X = split_df[feature_cols].copy().fillna(train_medians)
        y = split_df[TARGET_COL].copy()
        return X, y

    X_trainval, y_trainval = prepare_xy(trainval_df)
    X_test, y_test = prepare_xy(test_df)

    models = build_models()
    predictions = {}
    metrics_rows = {}

    for name, model in models.items():
        model.fit(X_trainval, y_trainval)
        pred_test = np.clip(model.predict(X_test), 0, None)
        predictions[name] = pred_test
        metrics_rows[name] = evaluate_predictions(name, y_test, pred_test, test_df, season_col, player_col)

    stacked_pred = build_stacked_predictions(X_trainval, y_trainval, X_test, trainval_df, feature_cols, season_col)
    predictions["Proposed Stacked Ensemble"] = stacked_pred
    metrics_rows["Proposed Stacked Ensemble"] = evaluate_predictions(
        "Proposed Stacked Ensemble", y_test, stacked_pred, test_df, season_col, player_col
    )

    results_df = pd.DataFrame(list(metrics_rows.values())).sort_values(["RMSE", "MAE"]).reset_index(drop=True)

    return {
        "df": df,
        "season_col": season_col,
        "player_col": player_col,
        "team_col": team_col,
        "feature_cols": feature_cols,
        "test_df": test_df,
        "predictions": predictions,
        "results_df": results_df,
        "test_seasons": test_seasons,
    }

def get_season_leaderboard(bundle, model_name, selected_season):
    test_df = bundle["test_df"].copy()
    preds = bundle["predictions"][model_name]
    test_df["Predicted_Pts_Won"] = np.clip(preds, 0, None)

    season_col = bundle["season_col"]
    player_col = bundle["player_col"]
    team_col = bundle["team_col"]

    season_df = test_df[test_df[season_col] == selected_season].copy()
    season_df = season_df.sort_values("Predicted_Pts_Won", ascending=False).reset_index(drop=True)
    season_df["Predicted Rank"] = np.arange(1, len(season_df) + 1)
    season_df["Actual Rank"] = season_df[TARGET_COL].rank(method="first", ascending=False).astype(int)

    cols = [player_col]
    if team_col and team_col in season_df.columns:
        cols.append(team_col)
    cols += ["Predicted Rank", "Actual Rank", "Predicted_Pts_Won", TARGET_COL]
    for stat in ["PPG", "APG", "RPG", "Win_Rate", "TS%", "G"]:
        if stat in season_df.columns:
            cols.append(stat)
    return season_df[cols].copy(), season_df

@st.cache_resource(show_spinner=True)
def fit_stacked_simulator(df, season_col, feature_cols):
    train_medians = df[feature_cols].median(numeric_only=True)
    X_full = df[feature_cols].copy().fillna(train_medians)
    y_full = df[TARGET_COL].copy()
    groups = df[season_col].copy()

    stack_base_models = build_stack_base_models()
    passthrough_cols = [c for c in ["PPG", "APG", "RPG", "Win_Rate", "TS%", "G"] if c in feature_cols]
    n_splits = min(5, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)

    oof_preds = np.zeros((len(X_full), len(stack_base_models)))
    model_names = list(stack_base_models.keys())
    fitted_base_models = {}

    for model_idx, (model_name, model) in enumerate(stack_base_models.items()):
        for tr_idx, va_idx in gkf.split(X_full, y_full, groups=groups):
            fitted = clone(model)
            fitted.fit(X_full.iloc[tr_idx], y_full.iloc[tr_idx])
            oof_preds[va_idx, model_idx] = fitted.predict(X_full.iloc[va_idx])
        final_fitted = clone(model)
        final_fitted.fit(X_full, y_full)
        fitted_base_models[model_name] = final_fitted

    oof_df = pd.DataFrame(oof_preds, columns=model_names, index=X_full.index)
    if passthrough_cols:
        X_meta_train = pd.concat([oof_df.reset_index(drop=True), X_full[passthrough_cols].reset_index(drop=True)], axis=1)
    else:
        X_meta_train = oof_df.copy()

    alpha_rows = []
    for alpha in RIDGE_ALPHA_GRID:
        fold_scores = []
        for tr_idx, va_idx in gkf.split(X_meta_train, y_full, groups=groups):
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_meta_train.iloc[tr_idx], y_full.iloc[tr_idx])
            pred_va = np.clip(ridge.predict(X_meta_train.iloc[va_idx]), 0, None)
            fold_scores.append(rmse(y_full.iloc[va_idx], pred_va))
        alpha_rows.append((alpha, np.mean(fold_scores)))

    best_alpha = sorted(alpha_rows, key=lambda x: x[1])[0][0]
    meta_model = Ridge(alpha=best_alpha)
    meta_model.fit(X_meta_train, y_full)

    return {
        "base_models": fitted_base_models,
        "meta_model": meta_model,
        "passthrough_cols": passthrough_cols,
        "model_names": model_names,
        "train_medians": train_medians,
    }

def fit_model_for_simulator(bundle, model_name):
    df = bundle["df"]
    feature_cols = bundle["feature_cols"]
    train_medians = df[feature_cols].median(numeric_only=True)
    X = df[feature_cols].copy().fillna(train_medians)
    y = df[TARGET_COL].copy()
    model = build_models()[model_name]
    model.fit(X, y)
    return model, train_medians

def build_simulator_input(bundle, base_player_row, overrides):
    feature_cols = bundle["feature_cols"]
    train_medians = bundle["df"][feature_cols].median(numeric_only=True)

    if base_player_row is not None:
        row = base_player_row.copy()
    else:
        row = pd.Series({c: train_medians.get(c, 0) for c in feature_cols})

    for key, value in overrides.items():
        row[key] = value

    if "W" in row.index and "L" in row.index and "Win_Rate" in feature_cols:
        denom = row["W"] + row["L"]
        row["Win_Rate"] = (row["W"] / denom) if denom else 0
    if "PPG" in row.index and "Win_Rate" in row.index and "Scoring_Team_Impact" in feature_cols:
        row["Scoring_Team_Impact"] = row["PPG"] * row["Win_Rate"]
    if "TS%" in row.index and "Avail_Rate" in row.index and "Efficiency_Availability" in feature_cols:
        row["Efficiency_Availability"] = row["TS%"] * row["Avail_Rate"]

    frame = pd.DataFrame([{c: row.get(c, train_medians.get(c, 0)) for c in feature_cols}])
    return frame.fillna(train_medians)

def predict_with_stacked_simulator(bundle, sample_df):
    stack_bundle = fit_stacked_simulator(bundle["df"], bundle["season_col"], bundle["feature_cols"])
    base_pred_cols = []
    for model_name in stack_bundle["model_names"]:
        pred = stack_bundle["base_models"][model_name].predict(sample_df)[0]
        base_pred_cols.append(pred)
    meta_row = pd.DataFrame([base_pred_cols], columns=stack_bundle["model_names"])
    if stack_bundle["passthrough_cols"]:
        meta_row = pd.concat([meta_row.reset_index(drop=True), sample_df[stack_bundle["passthrough_cols"]].reset_index(drop=True)], axis=1)
    pred = float(np.clip(stack_bundle["meta_model"].predict(meta_row)[0], 0, None))
    return pred

def draw_radar_chart(player_stats, labels):
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for name, stats in player_stats.items():
        vals = stats + stats[:1]
        ax.plot(angles, vals, linewidth=2, label=name)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    return fig

st.title("🏀 NBA MVP Prediction Dashboard")
st.caption("Interactive MVP prediction app with leaderboard, player comparison, and what-if analysis.")

try:
    bundle = train_all_models()
except Exception as e:
    st.error(f"Setup error: {e}")
    st.stop()

df = bundle["df"]
season_col = bundle["season_col"]
player_col = bundle["player_col"]
team_col = bundle["team_col"]
results_df = bundle["results_df"]

with st.sidebar:
    st.header("Controls")
    available_models = list(bundle["predictions"].keys())
    selected_model = st.selectbox("Choose model", available_models, index=len(available_models)-1)
    selected_test_season = st.selectbox("Choose test season", bundle["test_seasons"], index=len(bundle["test_seasons"]) - 1)
    top_n = st.slider("Top N leaderboard", 5, 20, 10)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Leaderboard", "Head-to-Head", "Simulator"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    best_row = results_df.iloc[0]
    
    c1.metric(
        label="Top Model", 
        value=best_row["Model"], 
        help="The algorithm that achieved the lowest error during validation."
    )
    c2.metric(
        label="Min RMSE", 
        value=f"{best_row['RMSE']:.2f}", 
        help="Root Mean Squared Error: The average 'miss' in points. Lower is better."
    )
    c3.metric(
        label="Top1 Acc", 
        value=f"{results_df['Top1 Acc'].max():.2f}", 
        help="How often the model correctly predicts the actual MVP winner."
    )
    c4.metric(
        label="Avg R²", 
        value=f"{results_df['R²'].mean():.2f}", 
        help="How much of the voting data variance the AI successfully explains (Goal: 1.0)."
    )

    st.subheader("Model Error Comparison (RMSE)")
    st.dataframe(results_df, use_container_width=True)

    fig = plt.figure(figsize=(10, 4))
    colors = ['#1E3A8A' if x == results_df['RMSE'].min() else '#D1D5DB' for x in results_df['RMSE']]
    plt.bar(results_df["Model"], results_df["RMSE"], color=colors)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

with tab2:
    leaderboard_df, full_season_df = get_season_leaderboard(bundle, selected_model, selected_test_season)
    st.subheader(f"MVP Forecast — {selected_test_season}")

    st.dataframe(
        leaderboard_df.head(top_n),
        column_config={
            "Predicted_Pts_Won": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=float(leaderboard_df["Predicted_Pts_Won"].max()), format="%.1f"
            ),
            "Win_Rate": st.column_config.NumberColumn("Win %", format="%.3f"),
        },
        use_container_width=True, hide_index=True
    )

with tab3:
    st.subheader("Skill Comparison (Normalized)")
    p_names = sorted(df[df[season_col] == selected_test_season][player_col].unique())
    p1 = st.selectbox("Player A", p_names, index=0)
    p2 = st.selectbox("Player B", p_names, index=min(1, len(p_names)-1))

    stats_list = ["PPG", "RPG", "APG", "Win_Rate", "TS%"]
    p1_vals = [df[df[player_col]==p1][s].mean() / df[s].max() for s in stats_list]
    p2_vals = [df[df[player_col]==p2][s].mean() / df[s].max() for s in stats_list]
    
    st.pyplot(draw_radar_chart({p1: p1_vals, p2: p2_vals}, stats_list))

with tab4:
    st.subheader("What-if MVP Predictor")
    with st.expander("Step 1: Choose Baseline"):
        base_choice = st.selectbox("Existing profile", ["Average"] + sorted(df[player_col].unique()))
        base_row = None if base_choice == "Average" else df[df[player_col] == base_choice].iloc[0]

    def default_for(col_name, fallback):
        if base_row is not None and col_name in base_row.index: return float(base_row[col_name])
        return fallback

    c1, c2, c3 = st.columns(3)
    overrides = {}
    with c1:
        overrides["PPG"] = st.number_input("PPG", 0.0, 50.0, default_for("PPG", 20.0))
        overrides["APG"] = st.number_input("APG", 0.0, 20.0, default_for("APG", 5.0))
    with c2:
        overrides["Win_Rate"] = st.slider("Win Rate", 0.0, 1.0, default_for("Win_Rate", 0.5))
        overrides["TS%"] = st.slider("TS%", 0.3, 0.8, default_for("TS%", 0.55))
    with c3:
        overrides["G"] = st.slider("Games", 0, 82, int(default_for("G", 70)))

    if st.button("Predict Score"):
        sample = build_simulator_input(bundle, base_row, overrides)
        if selected_model == "Proposed Stacked Ensemble":
            pred = predict_with_stacked_simulator(bundle, sample)
        else:
            sim_model, _ = fit_model_for_simulator(bundle, selected_model)
            pred = float(np.clip(sim_model.predict(sample)[0], 0, None))
        st.success(f"Predicted Points: {pred:.2f}")
        if pred > 300:
            st.info("This profile looks like a very strong MVP-level season.")
        elif pred > 100:
            st.info("This profile looks like a serious MVP candidate.")
        elif pred > 0:
            st.info("This profile may receive some MVP votes.")
        else:
            st.info("This profile is unlikely to receive MVP votes based on the model.")