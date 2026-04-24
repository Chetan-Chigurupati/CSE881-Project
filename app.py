import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

st.set_page_config(page_title="NBA MVP Predictor", page_icon="🏀", layout="wide")

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

TRAIN_DATA_PATH = Path("cleaned_data.csv")
DEFAULT_INFERENCE_DATA_PATH = Path("api_data.csv")
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

    # some datasets use different column names for the same thing
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
def load_training_data():
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Can't find {TRAIN_DATA_PATH.resolve()}. Make sure cleaned_data.csv is in the same folder."
        )
    df = pd.read_csv(TRAIN_DATA_PATH)
    df = engineer_features(df)

    season_col = find_first_existing(df, SEASON_CANDIDATES)
    player_col = find_first_existing(df, PLAYER_CANDIDATES)
    team_col = find_first_existing(df, TEAM_CANDIDATES)

    if season_col is None:
        raise ValueError(f"No season column found. Tried: {SEASON_CANDIDATES}")
    if player_col is None:
        raise ValueError(f"No player column found. Tried: {PLAYER_CANDIDATES}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}'")

    return df, season_col, player_col, team_col


def read_inference_source(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df, uploaded_file.name
    if DEFAULT_INFERENCE_DATA_PATH.exists():
        df = pd.read_csv(DEFAULT_INFERENCE_DATA_PATH)
        return df, DEFAULT_INFERENCE_DATA_PATH.name
    return None, None


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
    # using the three that tend to complement each other well
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
        raise ValueError("Not enough seasons to make a proper train/val/test split.")
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

    for model_idx, (_, model) in enumerate(stack_base_models.items()):
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

    # tune ridge alpha via cross-val
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


def fit_stacked_inference_model(X_full, y_full, groups, passthrough_cols):
    stack_base_models = build_stack_base_models()
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

        # retrain on full data for inference
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
        "model_names": model_names,
        "passthrough_cols": passthrough_cols,
    }


@st.cache_resource(show_spinner=True)
def train_all_models():
    df, season_col, player_col, team_col = load_training_data()
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
    predictions["Stacked Ensemble"] = stacked_pred
    metrics_rows["Stacked Ensemble"] = evaluate_predictions(
        "Stacked Ensemble", y_test, stacked_pred, test_df, season_col, player_col
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
        "train_medians": train_medians,
    }


def predict_inference_data(bundle, selected_model, uploaded_file=None):
    inference_df, source_name = read_inference_source(uploaded_file)
    if inference_df is None:
        return None, "No data found. Upload a CSV to run predictions.", None

    inference_df = engineer_features(inference_df)
    infer_player_col = find_first_existing(inference_df, PLAYER_CANDIDATES)
    infer_team_col = find_first_existing(inference_df, TEAM_CANDIDATES)

    # fill in any features missing from the inference file with training medians
    missing_features = [c for c in bundle["feature_cols"] if c not in inference_df.columns]
    for col in missing_features:
        inference_df[col] = bundle["train_medians"].get(col, 0)

    X_infer = inference_df[bundle["feature_cols"]].copy().fillna(bundle["train_medians"])

    if selected_model == "Stacked Ensemble":
        train_df = bundle["df"]
        season_col = bundle["season_col"]
        passthrough_cols = [c for c in ["PPG", "APG", "RPG", "Win_Rate", "TS%", "G"] if c in bundle["feature_cols"]]

        X_full = train_df[bundle["feature_cols"]].copy().fillna(bundle["train_medians"])
        y_full = train_df[TARGET_COL].copy()
        groups = train_df[season_col].copy()

        stack_bundle = fit_stacked_inference_model(X_full, y_full, groups, passthrough_cols)

        base_pred_cols = []
        for model_name in stack_bundle["model_names"]:
            pred = stack_bundle["base_models"][model_name].predict(X_infer)
            base_pred_cols.append(pred)

        meta_input = pd.DataFrame(np.column_stack(base_pred_cols), columns=stack_bundle["model_names"])
        if stack_bundle["passthrough_cols"]:
            meta_input = pd.concat(
                [meta_input.reset_index(drop=True), X_infer[stack_bundle["passthrough_cols"]].reset_index(drop=True)],
                axis=1,
            )
        pred_infer = np.clip(stack_bundle["meta_model"].predict(meta_input), 0, None)
    else:
        model = build_models()[selected_model]
        X_full = bundle["df"][bundle["feature_cols"]].copy().fillna(bundle["train_medians"])
        y_full = bundle["df"][TARGET_COL].copy()
        model.fit(X_full, y_full)
        pred_infer = np.clip(model.predict(X_infer), 0, None)

    out = inference_df.copy()
    out["Predicted_Pts_Won"] = pred_infer
    display_cols = []
    if infer_player_col:
        display_cols.append(infer_player_col)
    if infer_team_col:
        display_cols.append(infer_team_col)
    display_cols += ["Predicted_Pts_Won"]
    for stat in ["PPG", "RPG", "APG", "Win_Rate", "TS%", "G", "Age"]:
        if stat in out.columns and stat not in display_cols:
            display_cols.append(stat)

    out = out.sort_values("Predicted_Pts_Won", ascending=False).reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)
    final_cols = ["Rank"] + display_cols
    return out[final_cols], None, source_name


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


st.title("🏀 NBA MVP Predictor")
st.caption("Train on historical MVP voting data, evaluate on held-out seasons, and score new players from a CSV.")

try:
    bundle = train_all_models()
except Exception as e:
    st.error(f"Failed to load/train: {e}")
    st.stop()

results_df = bundle["results_df"]

with st.sidebar:
    st.header("Settings")
    available_models = list(bundle["predictions"].keys())
    selected_model = st.selectbox("Model", available_models, index=len(available_models) - 1)
    selected_test_season = st.selectbox("Test season", bundle["test_seasons"], index=len(bundle["test_seasons"]) - 1)
    top_n = st.slider("Rows to show", 5, 25, 10)
    uploaded_inference_file = st.file_uploader("Upload CSV for predictions", type=["csv"])


tab1, tab2, tab3 = st.tabs(["Model Comparison", "Season Leaderboard", "New Predictions"])

with tab1:
    best_row = results_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", best_row["Model"])
    c2.metric("Best RMSE", f"{best_row['RMSE']:.2f}")
    c3.metric("Best Top-1 Acc", f"{results_df['Top1 Acc'].max():.2f}")
    c4.metric("Mean R²", f"{results_df['R²'].mean():.2f}")

    st.subheader("Test set results")
    st.dataframe(results_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#1E3A8A" if x == results_df["RMSE"].min() else "#D1D5DB" for x in results_df["RMSE"]]
    ax.bar(results_df["Model"], results_df["RMSE"], color=colors)
    ax.set_ylabel("RMSE")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    leaderboard_df, _ = get_season_leaderboard(bundle, selected_model, selected_test_season)
    st.subheader(f"{selected_test_season} — {selected_model}")
    st.dataframe(leaderboard_df.head(top_n), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Score new players")
    st.caption("Upload a CSV using the sidebar, then hit Run.")
    run_inference = st.button("Run Predictions", type="primary")

    if run_inference:
        pred_df, pred_error, source_name = predict_inference_data(bundle, selected_model, uploaded_inference_file)

        if pred_error:
            st.error(pred_error)
        else:
            st.success(f"{len(pred_df)} players scored from {source_name} ({selected_model})")
            st.dataframe(pred_df.head(top_n), use_container_width=True, hide_index=True)

            csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name="mvp_predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("Pick a model and upload a CSV, then click Run Predictions.")