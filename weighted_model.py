import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.optimize import minimize
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# 1. LOAD DATA & REPRODUCE THE SAME PREPROCESSING
df = pd.read_csv('cleaned_data.csv')

# Feature engineering (same as notebook)
df['PPG'] = df['PTS']
df['RPG'] = df['TRB']
df['APG'] = df['AST']
df['SPG'] = df['STL']
df['BPG'] = df['BLK']
df['MPG'] = df['MP']
df['TPG'] = df['TOV']
df['TS%'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA'])).replace(0, 1)
df['Usage'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['MP'].replace(0, 1) * 100
df['Win_Rate'] = df['W'] / (df['W'] + df['L']).replace(0, 1)
df['Primary_Pos'] = df['Pos'].str.split('-').str[0]
pos_dummies = pd.get_dummies(df['Primary_Pos'], prefix='Pos')
df = pd.concat([df, pos_dummies], axis=1)

# Same time-based split
seasons = sorted(df['Year'].unique())
split_idx = int(len(seasons) * 0.9)
train_seasons = seasons[:split_idx]
test_seasons = seasons[split_idx:]
train_df = df[df['Year'].isin(train_seasons)]
test_df = df[df['Year'].isin(test_seasons)]

# Same features
feature_cols = [
    'W', 'Win_Rate', 'PS/G', 'PA/G',
    'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'MPG',
    'FG%', '3P%', 'eFG%', 'FT%', 'TS%',
    'Usage', 'G', 'Age'
]

X_train = train_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
y_train = train_df['Pts Won']
X_test = test_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
y_test = test_df['Pts Won']

# 2. GET OUT-OF-FOLD BASE MODEL PREDICTIONS
def get_oof_predictions(X_train, y_train, X_test, train_years, n_splits=5):
    """Get out-of-fold predictions for training data."""
    unique_years = sorted(np.unique(train_years))
    fold_size = len(unique_years) // n_splits
    
    model_configs = {
        'random_forest': RandomForestRegressor(
            n_estimators=600, max_depth=20, min_samples_split=10,
            min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'xgboost': xgb.XGBRegressor(
            n_estimators=900, max_depth=7, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
            random_state=42, n_jobs=-1
        ),
        'lightgbm': lgb.LGBMRegressor(
            n_estimators=700, max_depth=10, learning_rate=0.07,
            num_leaves=63, min_child_samples=30, subsample=0.85,
            colsample_bytree=0.85, random_state=42, verbose=-1, n_jobs=-1
        ),
        'catboost': CatBoostRegressor(
            iterations=700, depth=8, learning_rate=0.08,
            l2_leaf_reg=2.0, random_state=42, verbose=False
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=700, max_depth=7, learning_rate=0.07,
            min_samples_split=15, min_samples_leaf=6, subsample=0.85,
            max_features='sqrt', random_state=42
        )
    }
    
    n_train = len(X_train)
    n_test = len(X_test)
    n_models = len(model_configs)
    
    oof_train = np.zeros((n_train, n_models))
    oof_test = np.zeros((n_test, n_models))
    
    for m_idx, (name, base_model) in enumerate(model_configs.items()):
        print(f"  OOF for {name}...")
        test_preds_folds = []
        
        for fold in range(n_splits):
            start = fold * fold_size
            end = start + fold_size if fold < n_splits - 1 else len(unique_years)
            val_years = unique_years[start:end]
            
            val_mask = np.isin(train_years, val_years)
            train_mask = ~val_mask
            
            
            fold_model = clone(base_model) if hasattr(base_model, 'get_params') else base_model.__class__(**base_model.get_params())
            
            fold_model.fit(X_train[train_mask], y_train[train_mask])
            oof_train[val_mask, m_idx] = fold_model.predict(X_train[val_mask])
            test_preds_folds.append(fold_model.predict(X_test))
        
        # Average test predictions across folds
        oof_test[:, m_idx] = np.mean(test_preds_folds, axis=0)
    
    return oof_train, oof_test

print("Generating out-of-fold predictions...")
train_base_preds, test_base_preds = get_oof_predictions(
    X_train.values, y_train.values, X_test.values, 
    train_df['Year'].values, n_splits=5
)
# 3. ADAPTIVE WEIGHTED ENSEMBLE (custom algorithm)
class AdaptiveWeightedEnsemble:
    """
    Learns optimal base model weights using leave-one-season-out CV,
    then adaptively adjusts per-prediction based on model agreement.
    
    When models strongly disagree on a player, weights shift toward
    the models that were more reliable on similar disagreement levels
    during training. Built from scratch.
    """
    
    def __init__(self, n_base_models):
        self.n_base = n_base_models
        self.global_weights = None
        self.high_agreement_weights = None
        self.low_agreement_weights = None
        self.disagreement_threshold = None
    
    def _optimize_weights(self, base_preds, y_true, years, min_weight=0.05):
        n_models = base_preds.shape[1]
        unique_years = np.unique(years)
        
        def loso_objective(w):
            fold_errors = []
            for year in unique_years:
                val_mask = years == year
                val_pred = base_preds[val_mask] @ w
                val_true = y_true[val_mask]
                fold_errors.append(np.mean((val_pred - val_true) ** 2))
            return np.mean(fold_errors)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(min_weight, 1)] * n_models
        
        best_result = None
        best_loss = np.inf
        
        for _ in range(30):
            w0 = np.random.dirichlet(np.ones(n_models))
            w0 = np.clip(w0, min_weight, None)
            w0 /= w0.sum()
            result = minimize(loso_objective, w0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
        
        return best_result.x
    
    def fit(self, base_preds, years, y_true):
        self.global_weights = self._optimize_weights(base_preds, y_true, years)
        print(f"Global weights: {self.global_weights.round(3)}")
        
        disagreements = np.std(base_preds, axis=1)
        self.disagreement_threshold = np.median(disagreements)
        
        high_agree_mask = disagreements <= self.disagreement_threshold
        low_agree_mask = disagreements > self.disagreement_threshold
        
        print(f"Disagreement threshold: {self.disagreement_threshold:.2f}")
        print(f"High agreement samples: {high_agree_mask.sum()}, Low: {low_agree_mask.sum()}")
        
        if high_agree_mask.sum() > 500:
            self.high_agreement_weights = self._optimize_weights(
                base_preds[high_agree_mask], y_true[high_agree_mask], 
                years[high_agree_mask]
            )
            print(f"High agreement weights: {self.high_agreement_weights.round(3)}")
        else:
            self.high_agreement_weights = self.global_weights
        
        if low_agree_mask.sum() > 500:
            self.low_agreement_weights = self._optimize_weights(
                base_preds[low_agree_mask], y_true[low_agree_mask],
                years[low_agree_mask]
            )
            print(f"Low agreement weights: {self.low_agreement_weights.round(3)}")
        else:
            self.low_agreement_weights = self.global_weights
    
    def predict(self, base_preds):
        disagreements = np.std(base_preds, axis=1)
        predictions = np.zeros(len(base_preds))
        weights_used = np.zeros((len(base_preds), self.n_base))
        
        for i in range(len(base_preds)):
            if disagreements[i] <= self.disagreement_threshold:
                w = self.high_agreement_weights
            else:
                blend = min(disagreements[i] / (2 * self.disagreement_threshold), 1.0)
                w = (1 - blend) * self.low_agreement_weights + blend * self.global_weights
            
            predictions[i] = base_preds[i] @ w
            weights_used[i] = w
        
        return predictions, weights_used


# 4. TRAIN & EVALUATE
print("Training Adaptive Weighted Ensemble")
ensemble = AdaptiveWeightedEnsemble(n_base_models=5)
ensemble.fit(train_base_preds, train_df['Year'].values, y_train.values)

y_pred_ensemble, weight_matrix = ensemble.predict(test_base_preds)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
mae = mean_absolute_error(y_test, y_pred_ensemble)
r2 = r2_score(y_test, y_pred_ensemble)

print(f"\n{'Adaptive Ensemble':20s} | RMSE: {rmse:6.2f} | MAE: {mae:6.2f} | R²: {r2:.4f}")

y_pred_simple_avg = test_base_preds.mean(axis=1)
rmse_avg = np.sqrt(mean_squared_error(y_test, y_pred_simple_avg))
r2_avg = r2_score(y_test, y_pred_simple_avg)
print(f"{'Simple Average':20s} | RMSE: {rmse_avg:6.2f} | MAE: {mean_absolute_error(y_test, y_pred_simple_avg):6.2f} | R²: {r2_avg:.4f}")

print("\nBase model individual performance:")
model_names = ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost', 'Gradient Boosting']
for i, name in enumerate(model_names):
    y_pred_base = test_base_preds[:, i]
    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
    r2_base = r2_score(y_test, y_pred_base)
    print(f"  {name:20s} | RMSE: {rmse_base:6.2f} | R²: {r2_base:.4f}")

print(f"\nHigh agreement weights: {ensemble.high_agreement_weights.round(3)}")
print(f"Low agreement weights:  {ensemble.low_agreement_weights.round(3)}")

pickle.dump(ensemble, open('saved_models/gating_network.pkl', 'wb'))