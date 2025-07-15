import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor
import optuna
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings(
    "ignore",
    message="The default of observed=False is deprecated",
    category=FutureWarning,
    module="pandas.core.groupby"
)
warnings.filterwarnings(
    'ignore',
    message='.*does not have valid feature names.*',
    category=UserWarning
)

# Data types for DataFrame columns
DTYPES = {
    'player': 'category',
    'year': 'Int16',
    'round': 'category',
    'date': 'datetime64[ns]',
    'team': 'category',
    'opponent': 'category',
    'venue': 'category',
    'disposals': 'Int16',
    'kicks': 'Int16',
    'handballs': 'Int16',
    'goals': 'Int16',
    'behinds': 'Int16',
    'hitouts': 'Int16',
    'tackles': 'Int16',
    'rebound_50s': 'Int16',
    'inside_50s': 'Int16',
    'clearances': 'Int16',
    'clangers': 'Int16',
    'frees_for': 'Int16',
    'frees_against': 'Int16',
    'brownlow_votes': 'Int8',
    'contested_possessions': 'Int16',
    'uncontested_possessions': 'Int16',
    'contested_marks': 'Int16',
    'marks_inside_50': 'Int16',
    'one_percenters': 'Int16',
    'bounces': 'Int16',
    'goal_assists': 'Int16',
    'percentage_time_played': 'float32',
    'cba_percent': 'float32',
    'dob': 'datetime64[ns]'
}

NA_VALUES = ['NA', 'N/A', '', 'nan']

# Column renaming for consistency
RENAMES = {
    'hit_outs': 'hitouts',
    'free_kicks_for': 'frees_for',
    'free_kicks_against': 'frees_against',
}

# Required columns for the model
REQUIRED_COLUMNS = ['player', 'year', 'round', 'disposals', 'kicks', 'handballs', 'tackles', 'clearances', 'inside_50s']

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard names and validate required columns."""
    df = df.rename(columns=RENAMES)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def extract_dob_and_name(filepath):
    """Extract player name and DOB from filename (e.g., 'player_name_DDMMYYYY_performance_details.csv')."""
    parts = filepath.stem.split('_')
    if len(parts) < 3 or parts[-2] != 'performance' or parts[-1] != 'details':
        print(f"⚠️ Invalid filename format: {filepath.name}")
        return filepath.stem, pd.NaT
    try:
        dob_str = parts[-3]  # e.g., '09111990'
        player_name = ' '.join(parts[:-3]).title()  # e.g., 'Marcus Bontempelli'
        dob = pd.to_datetime(dob_str, format='%d%m%Y', errors='coerce')
        if pd.isna(dob):
            print(f"⚠️ Invalid DOB format in {filepath.name}: {dob_str}")
        return player_name, dob
    except Exception as e:
        print(f"❌ Error parsing filename {filepath.name}: {e}")
        return filepath.stem, pd.NaT

def extract_round_number(round_str):
    """Extract integer round number from round string (e.g., 'Round 1' -> 1)."""
    if pd.isna(round_str):
        return np.nan
    round_str = str(round_str).strip()
    if round_str.isdigit():
        return int(round_str)
    if round_str.startswith('Round '):
        try:
            return int(round_str.split()[1])
        except (IndexError, ValueError):
            return np.nan
    return np.nan

class AFLDisposalPredictor:
    def __init__(self, data_dir: str, target_year: int = 2025, rolling_window: int = 5, within_season_window: int = 3, debug_mode: bool = False):
        """Initialize predictor with data directory, target year, rolling window sizes, and debug mode."""
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {data_dir}")
        
        self.target_year = target_year
        self.rolling_window = rolling_window
        self.within_season_window = within_season_window
        self.debug_mode = debug_mode
        self.models = {}
        self.base_rolling_features = [
            'disposals', 'kicks', 'handballs', 'tackles', 'clearances', 'inside_50s', 'rebound_50s', 
            'contested_possessions', 'uncontested_possessions', 'contested_marks', 'marks_inside_50',
            'clangers', 'frees_for', 'frees_against', 'one_percenters', 'bounces', 'goal_assists'
        ]
        self.extra_features = ['cba_percent', 'percentage_time_played']
        self.feature_columns = []
        self.best_name = None
        self.training_feature_columns = None
        self.preprocessor = None  # Store preprocessor for prediction

    def load_player(self, filepath: Path) -> pd.DataFrame:
        """Load and preprocess player data from CSV."""
        try:
            df = pd.read_csv(filepath, na_values=NA_VALUES, dtype=str)
        except Exception as e:
            print(f"❌ Failed to read CSV {filepath.name}: {e}")
            return pd.DataFrame()
        
        try:
            player_name, dob = extract_dob_and_name(filepath)
            df['player'] = player_name
            df['dob'] = dob
            df = clean_columns(df)
        except Exception as e:
            print(f"❌ Failed to process player data for {filepath.name}: {e}")
            return pd.DataFrame()
        
        if 'year' not in df.columns:
            print(f"⚠️ 'year' column missing in {filepath.name}")
            return pd.DataFrame()
        
        try:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int16')
            df = df.dropna(subset=['year'])
            for col, dtype in DTYPES.items():
                if col in df.columns and col != 'year':
                    if dtype in ['Int8', 'Int16', 'Int32', 'Int64']:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
                    elif dtype == 'category':
                        df[col] = df[col].astype('category')
                    elif dtype == 'datetime64[ns]':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        df = df.dropna(subset=[col])  # Drop rows with invalid dates
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
        except Exception as e:
            print(f"❌ Error in data type conversion for {filepath.name}: {e}")
            return pd.DataFrame()
        
        dummy_cols = [col for col in ['venue', 'opponent'] if col in df.columns]
        if dummy_cols:
            df = pd.get_dummies(df, columns=dummy_cols, drop_first=True)
        return df

    def validate_and_clean_data(self, df: pd.DataFrame, target_year: int) -> pd.DataFrame:
        """Ensure data has required columns, keeping null 'disposals' for target year."""
        required_columns = {'year', 'round', 'date'}
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
            return pd.DataFrame()
        original_rows = len(df)
        # Keep rows where 'disposals' is null only for target_year
        df = df[(df['year'] >= target_year) | df['disposals'].notnull()].copy()
        cleaned_rows = len(df)
        if original_rows > cleaned_rows:
            print(f"⚠️ Dropped {original_rows - cleaned_rows} rows due to missing 'disposals' in historical data")
        return df

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and combine data from all player CSV files."""
        all_dfs = []
        birth_year_threshold = self.target_year - 40
        for filepath in self.data_dir.glob('*.csv'):
            if "_performance_details" not in filepath.name:
                continue
            try:
                player_name, dob = extract_dob_and_name(filepath)
                if pd.isna(dob) or dob.year <= birth_year_threshold:
                    continue
                df = self.load_player(filepath)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                print(f"❌ Error processing file {filepath.name}: {e}")
                continue
        if not all_dfs:
            print("⚠️ No valid data loaded")
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)

    def _add_rolling(self, df, col, window, group_by):
        """Calculate rolling averages with specified window and grouping."""
        try:
            return df.groupby(group_by, observed=False)[col].transform(
                lambda s: s.rolling(window=window, min_periods=1).mean().shift(1)
            )
        except Exception as e:
            print(f"❌ Rolling average failed for {col} with group_by {group_by}: {e}")
            return pd.Series(np.nan, index=df.index)

    def _add_expanding_mean(self, df, col, group_by):
        """Calculate season-to-date expanding mean with specified grouping."""
        try:
            return df.groupby(group_by, observed=False)[col].transform(
                lambda s: s.expanding().mean().shift(1)
            )
        except Exception as e:
            print(f"❌ Expanding mean failed for {col} with group_by {group_by}: {e}")
            return pd.Series(np.nan, index=df.index)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for training, including rolling averages and more."""
        df.loc[:, 'round_number'] = df['round'].apply(extract_round_number)
        df = df.sort_values(['player', 'year', 'round_number'])
        
        df['age'] = (df['year'].astype(float) - df['dob'].dt.year.astype(float)).astype('float32')
        df['debut_year'] = df.groupby('player')['year'].transform('min')
        df['experience'] = (df['year'] - df['debut_year']).astype('float32')
        
        rolling_cols_available = [col for col in self.base_rolling_features if col in df.columns]
        
        for col in rolling_cols_available:
            df.loc[:, f'across_season_rolling_avg_{col}_{self.rolling_window}'] = self._add_rolling(df, col, window=self.rolling_window, group_by=['player'])
            df.loc[:, f'within_season_rolling_avg_{col}_{self.within_season_window}'] = self._add_rolling(df, col, window=self.within_season_window, group_by=['player', 'year'])
            df.loc[:, f'season_to_date_mean_{col}'] = self._add_expanding_mean(df, col, group_by=['player', 'year'])
            df.loc[:, f'recent_form_{col}'] = df.groupby(['player'], observed=False)[col].transform(
                lambda x: x.ewm(span=3, adjust=False).mean().shift(1)
            )
        
        df['days_since_last_game'] = df.groupby('player', observed=False)['date'].diff().dt.days.fillna(0)
        
        across_season_features = [f'across_season_rolling_avg_{col}_{self.rolling_window}' for col in rolling_cols_available]
        within_season_features = [f'within_season_rolling_avg_{col}_{self.within_season_window}' for col in rolling_cols_available]
        season_to_date_features = [f'season_to_date_mean_{col}' for col in rolling_cols_available]
        recent_form_features = [f'recent_form_{col}' for col in rolling_cols_available]
        extra_feats = [feat for feat in self.extra_features if feat in df.columns]
        dummy_cols = [c for c in df.columns if c.startswith(('venue_', 'opponent_'))]
        self.feature_columns = across_season_features + within_season_features + season_to_date_features + recent_form_features + ['round_number', 'days_since_last_game', 'age', 'experience'] + extra_feats + dummy_cols
        
        if self.debug_mode:
            print(f"Engineered features: {self.feature_columns}")
            for col in self.feature_columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    print(f"⚠️ {nan_count} NaN values in feature '{col}'")
        
        return df.dropna(subset=across_season_features + within_season_features + season_to_date_features + recent_form_features + ['disposals'])

    def _engineer_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for prediction, matching training features."""
        df.loc[:, 'round_number'] = df['round'].apply(extract_round_number)
        df = df.sort_values(['player', 'year', 'round_number'])
        
        df['age'] = (df['year'].astype(float) - df['dob'].dt.year.astype(float)).astype('float32')
        df['debut_year'] = df.groupby('player')['year'].transform('min')
        df['experience'] = (df['year'] - df['debut_year']).astype('float32')
        
        rolling_cols_available = [col for col in self.base_rolling_features if col in df.columns]
        
        for col in rolling_cols_available:
            df.loc[:, f'across_season_rolling_avg_{col}_{self.rolling_window}'] = self._add_rolling(df, col, window=self.rolling_window, group_by=['player'])
            df.loc[:, f'within_season_rolling_avg_{col}_{self.within_season_window}'] = self._add_rolling(df, col, window=self.within_season_window, group_by=['player', 'year'])
            df.loc[:, f'season_to_date_mean_{col}'] = self._add_expanding_mean(df, col, group_by=['player', 'year'])
            df.loc[:, f'recent_form_{col}'] = df.groupby(['player'], observed=False)[col].transform(
                lambda x: x.ewm(span=3, adjust=False).mean().shift(1)
            )
        
        df['days_since_last_game'] = df.groupby('player', observed=False)['date'].diff().dt.days.fillna(0)
        
        return df

    def prepare_features_and_target(self, df: pd.DataFrame) -> tuple:
        """Prepare features (X) and target (y) for training."""
        global df_global
        df_global = df
        historical_data = df[df['year'] < self.target_year].copy()
        historical_data = historical_data[historical_data['disposals'].notnull()]  # Ensure 'disposals' not null for training
        if historical_data.empty:
            print("⚠️ No historical data available")
            return None, None
        engineered_df = self._engineer_features(historical_data)
        X = engineered_df[self.feature_columns]
        y = np.log1p(engineered_df['disposals'])  # Transform target
        self.training_feature_columns = self.feature_columns.copy()
        
        X.loc[:, 'missing_count'] = X.isna().sum(axis=1)
        self.training_feature_columns.append('missing_count')
        
        numerical_cols = X.select_dtypes(include=['float64', 'int64', 'float32', 'Int16']).columns
        categorical_cols = X.select_dtypes(include=['category']).columns
        remainder_cols = [col for col in X.columns if col not in numerical_cols and col not in categorical_cols]
        
        if self.debug_mode:
            print(f"Numerical columns: {numerical_cols}")
            print(f"Categorical columns: {categorical_cols}")
            print(f"Remainder columns: {remainder_cols}")
            print(f"Expected columns: {self.training_feature_columns}")
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), numerical_cols),
                ('cat', SimpleImputer(strategy='constant', fill_value='missing'), categorical_cols)
            ],
            remainder='passthrough'
        )
        X_transformed = preprocessor.fit_transform(X)
        
        transformed_columns = list(numerical_cols) + list(categorical_cols) + remainder_cols
        if len(transformed_columns) != X_transformed.shape[1]:
            raise ValueError(f"Column mismatch: expected {X_transformed.shape[1]} columns, got {len(transformed_columns)}")
        
        X = pd.DataFrame(X_transformed, columns=transformed_columns, index=X.index)
        X = X.astype(float)
        X = X.reindex(columns=self.training_feature_columns, fill_value=0)
        
        self.preprocessor = preprocessor
        
        if self.debug_mode:
            print(f"Transformed X shape: {X.shape}")
            print(f"Transformed columns: {X.columns.tolist()}")
        
        return X, y

    def tune_model(self, X, y):
        """Hyperparameter tuning for HistGradientBoostingRegressor using Optuna."""
        groups = df_global.loc[X.index, 'player']
        cv = GroupKFold(n_splits=5)
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 200),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100),
                'loss': trial.suggest_categorical('loss', ['poisson', 'quantile', 'squared_error'])
            }

            if params['loss'] == 'quantile':
                params['quantile'] = trial.suggest_float('quantile', 0.1, 0.9)

            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', HistGradientBoostingRegressor(**params, random_state=42))
            ])
            score = cross_val_score(model, X, y, cv=cv, groups=groups, scoring='neg_mean_squared_error', n_jobs=-1).mean()
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=150)
        best_params = study.best_params
        self.models['hgb_tuned'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', HistGradientBoostingRegressor(**best_params, random_state=42))
        ])
        print(f"Best HGB parameters: {best_params}")
        return study.best_value

    def tune_lgbm_cpu(self, X, y):
        """Hyperparameter tuning for LGBMRegressor on CPU using Optuna."""
        print("Tuning LGBM on CPU with Optuna...")

        groups = df_global.loc[X.index, 'player']
        cv = GroupKFold(n_splits=5)

        def objective(trial):
            params = {
                'device': 'cpu',
                'objective': 'poisson',
                'metric': 'rmse',
                'random_state': 42,
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
                'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255, 511, 1023]),
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                'verbose': -1
            }

            model = LGBMRegressor(**params)
            
            score = cross_val_score(
                model, X, y, cv=cv, groups=groups, scoring='neg_mean_squared_error', n_jobs=-1
            ).mean()
            
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=150, timeout=1200)  # 20-minute timeout
        
        print(f"Best LGBM CPU parameters: {study.best_params}")
        
        self.models['lgbm_cpu_tuned'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LGBMRegressor(**study.best_params, device='cpu', random_state=42))
        ])
        
        return study.best_value

    def tune_rf(self, X, y):
        """Hyperparameter tuning for RandomForestRegressor using Optuna."""
        groups = df_global.loc[X.index, 'player']
        cv = GroupKFold(n_splits=5)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42
            }

            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(**params, n_jobs=-1))
            ])
            score = cross_val_score(model, X, y, cv=cv, groups=groups, scoring='neg_mean_squared_error', n_jobs=-1).mean()
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=150)
        best_params = study.best_params
        self.models['rf_tuned'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(**best_params, random_state=42, n_jobs=-1))
        ])
        print(f"Best RF parameters: {best_params}")
        return study.best_value

    def train_models(self, X, y) -> dict:
        """Train all models, including CPU-tuned LightGBM, and evaluate."""
        if X is None or y is None or X.empty or y.empty:
            print("⚠️ No data to train models")
            return {}

        self.tune_model(X, y)
        self.tune_lgbm_cpu(X, y)
        self.tune_rf(X, y)

        ensemble_estimators = []
        if 'hgb_tuned' in self.models:
            ensemble_estimators.append(('hgb', self.models['hgb_tuned']['regressor']))
        if 'rf_tuned' in self.models:
            ensemble_estimators.append(('rf', self.models['rf_tuned']['regressor']))
        if 'lgbm_cpu_tuned' in self.models:
            ensemble_estimators.append(('lgbm_cpu', self.models['lgbm_cpu_tuned']['regressor']))

        if ensemble_estimators:
            ensemble_model = VotingRegressor(estimators=ensemble_estimators)
            self.models['ensemble'] = Pipeline([('scaler', StandardScaler()), ('regressor', ensemble_model)])

        scores = {}
        groups = df_global.loc[X.index, 'player']
        cv = GroupKFold(n_splits=5)

        print("\n--- Evaluating Model Performance ---")
        for name, model in self.models.items():
            try:
                cv_scores = cross_val_score(model, X, y, cv=cv, groups=groups, scoring='neg_mean_squared_error', n_jobs=-1)
                scores[name] = cv_scores.mean()
                if self.debug_mode:
                    print(f"{name} CV scores: {cv_scores}")
            except Exception as e:
                print(f"Could not evaluate model '{name}': {e}")

        across_col = f'across_season_rolling_avg_disposals_{self.rolling_window}'
        if across_col in X.columns:
            vals = pd.to_numeric(X[across_col], errors="coerce").astype(float)
            baseline_mse = np.mean((np.expm1(vals) - np.expm1(y.values)) ** 2)
            print(f"\nBaseline MSE (across-season rolling avg): {baseline_mse:.4f}")
        
        return scores

    def select_best_model(self, scores: dict) -> str:
        """Select the best model based on CV scores."""
        if not scores:
            print("⚠️ No models trained")
            return 'hgb_tuned'
        return max(scores, key=scores.get)

    def predict_current_season_disposals(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Predict disposals for the target year."""
        if player_data.empty:
            print("⚠️ Empty player data")
            return pd.DataFrame()
        player_data = self.validate_and_clean_data(player_data, self.target_year)
        if player_data.empty:
            print("⚠️ No valid data after cleaning")
            return pd.DataFrame()
        player_data = self._engineer_features_for_prediction(player_data)
        current_season_data = player_data[player_data['year'] == self.target_year].copy()
        if current_season_data.empty:
            player_name = player_data['player'].iloc[0] if 'player' in player_data.columns else 'Unknown'
            print(f"⚠️ No {self.target_year} data for {player_name}")
            return pd.DataFrame()
        dummy_cols = [col for col in ['venue', 'opponent'] if col in current_season_data.columns]
        if dummy_cols:
            current_season_data = pd.get_dummies(current_season_data, columns=dummy_cols, drop_first=True)
        
        # Reindex with training feature columns, filling missing with 0
        X_pred = current_season_data.reindex(
            columns=self.training_feature_columns,
            fill_value=0
        ).copy()
        
        # Add missing_count feature
        X_pred['missing_count'] = X_pred.isna().sum(axis=1)
        
        # Apply preprocessor
        X_transformed = self.preprocessor.transform(X_pred)
        
        # Reconstruct DataFrame with training feature columns
        X_final = pd.DataFrame(
            X_transformed,
            columns=self.training_feature_columns,
            index=X_pred.index
        )
        
        # Make predictions
        predictions = np.expm1(self.models[self.best_name].predict(X_final))
        predictions = np.clip(predictions, 0, 45)
        
        return current_season_data[['player', 'team', 'round', 'date', 'round_number']].assign(predicted_disposals=predictions)

    def get_next_round(self, df: pd.DataFrame, target_year) -> int:
        """
        Return the next round to predict for the specified target year.

        Strategy: take the largest round_number that exists in rows where
        'year' equals target_year, add 1. If no round_number is present or
        no rows match the target_year, fall back to 1.
        """
        # Check for required columns
        if 'year' not in df.columns:
            raise ValueError("'year' column is missing")
        if 'round_number' not in df.columns:
            raise ValueError("'round_number' column is missing")

        # Filter DataFrame to the target year
        df_target = df[df['year'] == target_year]

        # Extract round numbers, convert to numeric, and drop NaNs
        round_numbers = pd.to_numeric(df_target['round_number'], errors='coerce').dropna()
        print(round_numbers)

        # If no valid round numbers exist, return 1
        if round_numbers.empty:
            return 1
        
        # Compute max round and return next round
        max_round = round_numbers.max()
        print(max_round)
        return int(max_round) + 1

    def run(self):
        """Execute the prediction pipeline."""
        print("🚀 Starting AFL Disposal Prediction Pipeline...")
        try:
            df = self.load_and_prepare_data()
            if df.empty or df is None:
                raise ValueError("No valid data to process")
            
            # Early sanity check for next round
            next_round_preview = self.get_next_round(
                df if 'round_number' in df.columns else self._engineer_features_for_prediction(df), self.target_year
            )
            print(f"🔎 Early sanity-check: next_round will be {next_round_preview}")
            
            X, y = self.prepare_features_and_target(df)
            if X is None or y is None:
                raise ValueError("No historical data for training")
            scores = self.train_models(X, y)
            self.best_name = self.select_best_model(scores)
            self.models[self.best_name].fit(X, y)
            print(f"\n📊 Model Performance Summary:")
            for name, score in scores.items():
                print(f" {name}: MSE = {-score:.4f}")
            all_predictions_dfs = []
            birth_year_threshold = self.target_year - 40
            print(f"\n🔮 Generating predictions for {self.target_year}...")
            for filepath in self.data_dir.glob('*.csv'):
                if "_performance_details" not in filepath.name:
                    continue
                try:
                    player_name, dob = extract_dob_and_name(filepath)
                    if pd.isna(dob) or dob.year <= birth_year_threshold:
                        continue
                    player_df = self.load_player(filepath)
                    if 'year' not in player_df.columns or not (player_df['year'] == self.target_year).any():
                        continue
                    player_predictions = self.predict_current_season_disposals(player_df)
                    if not player_predictions.empty:
                        all_predictions_dfs.append(player_predictions)
                        print(f"✅ Generated predictions for {player_name}")
                except Exception as e:
                    print(f"❌ Prediction error for {filepath.name}: {e}")
                    continue
            if all_predictions_dfs:
                all_predictions = pd.concat(all_predictions_dfs, ignore_index=True)
                all_predictions['player'] = all_predictions['player'].str.replace(r'\s\d{8}$', '', regex=True)
                # Ensure round_number is present
                if 'round_number' not in all_predictions.columns:
                    all_predictions['round_number'] = all_predictions['round'].apply(extract_round_number)
                valid_predictions = all_predictions[all_predictions['round_number'].notnull()].copy()
                
                if not valid_predictions.empty:
                    # Get the next round
                    next_round = next_round_preview
                    
                    # Filter for future rounds
                    future_predictions = valid_predictions[valid_predictions['round_number'] >= (next_round - 1)].copy()
                    
                    if not future_predictions.empty:
                        # Get the next game per player
                        next_game_predictions = (
                            future_predictions
                                .sort_values(['round_number', 'date'])
                                .groupby('player')
                                .head(1)
                                .sort_values('predicted_disposals', ascending=False)
                        )
                        
                        if not next_game_predictions.empty:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                            prediction_dir = Path("./data/prediction")
                            prediction_dir.mkdir(parents=True, exist_ok=True)
                            csv_path = prediction_dir / f"next_round_{next_round}_prediction_{timestamp}.csv"
                            output_cols = ['player', 'team', 'predicted_disposals']  # round_number omitted
                            next_game_predictions[output_cols].to_csv(csv_path, index=False)
                            print(f"📄 Saved predictions for round {next_round} → {csv_path}")
                        else:
                            print("⚠️ No next game predictions generated after filtering")
                    else:
                        print("⚠️ No future predictions for rounds >= next_round")
                        print(f"Next round: {next_round}")
                        print(f"Available rounds: {valid_predictions['round_number'].min()} to {valid_predictions['round_number'].max()}")
                        print(f"Predictions before filter: {len(valid_predictions)}")
                else:
                    print("⚠️ No valid predictions with round numbers")
            else:
                print("⚠️ No predictions generated")
            
            # Print ML state summary
            print("\n--- Machine Learning State Summary ---")
            for name, model in self.models.items():
                print(f"\nModel: {name}")
                # Extract regressor from pipeline
                regressor = model.named_steps['regressor'] if isinstance(model, Pipeline) else model
                # Get parameters
                params = regressor.get_params()
                relevant_params = {k: v for k, v in params.items() if k in ['max_depth', 'learning_rate', 'n_estimators', 'num_leaves', 'loss', 'max_leaf_nodes', 'min_samples_leaf']}
                param_str = ", ".join(f"{k}={v}" for k, v in relevant_params.items())
                print(f"  Parameters: {param_str}")
                # Print CV score if available
                if name in scores:
                    print(f"  CV MSE: {-scores[name]:.4f}")
                # Feature importance
                if hasattr(regressor, 'feature_importances_'):
                    model.fit(X, y)  # Fit to get feature importances
                    importances = regressor.feature_importances_
                    top_indices = importances.argsort()[-5:][::-1]  # Top 5 features
                    print("  Top 5 Feature Importances:")
                    for idx in top_indices:
                        print(f"    {X.columns[idx]}: {importances[idx]:.4f}")
                elif isinstance(regressor, VotingRegressor):
                    print("  Feature importance not available for ensemble")
        except Exception as e:
            print(f"💥 Fatal error in pipeline: {e}")
            raise
        finally:
            print("\n🎉 Pipeline completed!")

if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    debug_mode = "--debug" in sys.argv
    try:
        data_dir = "./data/player_data/"
        predictor = AFLDisposalPredictor(data_dir, target_year=2025, rolling_window=5, within_season_window=3, debug_mode=debug_mode)
        predictor.run()
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)