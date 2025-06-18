import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
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
    'cba_percent': 'float32'
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
            'disposals', 'kicks', 'handballs', 'tackles', 'clearances', 'inside_50s', 'contested_possessions'
        ]
        self.extra_features = ['cba_percent', 'percentage_time_played']
        self.feature_columns = []
        self.best_name = None
        self.training_feature_columns = None
        self.preprocessor = None
        self.league_avg_disposals = None
        self.credibility_k = 5  # Credibility parameter for adjusting rolling averages

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
                        df = df.dropna(subset=[col])
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
        df = pd.concat(all_dfs, ignore_index=True)
        self.league_avg_disposals = df['disposals'].mean()
        return df

    def _validate_numeric_columns(self, df):
        """Validate and clean numeric columns before feature engineering."""
        numeric_cols = self.base_rolling_features + ['disposals']
        for col in numeric_cols:
            if col in df.columns:
                non_numeric = df[col].apply(lambda x: not pd.api.types.is_numeric_dtype(type(x)) and pd.notna(x))
                if non_numeric.any():
                    print(f"⚠️ Found non-numeric values in {col}, converting to numeric")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _add_rolling(self, df, col, window, group_by, agg_func='mean'):
        """Calculate rolling aggregates with specified window, grouping, and function."""
        try:
            if col not in df.columns:
                print(f"⚠️ Column {col} not found in DataFrame")
                return pd.Series(np.nan, index=df.index)
            df_clean = df.copy()
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            if agg_func == 'mean':
                rolling_result = (df_clean.groupby(group_by, observed=False)[col]
                                .rolling(window=window, min_periods=1)
                                .mean()
                                .reset_index(level=group_by, drop=True)
                                .shift(1))
            elif agg_func == 'std':
                rolling_result = (df_clean.groupby(group_by, observed=False)[col]
                                .rolling(window=window, min_periods=1)
                                .std()
                                .reset_index(level=group_by, drop=True)
                                .shift(1))
            else:
                raise ValueError(f"Unsupported agg_func: {agg_func}")
            return rolling_result.reindex(df.index)
        except Exception as e:
            print(f"❌ Rolling aggregate failed for {col} with agg {agg_func}: {e}")
            return pd.Series(np.nan, index=df.index)

    def _add_expanding_mean(self, df, col, group_by):
        """Calculate season-to-date expanding mean."""
        try:
            df_clean = df.copy()
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            expanding_result = (df_clean.groupby(group_by, observed=False)[col]
                              .expanding(min_periods=1)
                              .mean()
                              .reset_index(level=group_by, drop=True)
                              .shift(1))
            return expanding_result.reindex(df.index)
        except Exception as e:
            print(f"❌ Expanding mean failed for {col}: {e}")
            return pd.Series(np.nan, index=df.index)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for training with credibility adjustments for new players."""
        df = self._validate_numeric_columns(df)
        df.loc[:, 'round_number'] = df['round'].apply(extract_round_number)
        df = df.sort_values(['player', 'year', 'round_number'])
        rolling_cols_available = [col for col in self.base_rolling_features if col in df.columns]
        
        # Calculate base rolling features
        for col in rolling_cols_available:
            df.loc[:, f'across_season_rolling_avg_{col}_{self.rolling_window}'] = self._add_rolling(df, col, window=self.rolling_window, group_by=['player'])
            df.loc[:, f'within_season_rolling_avg_{col}_{self.within_season_window}'] = self._add_rolling(df, col, window=self.within_season_window, group_by=['player', 'year'])
            df.loc[:, f'season_to_date_mean_{col}'] = self._add_expanding_mean(df, col, group_by=['player', 'year'])
            df.loc[:, f'recent_form_{col}'] = df.groupby(['player'], observed=False)[col].transform(
                lambda x: x.ewm(span=3, adjust=False).mean().shift(1)
            )
            df.loc[:, f'within_season_volatility_{col}'] = self._add_rolling(df, col, window=self.within_season_window, group_by=['player', 'year'], agg_func='std')
        
        # Additional features
        if 'kicks' in df.columns and 'handballs' in df.columns:
            df.loc[:, 'kicks_to_handballs_ratio'] = df['kicks'] / (df['handballs'] + 1)
        if 'tackles' in df.columns and 'disposals' in df.columns:
            df.loc[:, 'tackles_per_disposal'] = df['tackles'] / (df['disposals'] + 1)
        df['age'] = (df['date'] - df['dob']).dt.days / 365.25
        df['games_played'] = df.groupby('player', observed=False).cumcount()
        df['days_since_last_game'] = df.groupby('player', observed=False)['date'].diff().dt.days.fillna(0)
        df['max_disposals_to_date'] = df.groupby('player', observed=False)['disposals'].transform(lambda x: x.expanding().max().shift(1))
        df['cv_disposals'] = df[f'within_season_volatility_disposals'] / (df[f'within_season_rolling_avg_disposals_{self.within_season_window}'] + 1e-6)
        
        # Calculate games_in_season for within-season adjustments
        df['games_in_season'] = df.groupby(['player', 'year'], observed=False).cumcount()
        
        # Adjust rolling features with credibility for players with limited data
        for col in rolling_cols_available:
            df[f'adjusted_across_season_rolling_avg_{col}_{self.rolling_window}'] = df.apply(
                lambda row: (
                    (row['games_played'] / (row['games_played'] + self.credibility_k)) * row[f'across_season_rolling_avg_{col}_{self.rolling_window}'] +
                    (self.credibility_k / (row['games_played'] + self.credibility_k)) * self.league_avg_disposals
                ) if pd.notnull(row[f'across_season_rolling_avg_{col}_{self.rolling_window}']) else self.league_avg_disposals,
                axis=1
            )
            df[f'adjusted_within_season_rolling_avg_{col}_{self.within_season_window}'] = df.apply(
                lambda row: (
                    (row['games_in_season'] / (row['games_in_season'] + self.credibility_k)) * row[f'within_season_rolling_avg_{col}_{self.within_season_window}'] +
                    (self.credibility_k / (row['games_in_season'] + self.credibility_k)) * self.league_avg_disposals
                ) if pd.notnull(row[f'within_season_rolling_avg_{col}_{self.within_season_window}']) else self.league_avg_disposals,
                axis=1
            )
        
        # Define feature sets with adjusted rolling averages
        across_season_features = [f'adjusted_across_season_rolling_avg_{col}_{self.rolling_window}' for col in rolling_cols_available]
        within_season_features = [f'adjusted_within_season_rolling_avg_{col}_{self.within_season_window}' for col in rolling_cols_available]
        season_to_date_features = [f'season_to_date_mean_{col}' for col in rolling_cols_available]
        recent_form_features = [f'recent_form_{col}' for col in rolling_cols_available]
        volatility_features = [f'within_season_volatility_{col}' for col in rolling_cols_available]
        interaction_features = ['kicks_to_handballs_ratio', 'tackles_per_disposal']
        extra_feats = [feat for feat in self.extra_features if feat in df.columns]
        dummy_cols = [c for c in df.columns if c.startswith(('venue_', 'opponent_'))]
        self.feature_columns = (
            across_season_features + within_season_features + season_to_date_features + 
            recent_form_features + volatility_features + interaction_features + 
            ['round_number', 'days_since_last_game', 'age', 'games_played', 'max_disposals_to_date', 'cv_disposals'] + 
            extra_feats + dummy_cols
        )
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        return df.dropna(subset=across_season_features + within_season_features + ['disposals'])

    def _engineer_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for prediction with credibility adjustments."""
        df.loc[:, 'round_number'] = df['round'].apply(extract_round_number)
        df = df.sort_values(['player', 'year', 'round_number'])
        rolling_cols_available = [col for col in self.base_rolling_features if col in df.columns]
        
        # Calculate base rolling features
        for col in rolling_cols_available:
            df.loc[:, f'across_season_rolling_avg_{col}_{self.rolling_window}'] = self._add_rolling(df, col, window=self.rolling_window, group_by=['player'])
            df.loc[:, f'within_season_rolling_avg_{col}_{self.within_season_window}'] = self._add_rolling(df, col, window=self.within_season_window, group_by=['player', 'year'])
            df.loc[:, f'season_to_date_mean_{col}'] = self._add_expanding_mean(df, col, group_by=['player', 'year'])
            df.loc[:, f'recent_form_{col}'] = df.groupby(['player'], observed=False)[col].transform(
                lambda x: x.ewm(span=3, adjust=False).mean().shift(1)
            )
            df.loc[:, f'within_season_volatility_{col}'] = self._add_rolling(df, col, window=self.within_season_window, group_by=['player', 'year'], agg_func='std')
        
        # Additional features
        if 'kicks' in df.columns and 'handballs' in df.columns:
            df.loc[:, 'kicks_to_handballs_ratio'] = df['kicks'] / (df['handballs'] + 1)
        if 'tackles' in df.columns and 'disposals' in df.columns:
            df.loc[:, 'tackles_per_disposal'] = df['tackles'] / (df['disposals'] + 1)
        df['age'] = (df['date'] - df['dob']).dt.days / 365.25
        df['games_played'] = df.groupby('player', observed=False).cumcount()
        df['days_since_last_game'] = df.groupby('player', observed=False)['date'].diff().dt.days.fillna(0)
        df['max_disposals_to_date'] = df.groupby('player', observed=False)['disposals'].transform(lambda x: x.expanding().max().shift(1))
        df['cv_disposals'] = df[f'within_season_volatility_disposals'] / (df[f'within_season_rolling_avg_disposals_{self.within_season_window}'] + 1e-6)
        
        # Calculate games_in_season for within-season adjustments
        df['games_in_season'] = df.groupby(['player', 'year'], observed=False).cumcount()
        
        # Adjust rolling features with credibility for players with limited data
        for col in rolling_cols_available:
            df[f'adjusted_across_season_rolling_avg_{col}_{self.rolling_window}'] = df.apply(
                lambda row: (
                    (row['games_played'] / (row['games_played'] + self.credibility_k)) * row[f'across_season_rolling_avg_{col}_{self.rolling_window}'] +
                    (self.credibility_k / (row['games_played'] + self.credibility_k)) * self.league_avg_disposals
                ) if pd.notnull(row[f'across_season_rolling_avg_{col}_{self.rolling_window}']) else self.league_avg_disposals,
                axis=1
            )
            df[f'adjusted_within_season_rolling_avg_{col}_{self.within_season_window}'] = df.apply(
                lambda row: (
                    (row['games_in_season'] / (row['games_in_season'] + self.credibility_k)) * row[f'within_season_rolling_avg_{col}_{self.within_season_window}'] +
                    (self.credibility_k / (row['games_in_season'] + self.credibility_k)) * self.league_avg_disposals
                ) if pd.notnull(row[f'within_season_rolling_avg_{col}_{self.within_season_window}']) else self.league_avg_disposals,
                axis=1
            )
        return df

    def prepare_features_and_target(self, df: pd.DataFrame) -> tuple:
        """Prepare features (X) and target (y) for training with winsorization."""
        global df_global
        df_global = df
        historical_data = df[df['year'] < self.target_year].copy()
        historical_data = historical_data[historical_data['disposals'].notnull()]
        if historical_data.empty:
            print("⚠️ No historical data available")
            return None, None
        engineered_df = self._engineer_features(historical_data)
        percentile_1 = historical_data['disposals'].quantile(0.01)
        percentile_99 = historical_data['disposals'].quantile(0.99)
        clipped_disposals = engineered_df['disposals'].clip(lower=percentile_1, upper=percentile_99)
        y = np.log1p(clipped_disposals)
        X = engineered_df[self.feature_columns]
        self.training_feature_columns = self.feature_columns.copy()
        X.loc[:, 'missing_count'] = X.isna().sum(axis=1)
        self.training_feature_columns.append('missing_count')
        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = X.select_dtypes(include=['category']).columns
        remainder_cols = [col for col in X.columns if col not in numerical_cols and col not in categorical_cols]
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), numerical_cols),
                ('cat', SimpleImputer(strategy='constant', fill_value='missing'), categorical_cols)
            ],
            remainder='passthrough'
        )
        X_transformed = preprocessor.fit_transform(X)
        transformed_columns = list(numerical_cols) + list(categorical_cols) + remainder_cols
        X = pd.DataFrame(X_transformed, columns=transformed_columns, index=X.index)
        X = X.astype(float)
        X = X.reindex(columns=self.training_feature_columns, fill_value=0)
        self.preprocessor = preprocessor
        return X, y

    def tune_lgbm_gpu(self, X, y):
        """Hyperparameter tuning for LGBMRegressor on GPU using Optuna."""
        def objective(trial):
            params = {
                'device': 'cuda',
                'objective': trial.suggest_categorical('objective', ['regression_l1', 'regression_l2', 'huber', 'poisson']),
                'metric': 'rmse',
                'random_state': 42,
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255])
            }
            if params['objective'] == 'huber':
                params['alpha'] = trial.suggest_float('alpha', 0.1, 10.0)
            model = LGBMRegressor(**params)
            groups = df_global.loc[X.index, 'player']
            cv = GroupKFold(n_splits=5)
            score = cross_val_score(model, X, y, cv=cv, groups=groups, scoring='neg_mean_squared_error').mean()
            return score
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, timeout=600)
        best_params = study.best_params
        self.models['lgbm_gpu_tuned'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LGBMRegressor(**best_params, device='cuda', random_state=42))
        ])
        return study.best_value

    def train_models(self, X, y) -> dict:
        """Train all models, including CatBoost and a Stacking ensemble, and evaluate."""
        if X is None or y is None or X.empty or y.empty:
            print("⚠️ No data to train models")
            return {}
        self.tune_lgbm_gpu(X, y)
        self.models['hgb'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', HistGradientBoostingRegressor(loss='poisson', learning_rate=0.05, max_depth=6, random_state=42))
        ])
        self.models['rf'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42))
        ])
        self.models['catboost'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', CatBoostRegressor(verbose=0, random_state=42, allow_writing_files=False))
        ])
        estimators = [
            ('hgb', self.models['hgb']['regressor']),
            ('rf', self.models['rf']['regressor']),
            ('lgbm_gpu', self.models['lgbm_gpu_tuned']['regressor']),
            ('catboost', self.models['catboost']['regressor'])
        ]
        meta_learner = Ridge(random_state=42)
        stacking_model = StackingRegressor(estimators=estimators, final_estimator=meta_learner, cv=5)
        self.models['stacking_ensemble'] = Pipeline([('scaler', StandardScaler()), ('regressor', stacking_model)])
        scores = {}
        groups = df_global.loc[X.index, 'player']
        cv = GroupKFold(n_splits=5)
        for name in ['hgb', "rf", 'lgbm_gpu_tuned', 'catboost', 'stacking_ensemble']:
            model = self.models[name]
            cv_scores = cross_val_score(model, X, y, cv=cv, groups=groups, scoring='neg_mean_squared_error')
            scores[name] = cv_scores.mean()
        return scores

    def select_best_model(self, scores: dict) -> str:
        """Select the stacking ensemble as the best model."""
        if not scores:
            print("⚠️ No models trained")
            return 'hgb'
        self.best_name = 'stacking_ensemble'
        return self.best_name

    def predict_current_season_disposals(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Predict disposals for the target year with stricter dynamic clipping."""
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
        X_pred = current_season_data.reindex(columns=self.training_feature_columns, fill_value=0).copy()
        X_pred['missing_count'] = X_pred.isna().sum(axis=1)
        X_transformed = self.preprocessor.transform(X_pred)
        X_final = pd.DataFrame(X_transformed, columns=self.training_feature_columns, index=X_pred.index)
        raw_predictions = self.models[self.best_name].predict(X_final)
        predictions = np.expm1(raw_predictions)
        # Stricter dynamic clipping
        max_realistic = current_season_data[f'adjusted_across_season_rolling_avg_disposals_{self.rolling_window}'] + 1.5 * current_season_data[f'within_season_volatility_disposals']
        max_realistic = max_realistic.fillna(25)
        predictions = np.clip(predictions, 0, max_realistic)
        predictions = np.clip(predictions, 0, 40)
        return current_season_data[['player', 'team', 'round', 'date', 'round_number']].assign(predicted_disposals=predictions)

    def get_next_round(self, df: pd.DataFrame, target_year) -> int:
        """Return the next round to predict for the specified target year."""
        if 'year' not in df.columns or 'round_number' not in df.columns:
            raise ValueError("Required columns are missing")
        df_target = df[df['year'] == target_year]
        round_numbers = pd.to_numeric(df_target['round_number'], errors='coerce').dropna()
        if round_numbers.empty:
            return 1
        max_round = round_numbers.max()
        return int(max_round) + 1

    def run(self):
        """Execute the prediction pipeline."""
        print("🚀 Starting AFL Disposal Prediction Pipeline...")
        try:
            df = self.load_and_prepare_data()
            if df.empty:
                raise ValueError("No valid data to process")
            next_round_preview = self.get_next_round(df if 'round_number' in df.columns else self._engineer_features_for_prediction(df), self.target_year)
            print(f"🔎 Next round to predict: {next_round_preview}")
            X, y = self.prepare_features_and_target(df)
            if X is None or y is None:
                raise ValueError("No historical data for training")
            scores = self.train_models(X, y)
            self.select_best_model(scores)
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
                if 'round_number' not in all_predictions.columns:
                    all_predictions['round_number'] = all_predictions['round'].apply(extract_round_number)
                valid_predictions = all_predictions[all_predictions['round_number'].notnull()].copy()
                if not valid_predictions.empty:
                    future_predictions = valid_predictions[valid_predictions['round_number'] >= (next_round_preview - 1)].copy()
                    if not future_predictions.empty:
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
                            csv_path = prediction_dir / f"next_round_{next_round_preview}_prediction_{timestamp}.csv"
                            output_cols = ['player', 'team', 'predicted_disposals']
                            next_game_predictions[output_cols].to_csv(csv_path, index=False)
                            print(f"📄 Saved predictions for round {next_round_preview} → {csv_path}")
                        else:
                            print("⚠️ No next game predictions generated after filtering")
                    else:
                        print("⚠️ No future predictions for rounds >= next_round")
                else:
                    print("⚠️ No valid predictions with round numbers")
            else:
                print("⚠️ No predictions generated")
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