import argparse
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

try:
    from scripts.feature_engineering import (
        compute_age_years,
        compute_career_games_to_date,
    )
except ModuleNotFoundError:  # direct-path invocation: put repo root on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.feature_engineering import (
        compute_age_years,
        compute_career_games_to_date,
    )


def _detect_lgbm_device() -> str:
    """Probe LightGBM GPU support and return the appropriate device string.

    Returns 'gpu' if a tiny LightGBM fit with device='gpu' succeeds, else 'cpu'.
    Cached as a module-level constant so the probe runs once per process.
    Kept dependency-light (no torch import) — we ask LightGBM itself whether
    it can use a GPU in this build/environment, which is the authoritative
    check for this code path.
    """
    import os
    try:
        import numpy as _np
        _probe_X = _np.random.RandomState(0).rand(64, 4)
        _probe_y = _np.random.RandomState(1).rand(64)
        # Silence LightGBM's C-level stderr ("[Fatal] GPU Tree Learner was not
        # enabled in this build.") on CPU-only hosts — the probe's exception
        # is the real signal; the log line is misleading noise. We dup the
        # OS-level stderr fd because Python-level redirection doesn't capture
        # writes from the LightGBM C extension.
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        _saved_stderr_fd = os.dup(2)
        try:
            os.dup2(_devnull_fd, 2)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                LGBMRegressor(
                    device='gpu',
                    n_estimators=1,
                    num_leaves=4,
                    min_child_samples=1,
                    verbose=-1,
                ).fit(_probe_X, _probe_y)
        finally:
            os.dup2(_saved_stderr_fd, 2)
            os.close(_saved_stderr_fd)
            os.close(_devnull_fd)
        return 'gpu'
    except Exception:
        return 'cpu'


LGBM_DEVICE = _detect_lgbm_device()

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
    # NOTE: 'cba_percent' (centre-bounce-attendance %) removed — no such
    # column, nor any equivalent, exists in data/player_data/ (Task S1b).
}

NA_VALUES = ['NA', 'N/A', '', 'nan']

# Column renaming for consistency
RENAMES = {
    'hit_outs': 'hitouts',
    'free_kicks_for': 'frees_for',
    'free_kicks_against': 'frees_against',
    # Time-on-ground %: the raw CSV column is 'percentage_of_game_played';
    # the model declares it as 'percentage_time_played'. Without this rename
    # the feature never resolved and was silently dropped (Task S1b).
    'percentage_of_game_played': 'percentage_time_played',
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
    def __init__(self, data_dir: str, target_year: int | None = None, rolling_window: int = 5, within_season_window: int = 3, debug_mode: bool = False, include_age_experience: bool = False):
        """Initialize predictor with data directory, target year, rolling window sizes, and debug mode.

        If ``target_year`` is None, it is auto-detected as the maximum ``year``
        observed across all player_data CSVs (falling back to the current
        calendar year if detection fails).

        ``include_age_experience`` (default False) is an OPT-IN switch for the
        Task S7 features (player_age_at_match, career_games_to_date). It is off
        by default so production predictions are unchanged until a backtest
        comparison validates the features (see docs/experiment-log.md). Both
        features are leak-proof; see scripts/feature_engineering.py.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {data_dir}")

        if target_year is None:
            target_year = self._autodetect_target_year()
            print(f"🗓  Auto-detected target_year = {target_year}")

        self.target_year = target_year
        self.rolling_window = rolling_window
        self.within_season_window = within_season_window
        self.debug_mode = debug_mode
        self.include_age_experience = include_age_experience
        self.models = {}
        self.base_rolling_features = [
            'disposals', 'kicks', 'handballs', 'tackles', 'clearances', 'inside_50s'
        ]
        # 'percentage_time_played' is wired from the raw 'percentage_of_game_played'
        # column via RENAMES. 'cba_percent' was dropped: no centre-bounce data
        # exists anywhere in data/player_data/ (Task S1b).
        self.extra_features = ['percentage_time_played']
        self.feature_columns = []
        self.best_name = None
        self.training_feature_columns = None
        self.preprocessor = None  # Store preprocessor for prediction
        # Groups (player) for the training rows, set during
        # ``prepare_features_and_target`` and consumed by GroupKFold-based
        # tuning/eval. Replaces an earlier module-level ``df_global``.
        self._train_groups: pd.Series | None = None
        # Cache of {filepath: loaded DataFrame} populated during
        # load_and_prepare_data() and reused for prediction so we don't
        # re-read & re-parse every CSV a second time.
        self._player_cache: dict = {}

    def _autodetect_target_year(self) -> int:
        """Return the highest ``year`` value present in any player CSV.

        Falls back to the current calendar year when no usable data is found,
        rather than guessing a fixed value that decays as time passes.
        """
        max_year = None
        # Scan CSVs lightly — only read the 'year' column so this is fast even
        # over the full player_data directory.
        for filepath in self.data_dir.glob('*_performance_details.csv'):
            try:
                years = pd.read_csv(filepath, usecols=['year'], na_values=NA_VALUES)
                years = pd.to_numeric(years['year'], errors='coerce').dropna()
                if not years.empty:
                    file_max = int(years.max())
                    if max_year is None or file_max > max_year:
                        max_year = file_max
            except Exception:
                # Bad / unreadable CSVs are tolerated — we just skip them.
                continue
        if max_year is None:
            fallback = datetime.now().year
            print(f"⚠️  Could not auto-detect target_year from data; falling back to {fallback}")
            return fallback
        return max_year

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
            # Carry DOB (from the filename token, verified == personal_details
            # born_date) as the join channel for the opt-in age feature.
            if self.include_age_experience:
                df['born_date'] = dob
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
        # Keep rows where 'disposals' is null only for target_year (current
        # season may have fixture rows or partial data with disposals == NaN
        # that we still need to keep around for feature engineering).
        df = df[(df['year'] == target_year) | df['disposals'].notnull()].copy()
        cleaned_rows = len(df)
        if original_rows > cleaned_rows:
            print(f"⚠️ Dropped {original_rows - cleaned_rows} rows due to missing 'disposals' in historical data")
        return df

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and combine data from all player CSV files.

        Populates ``self._player_cache`` keyed by filepath so the prediction
        pass can reuse the parsed per-player DataFrames without re-reading
        the CSVs from disk.
        """
        all_dfs = []
        birth_year_threshold = self.target_year - 40
        self._player_cache = {}
        for filepath in self.data_dir.glob('*.csv'):
            if "_performance_details" not in filepath.name:
                continue
            try:
                player_name, dob = extract_dob_and_name(filepath)
                if pd.isna(dob) or dob.year <= birth_year_threshold:
                    continue
                df = self.load_player(filepath)
                if not df.empty:
                    self._player_cache[filepath] = df
                    all_dfs.append(df)
            except Exception as e:
                print(f"❌ Error processing file {filepath.name}: {e}")
                continue
        if not all_dfs:
            print("⚠️ No valid data loaded")
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)

    @staticmethod
    def _group_keys(df, group_by):
        """Return the list of Series used to identify groups for ``group_by``."""
        if isinstance(group_by, (list, tuple)):
            return [df[g] for g in group_by]
        return [df[group_by]]

    def _add_rolling(self, df, col, window, group_by):
        """Calculate rolling averages with specified window and grouping.

        Equivalent to ``groupby(...).transform(lambda s: s.rolling(...).mean().shift(1))``
        but uses pandas' native ``GroupBy.rolling`` followed by a separate
        ``GroupBy.shift`` — avoids the per-group python lambda dispatch and
        is multiple-times faster on the (player, year) groupings used here.
        """
        try:
            grouper = df.groupby(group_by, observed=False, sort=False)[col]
            rolled = grouper.rolling(window=window, min_periods=1).mean()
            # ``rolled`` has a multi-index (group_keys..., original_index);
            # drop the group-key levels and reindex back to df's index so
            # the result lines up with df even when df's index is unsorted
            # (e.g. after sort_values reshuffled it).
            n_levels = len(group_by) if isinstance(group_by, (list, tuple)) else 1
            rolled = rolled.reset_index(
                level=list(range(n_levels)), drop=True
            ).reindex(df.index)
            return rolled.groupby(self._group_keys(df, group_by)).shift(1)
        except Exception as e:
            print(f"❌ Rolling average failed for {col} with group_by {group_by}: {e}")
            return pd.Series(np.nan, index=df.index)

    def _add_expanding_mean(self, df, col, group_by):
        """Calculate season-to-date expanding mean with specified grouping.

        Same vectorisation rationale as :meth:`_add_rolling`.
        """
        try:
            grouper = df.groupby(group_by, observed=False, sort=False)[col]
            expanded = grouper.expanding().mean()
            n_levels = len(group_by) if isinstance(group_by, (list, tuple)) else 1
            expanded = expanded.reset_index(
                level=list(range(n_levels)), drop=True
            ).reindex(df.index)
            return expanded.groupby(self._group_keys(df, group_by)).shift(1)
        except Exception as e:
            print(f"❌ Expanding mean failed for {col} with group_by {group_by}: {e}")
            return pd.Series(np.nan, index=df.index)

    # Names of the opt-in Task S7 features (kept as a constant so training and
    # prediction paths reference the exact same identifiers).
    AGE_EXPERIENCE_FEATURES = ['player_age_at_match', 'career_games_to_date']

    def _add_age_experience_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach the opt-in leak-proof age/experience features in place.

        Requires a 'round_number' column (added by the caller before this runs)
        and, for age, a 'born_date' column (added in load_player when
        include_age_experience is on). Both features use only current-row-and-
        earlier information — see scripts/feature_engineering.py for the
        temporal-cutoff invariant.
        """
        if 'born_date' in df.columns:
            df.loc[:, 'player_age_at_match'] = compute_age_years(df['date'], df['born_date'])
        else:
            df.loc[:, 'player_age_at_match'] = np.nan
        df.loc[:, 'career_games_to_date'] = compute_career_games_to_date(df).astype('float64')
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for training, including rolling averages and more."""
        df.loc[:, 'round_number'] = df['round'].apply(extract_round_number)
        df = df.sort_values(['player', 'year', 'round_number'])
        
        rolling_cols_available = [col for col in self.base_rolling_features if col in df.columns]
        
        for col in rolling_cols_available:
            df.loc[:, f'across_season_rolling_avg_{col}_{self.rolling_window}'] = self._add_rolling(df, col, window=self.rolling_window, group_by=['player'])
            df.loc[:, f'within_season_rolling_avg_{col}_{self.within_season_window}'] = self._add_rolling(df, col, window=self.within_season_window, group_by=['player', 'year'])
            df.loc[:, f'season_to_date_mean_{col}'] = self._add_expanding_mean(df, col, group_by=['player', 'year'])
            df.loc[:, f'recent_form_{col}'] = df.groupby(['player'], observed=False)[col].transform(
                lambda x: x.ewm(span=3, adjust=False).mean().shift(1)
            )
        
        df['days_since_last_game'] = df.groupby('player', observed=False)['date'].diff().dt.days.fillna(0)

        if self.include_age_experience:
            df = self._add_age_experience_features(df)

        across_season_features = [f'across_season_rolling_avg_{col}_{self.rolling_window}' for col in rolling_cols_available]
        within_season_features = [f'within_season_rolling_avg_{col}_{self.within_season_window}' for col in rolling_cols_available]
        season_to_date_features = [f'season_to_date_mean_{col}' for col in rolling_cols_available]
        recent_form_features = [f'recent_form_{col}' for col in rolling_cols_available]
        extra_feats = [feat for feat in self.extra_features if feat in df.columns]
        age_exp_feats = list(self.AGE_EXPERIENCE_FEATURES) if self.include_age_experience else []
        dummy_cols = [c for c in df.columns if c.startswith(('venue_', 'opponent_'))]
        self.feature_columns = across_season_features + within_season_features + season_to_date_features + recent_form_features + ['round_number', 'days_since_last_game'] + extra_feats + age_exp_feats + dummy_cols
        
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
        
        rolling_cols_available = [col for col in self.base_rolling_features if col in df.columns]
        
        for col in rolling_cols_available:
            df.loc[:, f'across_season_rolling_avg_{col}_{self.rolling_window}'] = self._add_rolling(df, col, window=self.rolling_window, group_by=['player'])
            df.loc[:, f'within_season_rolling_avg_{col}_{self.within_season_window}'] = self._add_rolling(df, col, window=self.within_season_window, group_by=['player', 'year'])
            df.loc[:, f'season_to_date_mean_{col}'] = self._add_expanding_mean(df, col, group_by=['player', 'year'])
            df.loc[:, f'recent_form_{col}'] = df.groupby(['player'], observed=False)[col].transform(
                lambda x: x.ewm(span=3, adjust=False).mean().shift(1)
            )
        
        df['days_since_last_game'] = df.groupby('player', observed=False)['date'].diff().dt.days.fillna(0)

        if self.include_age_experience:
            df = self._add_age_experience_features(df)

        return df

    def prepare_features_and_target(self, df: pd.DataFrame) -> tuple:
        """Prepare features (X) and target (y) for training."""
        historical_data = df[df['year'] < self.target_year].copy()
        # Drop rows whose target is NaN — we can't train on them.
        historical_data = historical_data[historical_data['disposals'].notnull()]
        if historical_data.empty:
            print("⚠️ No historical data available")
            return None, None
        engineered_df = self._engineer_features(historical_data)
        X = engineered_df[self.feature_columns].copy()
        # Backtest evidence (R1-R8 2026, n=2879): predicting on log1p(disposals)
        # then expm1 back caused severe top-end compression — max prediction
        # was 28 vs max actual 43, and 30+ disposal games were under-predicted
        # by 10-19. The Poisson-loss HGB candidate already uses a log link
        # internally, so log1p was double-compressing high values. Train on
        # raw disposals; the trees handle the mild right skew fine, and
        # Poisson loss still works (target is non-negative).
        y = engineered_df['disposals'].astype(float)
        # Track player groups for GroupKFold so tuning/eval don't leak the
        # same player across train and validation folds. cross_val_score
        # iterates positionally, so we keep groups aligned by row order
        # with X (same source DataFrame ⇒ same length & row order).
        self._train_groups = engineered_df['player'].astype('object').to_numpy()
        self.training_feature_columns = self.feature_columns.copy()

        # Compute missing_count BEFORE any imputation. We base it only on
        # ``self.feature_columns`` (engineered features) so that train and
        # predict both compute the same quantity over the same domain.
        X.loc[:, 'missing_count'] = X[self.feature_columns].isna().sum(axis=1)
        self.training_feature_columns.append('missing_count')

        # Numeric columns: ALL integer/float dtypes (incl. float32, Int16,
        # nullable Int/Float). Random Forest can't tolerate NaN, so e.g.
        # cba_percent (float32, NaN pre-2018) MUST be routed through the
        # median imputer rather than the passthrough branch — the previous
        # selection of just ['float64','int64'] silently dropped float32 /
        # nullable cols into 'remainder' and crashed RF on NaN.
        numerical_cols = X.select_dtypes(include=['number']).columns
        categorical_cols = X.select_dtypes(include=['category']).columns
        remainder_cols = [col for col in X.columns if col not in numerical_cols and col not in categorical_cols]

        # Cast nullable / narrow numeric dtypes to plain float64 BEFORE
        # imputation. SimpleImputer + sklearn don't always handle pandas
        # extension dtypes (Int16, Float32) cleanly, and downstream
        # estimators expect a homogeneous float matrix anyway.
        for c in numerical_cols:
            X[c] = pd.to_numeric(X[c], errors='coerce').astype('float64')

        if self.debug_mode:
            print(f"Numerical columns: {list(numerical_cols)}")
            print(f"Categorical columns: {list(categorical_cols)}")
            print(f"Remainder columns: {remainder_cols}")
            print(f"Expected columns: {self.training_feature_columns}")

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), list(numerical_cols)),
                ('cat', SimpleImputer(strategy='constant', fill_value='missing'), list(categorical_cols))
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
        """Hyperparameter tuning for HistGradientBoostingRegressor using Optuna.

        Uses GroupKFold by player so the same player can't appear in both
        train and validation folds — otherwise tuning rewards memorising
        per-player tendencies and overstates generalisation.
        """
        # Target is raw disposals (counts >= 0). Poisson loss requires y>=0,
        # which holds. squared_error and quantile remain as candidates so
        # Optuna can pick the loss that best matches our right-skewed target.
        groups = self._train_groups
        cv = GroupKFold(n_splits=5)

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 100),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
                'l2_regularization': trial.suggest_float('l2_regularization', 1e-6, 1.0, log=True),
                'loss': trial.suggest_categorical('loss', ['poisson', 'quantile', 'squared_error']),
            }

            if params['loss'] == 'quantile':
                params['quantile'] = trial.suggest_float('quantile', 0.1, 0.9)

            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', HistGradientBoostingRegressor(**params, random_state=42))
            ])
            score = cross_val_score(
                model, X, y, cv=cv, groups=groups, scoring='neg_mean_squared_error'
            ).mean()
            return score

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        self.models['hgb_tuned'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', HistGradientBoostingRegressor(**best_params, random_state=42))
        ])
        print(f"Best parameters: {best_params}")
        return study.best_value

    def tune_lgbm_gpu(self, X, y):
        """Hyperparameter tuning for LGBMRegressor using Optuna.

        Device is selected at module load via _detect_lgbm_device():
        uses 'gpu' when LightGBM can train on a GPU in this environment,
        and falls back to 'cpu' otherwise. LightGBM CPU produces fits
        equivalent to GPU given the same seed, just slower — so the
        fallback is correctness-preserving but the 600s Optuna timeout
        will explore a smaller search space on CPU-only hosts.
        """
        print(f"Tuning LGBM ({LGBM_DEVICE.upper()}) with Optuna...")

        groups = self._train_groups
        cv = GroupKFold(n_splits=5)

        def objective(trial):
            params = {
                'device': LGBM_DEVICE,  # 'gpu' if available, else CPU fallback
                'verbose': -1,
                # Switched from regression_l1 (MAE → predicts the median) to
                # regression (L2 → predicts the mean). Backtest showed L1 was
                # baking in median-bias on right-skewed disposals, so 30+
                # disposal games systematically under-predicted by 7-15.
                # L2 is slightly more sensitive to outliers but materially
                # better calibrated at the top end, which is where the
                # SuperCoach scoring and player-of-interest decisions live.
                'objective': 'regression',
                'metric': 'rmse',
                'random_state': 42,
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'subsample_freq': 1,
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255])  # Key GPU performance parameter
            }

            model = LGBMRegressor(**params)

            score = cross_val_score(
                model, X, y, cv=cv, groups=groups, scoring='neg_mean_squared_error'
            ).mean()

            return score

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=50, timeout=600)  # 10-minute timeout

        print(f"Best LGBM GPU parameters: {study.best_params}")

        self.models['lgbm_gpu_tuned'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LGBMRegressor(**study.best_params, device=LGBM_DEVICE, verbose=-1, random_state=42))
        ])

        return study.best_value

    def train_models(self, X, y) -> dict:
        """Train all models, including GPU-tuned LightGBM, and evaluate."""
        if X is None or y is None or X.empty or y.empty:
            print("⚠️ No data to train models")
            return {}

        self.tune_model(X, y)
        self.tune_lgbm_gpu(X, y)

        self.models['rf'] = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42))
        ])

        ensemble_estimators = []
        if 'hgb_tuned' in self.models:
            ensemble_estimators.append(('hgb', self.models['hgb_tuned']['regressor']))
        if 'rf' in self.models:
            ensemble_estimators.append(('rf', self.models['rf']['regressor']))
        if 'lgbm_gpu_tuned' in self.models:
            ensemble_estimators.append(('lgbm_gpu', self.models['lgbm_gpu_tuned']['regressor']))

        if ensemble_estimators:
            ensemble_model = VotingRegressor(estimators=ensemble_estimators)
            self.models['ensemble'] = Pipeline([('scaler', StandardScaler()), ('regressor', ensemble_model)])

        scores = {}
        groups = self._train_groups
        cv = GroupKFold(n_splits=5)

        print("\n--- Evaluating Model Performance ---")
        for name, model in self.models.items():
            try:
                cv_scores = cross_val_score(model, X, y, cv=cv, groups=groups, scoring='neg_mean_squared_error')
                scores[name] = cv_scores.mean()
                if self.debug_mode:
                    print(f"{name} CV scores: {cv_scores}")
            except Exception as e:
                print(f"Could not evaluate model '{name}': {e}")

        across_col = f'across_season_rolling_avg_disposals_{self.rolling_window}'
        if across_col in X.columns:
            vals = pd.to_numeric(X[across_col], errors="coerce").astype(float)
            # y is now raw disposals (no log1p) — compare directly.
            baseline_mse = np.mean((vals - y.values) ** 2)
            print(f"\nBaseline MSE (across-season rolling avg): {baseline_mse:.4f}")
        
        return scores

    def select_best_model(self, scores: dict) -> str:
        """Select the best model based on CV scores."""
        if not scores:
            print("⚠️ No models trained")
            return 'hgb_tuned'
        return max(scores, key=scores.get)

    def _fit_calibration(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit a linear (slope, intercept) calibration on out-of-fold predictions.

        We perform a GroupKFold pass with the chosen best model, collect OOF
        predictions, and fit ``actual = a * pred + b`` via least squares.
        Stored as ``self._calib_slope`` / ``self._calib_intercept``. If
        anything goes wrong, calibration falls back to identity (a=1, b=0)
        so we never silently degrade predictions.
        """
        # Identity defaults — used if calibration fails or is rejected.
        self._calib_slope = 1.0
        self._calib_intercept = 0.0
        try:
            from sklearn.base import clone
            groups = self._train_groups
            cv = GroupKFold(n_splits=5)
            model_template = self.models[self.best_name]
            oof_pred = np.full(len(y), np.nan)
            for tr_idx, va_idx in cv.split(X, y, groups=groups):
                m = clone(model_template)
                m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                oof_pred[va_idx] = m.predict(X.iloc[va_idx])
            mask = ~np.isnan(oof_pred)
            if mask.sum() < 100:
                print("⚠️ Calibration: too few OOF preds, skipping (using identity)")
                return
            p = oof_pred[mask]
            t = y.values[mask].astype(float)
            # Closed-form OLS for actual = a*pred + b
            p_mean = p.mean(); t_mean = t.mean()
            denom = ((p - p_mean) ** 2).sum()
            if denom <= 0:
                print("⚠️ Calibration: zero-variance OOF preds, using identity")
                return
            a = float(((p - p_mean) * (t - t_mean)).sum() / denom)
            b = float(t_mean - a * p_mean)
            # Sanity-bound: a sensible calibration should be within (0.5, 2.0)
            # in slope; anything outside that is more likely a numerical issue
            # than a real correction we want to apply.
            if not (0.5 <= a <= 2.0) or abs(b) > 20.0:
                print(f"⚠️ Calibration out of bounds (a={a:.3f}, b={b:.3f}); using identity")
                return
            self._calib_slope = a
            self._calib_intercept = b
            # Diagnostics — quantify pre/post bias and compression.
            raw_bias = (p - t).mean()
            cal = a * p + b
            cal_bias = (cal - t).mean()
            print(
                f"📐 Calibration fit: actual ≈ {a:.3f}·pred + {b:+.3f}  "
                f"(OOF bias {raw_bias:+.3f} → {cal_bias:+.3f}, "
                f"OOF max pred {p.max():.1f} → {cal.max():.1f}, "
                f"actual max {t.max():.1f})"
            )
        except Exception as e:
            print(f"⚠️ Calibration fit failed ({e}); using identity")

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
        
        # Compute missing_count over the engineered-feature domain BEFORE
        # reindexing — the reindex below fills truly-absent columns with 0
        # and would otherwise hide their NaN-ness from the count, producing
        # a feature distribution that doesn't match training.
        engineered_cols = [c for c in self.feature_columns if c in current_season_data.columns]
        missing_count_pred = current_season_data[engineered_cols].isna().sum(axis=1)

        # Reindex with training feature columns, filling absent ones with 0.
        X_pred = current_season_data.reindex(
            columns=self.training_feature_columns,
            fill_value=0
        ).copy()
        X_pred['missing_count'] = missing_count_pred.reindex(X_pred.index).fillna(0).astype(int)

        # Cast numeric cols to float so transform mirrors training-time dtypes.
        for c in X_pred.columns:
            if pd.api.types.is_numeric_dtype(X_pred[c]):
                X_pred[c] = pd.to_numeric(X_pred[c], errors='coerce').astype('float64')

        # Apply preprocessor
        X_transformed = self.preprocessor.transform(X_pred)

        # Reconstruct DataFrame with training feature columns
        X_final = pd.DataFrame(
            X_transformed,
            columns=self.training_feature_columns,
            index=X_pred.index
        )

        # Raw model output is now on the disposals scale directly — the
        # earlier np.log1p(target) / np.expm1(prediction) round-trip was
        # responsible for severe top-end compression in backtest (max
        # prediction was 28 vs max actual 43). Apply the post-hoc linear
        # calibration learned from OOF predictions to correct residual
        # mean-bias and top-end stretch.
        raw_pred = self.models[self.best_name].predict(X_final)
        a = getattr(self, "_calib_slope", 1.0)
        b = getattr(self, "_calib_intercept", 0.0)
        predictions = a * raw_pred + b
        # Clip to a physically reasonable range. Lower=1: a 0-disposal
        # prediction implies DNP, which the upstream pipeline doesn't model,
        # so floor at 1 to avoid pathologically low values. Upper=55:
        # historical max single-game disposals in the era covered is ~50,
        # so 55 is a safe ceiling that never truncates a realistic forecast.
        predictions = np.clip(predictions, 1.0, 55.0)

        return current_season_data[['player', 'team', 'round', 'date', 'round_number']].assign(predicted_disposals=predictions)

    def get_next_round(self, df: pd.DataFrame, target_year) -> int:
        """
        Return the next round to predict for the specified target year.

        Strategy: take the largest round_number among target-year rows that
        already have a recorded ``disposals`` value (i.e. games actually
        played, not future fixtures or missed games), add 1. Falls back to
        any round_number if disposals is unavailable. If nothing matches,
        returns 1.
        """
        if 'year' not in df.columns:
            raise ValueError("'year' column is missing")
        if 'round_number' not in df.columns:
            raise ValueError("'round_number' column is missing")

        df_target = df[df['year'] == target_year]

        if 'disposals' in df_target.columns:
            played = df_target[df_target['disposals'].notnull()]
            round_numbers = pd.to_numeric(played['round_number'], errors='coerce').dropna()
            if round_numbers.empty:
                round_numbers = pd.to_numeric(df_target['round_number'], errors='coerce').dropna()
        else:
            round_numbers = pd.to_numeric(df_target['round_number'], errors='coerce').dropna()

        if round_numbers.empty:
            return 1

        max_round = int(round_numbers.max())
        return max_round + 1

    def run(self):
        """Execute the prediction pipeline."""
        print("🚀 Starting AFL Disposal Prediction Pipeline...")
        try:
            df = self.load_and_prepare_data()
            if df.empty or df is None:
                raise ValueError("No valid data to process")
            
            # Early sanity check for next round.
            # We only need a round_number column to compute the next round —
            # there is no need to run the full feature-engineering pipeline
            # (which is what the old code path triggered as a side effect).
            if 'round_number' in df.columns:
                next_round_preview = self.get_next_round(df, self.target_year)
            else:
                df_for_round = df.assign(
                    round_number=df['round'].apply(extract_round_number)
                )
                next_round_preview = self.get_next_round(df_for_round, self.target_year)
            print(f"🔎 Earliest sanity-check: next_round will be {next_round_preview}")
            
            X, y = self.prepare_features_and_target(df)
            if X is None or y is None:
                raise ValueError("No historical data for training")
            scores = self.train_models(X, y)
            self.best_name = self.select_best_model(scores)
            self.models[self.best_name].fit(X, y)
            print(f"\n📊 Model Performance Summary:")
            for name, score in scores.items():
                print(f" {name}: MSE = {-score:.4f}")
            # Fit a post-hoc linear calibration of predicted_disposals on
            # actual_disposals using out-of-fold predictions from the chosen
            # model. Backtest evidence (R1-R8 2026): raw model output had a
            # compressed range (max 28 vs max actual 43) and -1.32 mean bias.
            # A single (slope, intercept) on OOF predictions corrects both
            # the central bias and the top-end compression without touching
            # the model itself, and applies cleanly at predict time.
            self._fit_calibration(X, y)
            all_predictions_dfs = []
            birth_year_threshold = self.target_year - 40
            print(f"\n🔮 Generating predictions for {self.target_year}...")
            # Reuse already-loaded per-player DataFrames from the cache
            # populated by load_and_prepare_data(); fall back to disk only
            # if the cache is empty (e.g. someone called run() in pieces).
            if self._player_cache:
                cache_items = list(self._player_cache.items())
            else:
                cache_items = []
                for filepath in self.data_dir.glob('*.csv'):
                    if "_performance_details" not in filepath.name:
                        continue
                    cache_items.append((filepath, None))

            for filepath, cached_df in cache_items:
                try:
                    player_name, dob = extract_dob_and_name(filepath)
                    if pd.isna(dob) or dob.year <= birth_year_threshold:
                        continue
                    player_df = cached_df if cached_df is not None else self.load_player(filepath)
                    if 'year' not in player_df.columns or not (player_df['year'] == self.target_year).any():
                        continue
                    player_predictions = self.predict_current_season_disposals(player_df)
                    if not player_predictions.empty:
                        all_predictions_dfs.append(player_predictions)
                        print(f"✅ Generated predictions for {player_name}")
                except Exception as e:
                    print(f"❌ Prediction error for {filepath.name}: {e}")
                    continue
            # Release the per-player cache once we're done with the
            # prediction loop — these DataFrames can occupy a lot of
            # memory at scale and aren't needed for the post-processing
            # / model summary that follows.
            cache_items = None
            self._player_cache = {}
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
                            # Round predicted_disposals to whole numbers ONLY for
                            # the user-facing CSV. Internal calculations above
                            # (rolling averages, model output, calibration,
                            # ranking sort) all retain float precision.
                            output_df = next_game_predictions[output_cols].copy()
                            output_df['predicted_disposals'] = (
                                np.round(output_df['predicted_disposals']).astype(int)
                            )
                            output_df.to_csv(csv_path, index=False)
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
            print("\n📈 Machine Learning State Summary:")
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

def _parse_cli_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for the prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="AFL SuperCoach disposal prediction pipeline.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help=(
            "Target season year to predict for (e.g. 2026). "
            "If omitted, auto-detects from the latest year present in "
            "./data/player_data/."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/player_data/",
        help="Directory containing *_performance_details.csv files.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Across-season rolling window size (games).",
    )
    parser.add_argument(
        "--within-season-window",
        type=int,
        default=3,
        help="Within-season rolling window size (games).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print verbose feature / model diagnostics.",
    )
    parser.add_argument(
        "--include-age-experience",
        action="store_true",
        help=(
            "Opt-in: add the Task S7 leak-proof features "
            "(player_age_at_match, career_games_to_date). Off by default so "
            "production predictions are unchanged until a backtest validates "
            "them (see docs/experiment-log.md)."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    args = _parse_cli_args(sys.argv[1:])
    try:
        predictor = AFLDisposalPredictor(
            data_dir=args.data_dir,
            target_year=args.year,
            rolling_window=args.rolling_window,
            within_season_window=args.within_season_window,
            debug_mode=args.debug,
            include_age_experience=args.include_age_experience,
        )
        predictor.run()
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)