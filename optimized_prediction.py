import sys
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import warnings
import joblib
from functools import lru_cache
import hashlib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Optimized data types for memory efficiency
DTYPES = {
    'player': 'category',
    'year': 'int16',
    'round': 'category', 
    'date': 'datetime64[ns]',
    'team': 'category',
    'opponent': 'category',
    'venue': 'category',
    'disposals': 'float32',  # Changed to float32 for NaN support
    'kicks': 'float32',
    'handballs': 'float32',
    'goals': 'float32',
    'behinds': 'float32',
    'hitouts': 'float32',
    'tackles': 'float32',
    'rebound_50s': 'float32',
    'inside_50s': 'float32',
    'clearances': 'float32',
    'clangers': 'float32',
    'frees_for': 'float32',
    'frees_against': 'float32',
    'brownlow_votes': 'int8',
    'contested_possessions': 'float32',
    'uncontested_possessions': 'float32',
    'contested_marks': 'float32',
    'marks_inside_50': 'float32',
    'one_percenters': 'float32',
    'bounces': 'float32',
    'goal_assists': 'float32',
    'percentage_time_played': 'float32',
    'cba_percent': 'float32'
}

NA_VALUES = ['NA', 'N/A', '', 'nan']
RENAMES = {
    'hit_outs': 'hitouts',
    'free_kicks_for': 'frees_for',
    'free_kicks_against': 'frees_against',
}
REQUIRED_COLUMNS = ['player', 'year', 'round', 'disposals', 'kicks', 'handballs', 'tackles', 'clearances', 'inside_50s']

class OptimizedAFLPredictor:
    def __init__(self, data_dir: str, target_year: int = 2025, rolling_window: int = 5, 
                 within_season_window: int = 3, debug_mode: bool = False, use_cache: bool = True):
        """Initialize optimized predictor with caching and performance improvements."""
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
        self.target_year = target_year
        self.rolling_window = rolling_window
        self.within_season_window = within_season_window
        self.debug_mode = debug_mode
        self.use_cache = use_cache
        
        # Cache directory for performance
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Simplified model approach - use only best performing model
        self.model = None
        self.preprocessor = None
        self.feature_columns = []
        self.training_feature_columns = None
        
        # Core features for rolling calculations
        self.base_rolling_features = ['disposals', 'kicks', 'handballs', 'tackles', 'clearances', 'inside_50s']
        self.extra_features = ['cba_percent', 'percentage_time_played']

    def _get_cache_key(self, data_hash: str, stage: str) -> str:
        """Generate cache key for different processing stages."""
        return f"{stage}_{data_hash}_{self.target_year}_{self.rolling_window}.pkl"

    def _get_data_hash(self, filepath: Path) -> str:
        """Generate hash of file for cache invalidation."""
        try:
            stat = filepath.stat()
            return hashlib.md5(f"{filepath}_{stat.st_mtime}_{stat.st_size}".encode()).hexdigest()[:16]
        except:
            return "unknown"

    def _load_from_cache(self, cache_key: str):
        """Load data from cache if available."""
        if not self.use_cache:
            return None
        cache_file = self.cache_dir / cache_key
        if cache_file.exists():
            try:
                return joblib.load(cache_file)
            except:
                pass
        return None

    def _save_to_cache(self, data, cache_key: str):
        """Save data to cache."""
        if not self.use_cache:
            return
        cache_file = self.cache_dir / cache_key
        try:
            joblib.dump(data, cache_file, compress=3)
        except:
            pass

    @lru_cache(maxsize=128)
    def _extract_round_number(self, round_str):
        """Cached round number extraction."""
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

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns and validate required columns."""
        df = df.rename(columns=RENAMES)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df

    def _extract_dob_and_name(self, filepath: Path):
        """Extract player name and DOB from filename."""
        parts = filepath.stem.split('_')
        if len(parts) < 3 or parts[-2] != 'performance' or parts[-1] != 'details':
            return filepath.stem, pd.NaT
        try:
            dob_str = parts[-3]
            player_name = ' '.join(parts[:-3]).title()
            dob = pd.to_datetime(dob_str, format='%d%m%Y', errors='coerce')
            return player_name, dob
        except:
            return filepath.stem, pd.NaT

    def load_player_optimized(self, filepath: Path) -> pd.DataFrame:
        """Optimized player data loading with caching."""
        data_hash = self._get_data_hash(filepath)
        cache_key = self._get_cache_key(data_hash, "player_data")
        
        # Try cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # Efficient CSV reading with optimized dtypes
            df = pd.read_csv(filepath, na_values=NA_VALUES, dtype={col: 'str' for col in ['player', 'team', 'opponent', 'venue', 'round']})
            
            player_name, dob = self._extract_dob_and_name(filepath)
            df['player'] = player_name
            df = self._clean_columns(df)
            
            # Convert to optimized dtypes
            for col, dtype in DTYPES.items():
                if col in df.columns:
                    if dtype in ['int8', 'int16', 'int32', 'int64']:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')  # Use float32 for NaN support
                    elif dtype == 'category':
                        df[col] = df[col].astype('category')
                    elif dtype == 'datetime64[ns]':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
            
            # Cache processed data
            self._save_to_cache(df, cache_key)
            return df
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error loading {filepath.name}: {e}")
            return pd.DataFrame()

    def load_and_prepare_data_optimized(self) -> pd.DataFrame:
        """Optimized data loading with parallel processing and caching."""
        all_files_hash = hashlib.md5(str(sorted(self.data_dir.glob('*_performance_details.csv'))).encode()).hexdigest()[:16]
        cache_key = self._get_cache_key(all_files_hash, "combined_data")
        
        # Try cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        all_dfs = []
        birth_year_threshold = self.target_year - 40
        
        for filepath in self.data_dir.glob('*_performance_details.csv'):
            try:
                player_name, dob = self._extract_dob_and_name(filepath)
                if pd.isna(dob) or dob.year <= birth_year_threshold:
                    continue
                    
                df = self.load_player_optimized(filepath)
                if not df.empty:
                    all_dfs.append(df)
                    
            except Exception as e:
                if self.debug_mode:
                    print(f"Error processing {filepath.name}: {e}")
                continue

        if not all_dfs:
            return pd.DataFrame()
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Cache combined data
        self._save_to_cache(combined_df, cache_key)
        return combined_df

    def _engineer_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized feature engineering for better performance."""
        # Sort once
        df = df.sort_values(['player', 'year', 'round'])
        df['round_number'] = df['round'].apply(self._extract_round_number)
        
        # Vectorized rolling calculations
        rolling_cols = [col for col in self.base_rolling_features if col in df.columns]
        
        # Group operations are expensive, so minimize them
        grouped = df.groupby('player', observed=False)
        grouped_year = df.groupby(['player', 'year'], observed=False)
        
        # Calculate all rolling features at once
        for col in rolling_cols:
            # Across-season rolling average
            df[f'across_season_rolling_avg_{col}_{self.rolling_window}'] = (
                grouped[col].transform(lambda x: x.rolling(window=self.rolling_window, min_periods=1).mean().shift(1))
            )
            
            # Within-season rolling average  
            df[f'within_season_rolling_avg_{col}_{self.within_season_window}'] = (
                grouped_year[col].transform(lambda x: x.rolling(window=self.within_season_window, min_periods=1).mean().shift(1))
            )
            
            # Season-to-date mean
            df[f'season_to_date_mean_{col}'] = (
                grouped_year[col].transform(lambda x: x.expanding().mean().shift(1))
            )
            
            # Recent form (exponential weighted)
            df[f'recent_form_{col}'] = (
                grouped[col].transform(lambda x: x.ewm(span=3, adjust=False).mean().shift(1))
            )
        
        # Calculate days since last game
        df['days_since_last_game'] = grouped['date'].transform(lambda x: x.diff().dt.days.fillna(0))
        
        # Build feature column list
        across_season_features = [f'across_season_rolling_avg_{col}_{self.rolling_window}' for col in rolling_cols]
        within_season_features = [f'within_season_rolling_avg_{col}_{self.within_season_window}' for col in rolling_cols]
        season_to_date_features = [f'season_to_date_mean_{col}' for col in rolling_cols]
        recent_form_features = [f'recent_form_{col}' for col in rolling_cols]
        extra_feats = [feat for feat in self.extra_features if feat in df.columns]
        
        self.feature_columns = (across_season_features + within_season_features + 
                              season_to_date_features + recent_form_features + 
                              ['round_number', 'days_since_last_game'] + extra_feats)
        
        # Remove rows with missing target or key features
        required_features = across_season_features + within_season_features + season_to_date_features + recent_form_features
        df = df.dropna(subset=required_features + ['disposals'])
        
        return df

    def train_optimized_model(self, X, y):
        """Train single optimized model instead of multiple models."""
        if X is None or y is None or X.empty or y.empty:
            print("No data available for training")
            return None

        # Use only HistGradientBoostingRegressor with optimized parameters
        # Reduced hyperparameter search for speed
        best_params = {
            'max_depth': 5,
            'learning_rate': 0.05,
            'max_leaf_nodes': 50,
            'min_samples_leaf': 20,
            'loss': 'poisson'
        }
        
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', HistGradientBoostingRegressor(**best_params, random_state=42))
        ])
        
        # Train the model
        self.model.fit(X, y)
        
        # Quick validation
        groups = X.index  # Simplified grouping
        cv_score = cross_val_score(self.model, X, y, cv=3, scoring='neg_mean_squared_error').mean()
        
        if self.debug_mode:
            print(f"Optimized model CV score: {-cv_score:.4f}")
        
        return -cv_score

    def predict_disposals(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Optimized prediction pipeline."""
        if player_data.empty or self.model is None:
            return pd.DataFrame()
            
        # Engineer features for prediction
        engineered_data = self._engineer_features_vectorized(player_data)
        current_season = engineered_data[engineered_data['year'] == self.target_year].copy()
        
        if current_season.empty:
            return pd.DataFrame()
        
        # Prepare features
        X_pred = current_season[self.feature_columns].copy()
        X_pred['missing_count'] = X_pred.isna().sum(axis=1)
        
        # Handle missing values
        X_pred = X_pred.fillna(X_pred.median())
        
        # Make predictions
        predictions = np.expm1(self.model.predict(X_pred))
        predictions = np.clip(predictions, 0, 45)  # Reasonable bounds
        
        return current_season[['player', 'team', 'round', 'date']].assign(predicted_disposals=predictions)

    def run_optimized(self):
        """Execute optimized prediction pipeline."""
        print("🚀 Starting Optimized AFL Disposal Prediction...")
        
        # Load data
        df = self.load_and_prepare_data_optimized()
        if df.empty:
            print("❌ No valid data found")
            return
            
        # Prepare training data
        historical_data = df[df['year'] < self.target_year].copy()
        historical_data = historical_data[historical_data['disposals'].notnull()]
        
        if historical_data.empty:
            print("❌ No historical training data")
            return
            
        # Engineer features
        print("🔧 Engineering features...")
        engineered_df = self._engineer_features_vectorized(historical_data)
        
        # Prepare features and target
        X = engineered_df[self.feature_columns].copy()
        X['missing_count'] = X.isna().sum(axis=1)
        X = X.fillna(X.median())  # Simple imputation
        y = np.log1p(engineered_df['disposals'])
        
        self.training_feature_columns = X.columns.tolist()
        
        # Train model
        print("🏋️ Training optimized model...")
        score = self.train_optimized_model(X, y)
        print(f"✅ Model trained with CV score: {score:.4f}")
        
        # Generate predictions
        print(f"🔮 Generating {self.target_year} predictions...")
        all_predictions = []
        
        for filepath in self.data_dir.glob('*_performance_details.csv'):
            player_data = self.load_player_optimized(filepath)
            if not player_data.empty:
                predictions = self.predict_disposals(player_data)
                if not predictions.empty:
                    all_predictions.append(predictions)
        
        if all_predictions:
            final_predictions = pd.concat(all_predictions, ignore_index=True)
            output_file = f"optimized_predictions_{self.target_year}.csv"
            final_predictions.to_csv(output_file, index=False)
            print(f"💾 Saved {len(final_predictions)} predictions to {output_file}")
        else:
            print("⚠️ No predictions generated")

if __name__ == "__main__":
    predictor = OptimizedAFLPredictor("data/player_data", target_year=2025, debug_mode=True)
    predictor.run_optimized()