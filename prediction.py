import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

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
        self.rolling_window = rolling_window  # Across-season rolling window
        self.within_season_window = within_season_window  # Within-season rolling window
        self.debug_mode = debug_mode
        self.models = {
            'hgb_poisson': make_pipeline(
                StandardScaler(),
                HistGradientBoostingRegressor(
                    loss='poisson',
                    max_depth=4,
                    learning_rate=0.03,
                    max_leaf_nodes=63,
                    random_state=42
                )
            )
        }
        # Core feature set for rolling averages
        self.base_rolling_features = [
            'disposals', 'kicks', 'handballs', 'tackles', 'clearances', 'inside_50s'
        ]
        self.extra_features = ['cba_percent', 'percentage_time_played']
        self.feature_columns = []
        self.best_name = None
        self.training_feature_columns = None

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
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
        except Exception as e:
            print(f"❌ Error in data type conversion for {filepath.name}: {e}")
            return pd.DataFrame()
        
        dummy_cols = [col for col in ['venue', 'opponent'] if col in df.columns]
        if dummy_cols:
            df = pd.get_dummies(df, columns=dummy_cols, drop_first=True)
        return df

    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure data has required columns and no nulls in key fields."""
        required_columns = {'year', 'round', 'disposals'}
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
            return pd.DataFrame()
        original_rows = len(df)
        df = df[df['year'].notnull() & df['disposals'].notnull()]
        cleaned_rows = len(df)
        if original_rows > cleaned_rows:
            print(f"⚠️ Dropped {original_rows - cleaned_rows} rows due to missing 'year' or 'disposals'")
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
            return df.groupby(group_by)[col].transform(
                lambda s: s.rolling(window=window, min_periods=1).mean().shift(1)
            )
        except Exception as e:
            print(f"❌ Rolling average failed for {col} with group_by {group_by}: {e}")
            return pd.Series(np.nan, index=df.index)

    def _add_expanding_mean(self, df, col, group_by):
        """Calculate season-to-date expanding mean with specified grouping."""
        try:
            return df.groupby(group_by)[col].transform(
                lambda s: s.expanding().mean().shift(1)
            )
        except Exception as e:
            print(f"❌ Expanding mean failed for {col} with group_by {group_by}: {e}")
            return pd.Series(np.nan, index=df.index)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for training, including rolling averages, season-to-date means, and round number."""
        df['round_number'] = df['round'].apply(extract_round_number)
        df = df.sort_values(['player', 'year', 'round_number'])
        
        rolling_cols_available = [col for col in self.base_rolling_features if col in df.columns]
        
        # Across-season rolling averages
        for col in rolling_cols_available:
            df[f'across_season_rolling_avg_{col}_{self.rolling_window}'] = self._add_rolling(df, col, window=self.rolling_window, group_by=['player'])
        
        # Within-season rolling averages
        for col in rolling_cols_available:
            df[f'within_season_rolling_avg_{col}_{self.within_season_window}'] = self._add_rolling(df, col, window=self.within_season_window, group_by=['player', 'year'])
        
        # Season-to-date mean
        for col in rolling_cols_available:
            df[f'season_to_date_mean_{col}'] = self._add_expanding_mean(df, col, group_by=['player', 'year'])
        
        # Define feature columns
        across_season_features = [f'across_season_rolling_avg_{col}_{self.rolling_window}' for col in rolling_cols_available]
        within_season_features = [f'within_season_rolling_avg_{col}_{self.within_season_window}' for col in rolling_cols_available]
        season_to_date_features = [f'season_to_date_mean_{col}' for col in rolling_cols_available]
        extra_feats = [feat for feat in self.extra_features if feat in df.columns]
        dummy_cols = [c for c in df.columns if c.startswith(('venue_', 'opponent_'))]
        self.feature_columns = across_season_features + within_season_features + season_to_date_features + ['round_number'] + extra_feats + dummy_cols
        
        for col in self.feature_columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"⚠️ {nan_count} NaN values in feature '{col}'")
        
        return df.dropna(subset=across_season_features + within_season_features + season_to_date_features + ['disposals'])

    def _engineer_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for prediction, matching training features."""
        df['round_number'] = df['round'].apply(extract_round_number)
        df = df.sort_values(['player', 'year', 'round_number'])
        
        rolling_cols_available = [col for col in self.base_rolling_features if col in df.columns]
        
        for col in rolling_cols_available:
            df[f'across_season_rolling_avg_{col}_{self.rolling_window}'] = self._add_rolling(df, col, window=self.rolling_window, group_by=['player'])
            df[f'within_season_rolling_avg_{col}_{self.within_season_window}'] = self._add_rolling(df, col, window=self.within_season_window, group_by=['player', 'year'])
            df[f'season_to_date_mean_{col}'] = self._add_expanding_mean(df, col, group_by=['player', 'year'])
        
        return df

    def prepare_features_and_target(self, df: pd.DataFrame) -> tuple:
        """Prepare features (X) and target (y) for training."""
        historical_data = df[df['year'] < self.target_year]
        if historical_data.empty:
            print("⚠️ No historical data available")
            return None, None
        engineered_df = self._engineer_features(historical_data)
        X = engineered_df[self.feature_columns]
        y = engineered_df['disposals']
        self.training_feature_columns = self.feature_columns.copy()
        X = X.fillna(-1)  # Impute missing values
        print(f"Target 'disposals' statistics:")
        print(y.describe())
        if self.debug_mode:
            print("Training feature columns:", self.feature_columns)
            print("Sample of X (features):")
            print(X.head())
            print("Sample of y (disposals):")
            print(y.head())
        return X, y

    def train_models(self, X, y) -> dict:
        """Train models, evaluate with cross-validation, and compare to baseline."""
        scores = {}
        if X is None or y is None or X.empty or y.empty:
            print("⚠️ No data to train models")
            return scores
        for name, model in self.models.items():
            try:
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                scores[name] = cv_scores.mean()
                if self.debug_mode:
                    print(f"Debug: {name} CV scores: {cv_scores}")
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                scores[name] = float('-inf')
        across_season_rolling_col = f'across_season_rolling_avg_disposals_{self.rolling_window}'
        if across_season_rolling_col in X.columns:
            baseline_mse = np.mean((X[across_season_rolling_col] - y) ** 2)
            print(f"Baseline MSE (across-season rolling avg): {baseline_mse:.4f}")
        return scores

    def select_best_model(self, scores: dict) -> str:
        """Select the best model based on CV scores."""
        if not scores:
            print("⚠️ No models trained")
            return 'hgb_poisson'
        return max(scores, key=scores.get)

    def predict_current_season_disposals(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Predict disposals for the target year."""
        if player_data.empty:
            print("⚠️ Empty player data")
            return pd.DataFrame()
        player_data = self.validate_and_clean_data(player_data)
        player_data = self._engineer_features_for_prediction(player_data)
        current_season_data = player_data[player_data['year'] == self.target_year]
        if current_season_data.empty:
            player_name = player_data['player'].iloc[0] if 'player' in player_data.columns else 'Unknown'
            print(f"⚠️ No {self.target_year} data for {player_name}")
            return pd.DataFrame()
        dummy_cols = [col for col in ['venue', 'opponent'] if col in current_season_data.columns]
        if dummy_cols:
            current_season_data = pd.get_dummies(current_season_data, columns=dummy_cols, drop_first=True)
        X_pred = current_season_data.reindex(columns=self.training_feature_columns, fill_value=0)
        X_pred = X_pred.fillna(-1)  # Impute missing values
        predictions = self.models[self.best_name].predict(X_pred)
        predictions = np.clip(predictions, 0, 45)
        return current_season_data[['player', 'round', 'date']].assign(predicted_disposals=predictions)

    def get_round_number(self, round_str):
        """Extract numerical round from string for final output."""
        return extract_round_number(round_str)

    def run(self) -> None:
        """Execute the prediction pipeline."""
        print("🚀 Starting AFL Disposal Prediction Pipeline...")
        try:
            df = self.load_and_prepare_data()
            if df.empty:
                raise ValueError("No valid data to process")
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
                all_predictions['round_number'] = all_predictions['round'].apply(self.get_round_number)
                valid_predictions = all_predictions[all_predictions['round_number'].notnull()]
                if not valid_predictions.empty:
                    valid_predictions = (
                        valid_predictions
                        .assign(date=pd.to_datetime(valid_predictions['date'], errors='coerce'))
                        .dropna(subset=['date'])
                        .copy()
                    )
                    next_game_predictions = (
                        valid_predictions
                        .sort_values('date')
                        .groupby('player')
                        .head(1)
                        .sort_values('predicted_disposals', ascending=False)
                    )
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    prediction_dir = Path("./data/prediction")
                    prediction_dir.mkdir(parents=True, exist_ok=True)
                    csv_path = prediction_dir / f"next_game_predictions_{timestamp}.csv"
                    next_game_predictions[['player', 'predicted_disposals']].to_csv(csv_path, index=False)
                    print(f"📄 Next game predictions saved to {csv_path}")
                else:
                    print("⚠️ No valid round predictions generated")
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