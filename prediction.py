import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
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
    'percentage_time_played': 'float32'
}

NA_VALUES = ['NA', 'N/A', '', 'nan']

# Column renaming dictionary to standardize names
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
    """Extract player name and date of birth from filename in DDMMYYYY format."""
    parts = filepath.stem.split('_')
    if len(parts) < 3 or parts[-2] != 'performance' or parts[-1] != 'details':
        print(f"⚠️ Invalid filename format: {filepath.name}. Expected 'player_name_DDMMYYYY_performance_details.csv'")
        return filepath.stem, pd.NaT
    
    try:
        dob_str = parts[-3]  # e.g., '09111888'
        player_name = ' '.join(parts[:-3]).title()  # e.g., 'Abbott Clarrie'
        dob = pd.to_datetime(dob_str, format='%d%m%Y', errors='coerce')  # Parse DDMMYYYY
        if pd.isna(dob):
            print(f"⚠️ Invalid DOB format in {filepath.name}: {dob_str}")
        return player_name, dob
    except Exception as e:
        print(f"❌ Error parsing filename {filepath.name}: {e}")
        return filepath.stem, pd.NaT

class AFLDisposalPredictor:
    def __init__(self, data_dir: str, target_year: int = 2025, debug_mode: bool = False):
        """Initialize the AFL Disposal Predictor with path validation."""
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {data_dir}")
        
        self.target_year = target_year
        self.debug_mode = debug_mode
        self.models = {
            'hgb_poisson': make_pipeline(
                StandardScaler(),
                HistGradientBoostingRegressor(loss='poisson', max_depth=3, learning_rate=0.05, random_state=42)
            )
        }
        self.feature_columns = [
            'rolling_avg_disposals_5', 'rolling_avg_kicks_5', 'rolling_avg_handballs_5',
            'rolling_avg_tackles_5', 'rolling_avg_clearances_5', 'rolling_avg_inside_50s_5'
        ]
        self.best_name = None

    def load_player(self, filepath: Path) -> pd.DataFrame:
        """Load and preprocess player data from a CSV file with robust error handling."""
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
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df.dropna(subset=['year'])
            df['year'] = df['year'].astype('Int16')
            
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
        
        return df

    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the data."""
        required_columns = {'year', 'round', 'disposals'}
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
            return pd.DataFrame()
        df = df[df['year'].notnull() & df['disposals'].notnull()]
        return df

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and concatenate data from all player files."""
        all_dfs = []
        birth_year_threshold = self.target_year - 40
        for filepath in self.data_dir.glob('*.csv'):
            if "_performance_details" not in filepath.name:
                continue
            try:
                player_name, dob = extract_dob_and_name(filepath)
                if pd.isna(dob):
                    print(f"⚠️ Skipping {filepath.name} due to invalid DOB")
                    continue
                if dob.year <= birth_year_threshold:
                    print(f"⚠️ Skipping {player_name} (DOB {dob.year} <= {birth_year_threshold})")
                    continue
                df = self.load_player(filepath)
                if not df.empty:
                    all_dfs.append(df)
                else:
                    print(f"⚠️ No valid data in {filepath.name}")
            except Exception as e:
                print(f"❌ Error processing file {filepath.name}: {e}")
                continue
        if not all_dfs:
            print("⚠️ No valid data loaded from any files")
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)

    def _add_rolling(self, df, col):
        """Calculate rolling averages, resetting each season with error handling."""
        try:
            return df.groupby('year')[col].transform(
                lambda s: s.ewm(span=5, adjust=False, min_periods=1).mean().shift(1)
            )
        except Exception as e:
            print(f"❌ Rolling average calculation failed for {col}: {e}")
            return pd.Series(np.nan, index=df.index)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for training, resetting rolling averages per season."""
        df = df.sort_values(['year', 'round'])
        for col in ['disposals', 'kicks', 'handballs', 'tackles', 'clearances', 'inside_50s']:
            df[f'rolling_avg_{col}_5'] = self._add_rolling(df, col)
        return df.dropna(subset=self.feature_columns + ['disposals'])

    def _engineer_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for prediction, resetting rolling averages per season."""
        df = df.sort_values(['year', 'round'])
        for col in ['disposals', 'kicks', 'handballs', 'tackles', 'clearances', 'inside_50s']:
            df[f'rolling_avg_{col}_5'] = self._add_rolling(df, col)
        return df

    def prepare_features_and_target(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target for training."""
        historical_data = df[df['year'] < self.target_year]
        if historical_data.empty:
            print("⚠️ No historical data available")
            return None, None
        engineered_df = self._engineer_features(historical_data)
        X = engineered_df[self.feature_columns]
        y = engineered_df['disposals']
        return X, y

    def train_models(self, X, y) -> dict:
        """Train models and evaluate with cross-validation."""
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
        return scores

    def select_best_model(self, scores: dict) -> str:
        """Select the best model based on cross-validation scores."""
        if not scores:
            print("⚠️ No models trained")
            return 'hgb_poisson'
        best_name = max(scores, key=scores.get)
        return best_name

    def predict_current_season_disposals(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Predict disposals for the target year."""
        if player_data.empty:
            print("⚠️ Empty player data provided")
            return pd.DataFrame()
        
        player_data = self.validate_and_clean_data(player_data)
        player_data = self._engineer_features_for_prediction(player_data)
        current_season_data = player_data[player_data['year'] == self.target_year]
        
        if current_season_data.empty:
            player_name = player_data['player'].iloc[0] if 'player' in player_data.columns else 'Unknown'
            print(f"⚠️ No {self.target_year} data for {player_name}")
            return pd.DataFrame()
        
        X_pred = current_season_data[self.feature_columns].fillna(0)
        predictions = self.models[self.best_name].predict(X_pred)
        predictions = np.clip(predictions, 0, 40)
        predictions_df = current_season_data[['player', 'round', 'date']].assign(predicted_disposals=predictions)
        return predictions_df

    def get_round_number(self, round_str):
        """Extract the numerical part from the round string."""
        round_str = str(round_str).strip()
        try:
            if round_str.isdigit():
                return int(round_str)
            parts = round_str.split()
            if len(parts) >= 2 and parts[0] == 'Round':
                return int(parts[1])
            else:
                print(f"⚠️ Unexpected round format: {round_str}")
                return None
        except (IndexError, ValueError):
            print(f"⚠️ Invalid round format or non-integer: {round_str}")
            return None

    def run(self) -> None:
        """Run the prediction pipeline and save next game predictions for each player in 2025."""
        print("🚀 Starting AFL Disposal Prediction Pipeline...")
        try:
            df = self.load_and_prepare_data()
            if df.empty:
                raise ValueError("No valid data available to process")
            X, y = self.prepare_features_and_target(df)
            if X is None or y is None:
                raise ValueError("No historical data available for training")
            
            scores = self.train_models(X, y)
            self.best_name = self.select_best_model(scores)
            self.models[self.best_name].fit(X, y)
            
            print(f"\n📊 Model Performance Summary:")
            for name, score in scores.items():
                print(f" {'✅' if score > -1.0 else '❌'} {name}: {score:.4f}")
            
            all_predictions_dfs = []
            birth_year_threshold = self.target_year - 40
            print(f"\n🔮 Generating predictions for {self.target_year}...")
            
            for filepath in self.data_dir.glob('*.csv'):
                if "_performance_details" not in filepath.name:
                    continue
                try:
                    player_name, dob = extract_dob_and_name(filepath)
                    if pd.isna(dob):
                        continue
                    if dob.year <= birth_year_threshold:
                        continue
                    player_df = self.load_player(filepath)
                    if 'year' not in player_df.columns or not (player_df['year'] == self.target_year).any():
                        print(f"⚠️ No {self.target_year} data for {player_name}")
                        continue
                    player_predictions = self.predict_current_season_disposals(player_df)
                    if len(player_predictions) > 0:
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
                    
                    # Select the earliest game per player in 2025 (no date restriction)
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
                    print("⚠️ No valid round predictions generated.")
            else:
                print("⚠️ No predictions generated.")
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
        predictor = AFLDisposalPredictor(data_dir, target_year=2025, debug_mode=debug_mode)
        predictor.run()
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)