import sys
import numpy as np
import pandas as pd
import cudf
import cupy as cp
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score

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

def extract_dob_and_name(filepath):
    """Extract player name and date of birth from filename.

    Args:
        filepath (Path): Path to the player's CSV file.

    Returns:
        tuple: (player_name, dob) where dob is a datetime object.
    """
    parts = filepath.stem.split('_')
    player_name = ' '.join(parts[:-2]).title()
    dob_str = '_'.join(parts[-2:])
    dob = pd.to_datetime(dob_str, format='%Y_%m_%d', errors='coerce')
    return player_name, dob

class AFLDisposalPredictor:
    def __init__(self, data_dir: str, target_year: int = 2025, debug_mode: bool = False):
        """Initialize the AFL Disposal Predictor.

        Args:
            data_dir (str): Directory containing player data CSV files.
            target_year (int): Year for which to make predictions.
            debug_mode (bool): Enable debug output.
        """
        self.data_dir = Path(data_dir)
        self.target_year = target_year
        self.debug_mode = debug_mode
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        self.scaler = StandardScaler()
        self.feature_columns = [
            'rolling_avg_disposals_5', 'rolling_avg_kicks_5', 'rolling_avg_handballs_5',
            'rolling_avg_tackles_5', 'rolling_avg_clearances_5', 'rolling_avg_inside_50s_5'
        ]
        self.best_name = None

    def load_player(self, filepath: Path) -> pd.DataFrame:
        """Load and preprocess player data from a CSV file.

        Args:
            filepath (Path): Path to the player's CSV file.

        Returns:
            pd.DataFrame: Preprocessed player data.
        """
        df = pd.read_csv(filepath, na_values=NA_VALUES, dtype=str)
        int_columns = [col for col, dtype in DTYPES.items() if 'Int' in dtype]
        for col in int_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: np.nan if pd.isna(x) else int(''.join(filter(str.isdigit, str(x)))) if ''.join(filter(str.isdigit, str(x))) else np.nan)
        for col, dtype in DTYPES.items():
            if col in df.columns:
                if dtype in ['Int8', 'Int16', 'Int32', 'Int64']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
                elif dtype == 'category':
                    df[col] = df[col].astype('category')
                elif dtype == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
        player_name, _ = extract_dob_and_name(filepath)
        df['player'] = player_name
        return df

    def validate_and_clean_data(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """Validate and clean the data.

        Args:
            gdf (cudf.DataFrame): DataFrame to clean.

        Returns:
            cudf.DataFrame: Cleaned DataFrame.
        """
        required_columns = {'year', 'round', 'disposals'}
        missing_cols = required_columns - set(gdf.columns)
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
            return cudf.DataFrame()
        gdf = gdf[gdf['year'].notnull() & gdf['disposals'].notnull()]
        return gdf

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and concatenate data from all player files.

        Returns:
            pd.DataFrame: Combined DataFrame of all player data.
        """
        all_dfs = []
        birth_year_threshold = self.target_year - 40
        for filepath in self.data_dir.glob('*.csv'):
            if "_performance_details" not in filepath.name:
                continue
            player_name, dob = extract_dob_and_name(filepath)
            if dob.year <= birth_year_threshold:
                continue
            df = self.load_player(filepath)
            if not df.empty:
                all_dfs.append(df)
        if not all_dfs:
            print("⚠️ No valid data loaded")
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for training.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        df = df.sort_values(['year', 'round'])
        df['rolling_avg_disposals_5'] = df['disposals'].rolling(window=5, min_periods=1).mean().shift(1)
        df['rolling_avg_kicks_5'] = df['kicks'].rolling(window=5, min_periods=1).mean().shift(1)
        df['rolling_avg_handballs_5'] = df['handballs'].rolling(window=5, min_periods=1).mean().shift(1)
        df['rolling_avg_tackles_5'] = df['tackles'].rolling(window=5, min_periods=1).mean().shift(1)
        df['rolling_avg_clearances_5'] = df['clearances'].rolling(window=5, min_periods=1).mean().shift(1)
        df['rolling_avg_inside_50s_5'] = df['inside_50s'].rolling(window=5, min_periods=1).mean().shift(1)
        return df.dropna(subset=self.feature_columns + ['disposals'])

    def _engineer_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for prediction.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        df = df.sort_values(['year', 'round'])
        df['rolling_avg_disposals_5'] = df['disposals'].rolling(window=5, min_periods=1).mean().shift(1)
        df['rolling_avg_kicks_5'] = df['kicks'].rolling(window=5, min_periods=1).mean().shift(1)
        df['rolling_avg_handballs_5'] = df['handballs'].rolling(window=5, min_periods=1).mean().shift(1)
        df['rolling_avg_tackles_5'] = df['tackles'].rolling(window=5, min_periods=1).mean().shift(1)
        df['rolling_avg_clearances_5'] = df['clearances'].rolling(window=5, min_periods=1).mean().shift(1)
        df['rolling_avg_inside_50s_5'] = df['inside_50s'].rolling(window=5, min_periods=1).mean().shift(1)
        return df

    def prepare_features_and_target(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target for training.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            tuple: (X, y) where X is features and y is target.
        """
        historical_data = df[df['year'] < self.target_year]
        if historical_data.empty:
            print("⚠️ No historical data available")
            return None, None
        engineered_df = self._engineer_features(historical_data)
        X = engineered_df[self.feature_columns].fillna(0)
        y = engineered_df['disposals']
        return X, y

    def train_models(self, X, y) -> dict:
        """Train models and evaluate with cross-validation.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.

        Returns:
            dict: Model names and their cross-validation scores.
        """
        scores = {}
        if X is None or y is None or X.empty or y.empty:
            print("⚠️ No data to train models")
            return scores
        for name, model in self.models.items():
            X_scaled = self.scaler.fit_transform(X) if 'ridge' in name or 'lasso' in name else X
            try:
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
                scores[name] = cv_scores.mean()
                if self.debug_mode:
                    print(f"Debug: {name} CV scores: {cv_scores}")
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                scores[name] = float('-inf')
        return scores

    def select_best_model(self, scores: dict) -> str:
        """Select the best model based on cross-validation scores.

        Args:
            scores (dict): Model scores.

        Returns:
            str: Name of the best model.
        """
        if not scores:
            print("⚠️ No models trained")
            return 'linear'
        best_name = max(scores, key=scores.get)
        return best_name

    def predict_current_season_disposals(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Predict disposals for the target year.

        Args:
            player_data (pd.DataFrame): Player data.

        Returns:
            pd.DataFrame: Predictions with player, round, date, and predicted disposals.
        """
        if player_data.empty:
            print("⚠️ Empty player data provided")
            return pd.DataFrame()
        
        temp_gdf = cudf.from_pandas(player_data)
        temp_gdf = self.validate_and_clean_data(temp_gdf)
        player_data = temp_gdf.to_pandas()
        del temp_gdf
        cp.cuda.runtime.deviceSynchronize()
        
        player_data = self._engineer_features_for_prediction(player_data)
        current_season_data = player_data[player_data['year'] == self.target_year]
        
        if current_season_data.empty:
            player_name = player_data['player'].iloc[0] if 'player' in player_data.columns else 'Unknown'
            print(f"⚠️ No {self.target_year} data for {player_name}")
            return pd.DataFrame()
        
        X_pred = current_season_data[self.feature_columns].fillna(0)
        if self.best_name in ('ridge', 'lasso'):
            X_pred = self.scaler.transform(X_pred)
        
        predictions = self.models[self.best_name].predict(X_pred)
        predictions_df = current_season_data[['player', 'round', 'date']].assign(predicted_disposals=predictions)
        return predictions_df

    def get_round_number(self, round_str):
        """Extract the numerical part from the round string.

        Handles both numeric strings (e.g., '5') and 'Round N' formats.

        Args:
            round_str (str): Round identifier.

        Returns:
            int or None: Round number if valid, else None.
        """
        round_str = round_str.strip()  # Remove leading/trailing spaces
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
        """Run the prediction pipeline and save next round predictions."""
        print("🚀 Starting AFL Disposal Prediction Pipeline...")
        df = self.load_and_prepare_data()
        X, y = self.prepare_features_and_target(df)
        
        scores = self.train_models(X, y)
        self.best_name = self.select_best_model(scores)
        
        # Fit the chosen model on the full training data
        if self.best_name in ('ridge', 'lasso'):
            X_full = self.scaler.fit_transform(X)
        else:
            X_full = X
        self.models[self.best_name].fit(X_full, y)
        
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
            # Clean player names by removing any appended 8-digit date of birth
            all_predictions['player'] = all_predictions['player'].str.replace(r'\s\d{8}$', '', regex=True)
            all_predictions['round_number'] = all_predictions['round'].apply(self.get_round_number)
            valid_predictions = all_predictions[all_predictions['round_number'].notnull()]
            if not valid_predictions.empty:
                next_round_number = valid_predictions['round_number'].min()
                next_round_predictions = valid_predictions[valid_predictions['round_number'] == next_round_number]
                next_round_predictions = next_round_predictions.sort_values(by='predicted_disposals', ascending=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                prediction_dir = Path("./data/prediction")
                prediction_dir.mkdir(parents=True, exist_ok=True)
                csv_path = prediction_dir / f"next_round_predictions_{timestamp}.csv"
                # Save only 'player' and 'predicted_disposals' columns
                next_round_predictions[['player', 'predicted_disposals']].to_csv(csv_path, index=False)
                print(f"📄 Next round predictions saved to {csv_path}")
            else:
                print("⚠️ No valid round predictions generated.")
        else:
            print("⚠️ No predictions generated.")
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