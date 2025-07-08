import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

class DisposalPredictor:
    def __init__(self, rolling_window=5, within_season_window=3, extra_features=['cba_percent', 'percentage_time_played']):
        self.rolling_window = rolling_window
        self.within_season_window = within_season_window
        self.extra_features = extra_features
        self.feature_columns = None
        self.training_feature_columns = None
        self.model = None
        self.target_year = None

    # Data type definitions for loading data
    DTYPES = {
        'player': 'category',
        'year': 'category',
        'round': 'category',
        'team': 'category',
        'opponent': 'category',
        'venue': 'category',
        'disposals': 'float32',
        'kicks': 'float32',
        'handballs': 'float32',
        'cba_percent': 'float32',
        'percentage_time_played': 'float32',
        'date': 'datetime64[ns]'
    }

    def load_player_optimized(self, file_path):
        """Load and optimize player data from a CSV file."""
        df = pd.read_csv(file_path)
        for col, dtype in self.DTYPES.items():
            if col in df.columns:
                if dtype in ['int8', 'int16', 'int32', 'int64']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
                elif dtype == 'category':
                    df[col] = df[col].astype('category')
                elif dtype == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
        return df

    def _extract_round_number(self, round_str):
        """Extract numerical round number from a string."""
        try:
            return int(''.join(filter(str.isdigit, str(round_str))))
        except (ValueError, TypeError):
            return np.nan

    def _engineer_features_vectorized(self, df):
        """Engineer features for the model."""
        df = df.copy()
        df['round_number'] = df['round'].apply(self._extract_round_number).astype('float32')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(['player', 'date'])

        rolling_cols = ['disposals', 'kicks', 'handballs']
        
        # Across-season rolling averages
        across_season_features = [
            f'across_season_rolling_avg_{col}_{self.rolling_window}' for col in rolling_cols
        ]
        for col, feat in zip(rolling_cols, across_season_features):
            df[feat] = (
                df.groupby('player')[col]
                .rolling(self.rolling_window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            ).astype('float32')

        # Within-season rolling averages
        within_season_features = [
            f'within_season_rolling_avg_{col}_{self.within_season_window}' for col in rolling_cols
        ]
        for col, feat in zip(rolling_cols, within_season_features):
            df[feat] = (
                df.groupby(['player', 'year'])[col]
                .rolling(self.within_season_window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            ).astype('float32')

        # Season-to-date means
        season_to_date_features = [f'season_to_date_mean_{col}' for col in rolling_cols]
        for col, feat in zip(rolling_cols, season_to_date_features):
            df[feat] = (
                df.groupby(['player', 'year'])[col]
                .expanding()
                .mean()
                .reset_index(level=[0, 1], drop=True)
            ).astype('float32')

        # Recent form (last game)
        recent_form_features = [f'recent_form_{col}' for col in rolling_cols]
        for col, feat in zip(rolling_cols, recent_form_features):
            df[feat] = df.groupby('player')[col].shift(1).astype('float32')

        # Days since last game
        df['days_since_last_game'] = (
            df.groupby('player')['date']
            .diff()
            .dt.days
            .fillna(0)
            .astype('float32')
        )

        # Extra features
        extra_feats = [feat for feat in self.extra_features if feat in df.columns]
        
        # Set feature columns
        self.feature_columns = (
            across_season_features +
            within_season_features +
            season_to_date_features +
            recent_form_features +
            ['round_number', 'days_since_last_game'] +
            extra_feats
        )

        return df

    def load_and_prepare_data_optimized(self, player_files):
        """Load and concatenate data for all players."""
        player_dfs = [self.load_player_optimized(file) for file in player_files]
        combined_df = pd.concat(player_dfs, ignore_index=True)
        return combined_df

    def train_optimized_model(self, X, y):
        """Train the model with cross-validation."""
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = HistGradientBoostingRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = mean_squared_error(y_val, y_pred, squared=False)
            scores.append(score)
        
        self.model = HistGradientBoostingRegressor(random_state=42)
        self.model.fit(X, y)
        return np.mean(scoresОн

    def run_optimized(self, player_files, target_year=2023):
        """Run the optimized training process."""
        self.target_year = target_year
        historical_data = self.load_and_prepare_data_optimized(player_files)
        
        # Engineer features
        engineered_df = self._engineer_features_vectorized(historical_data)
        
        # Prepare training data
        X = engineered_df.reindex(columns=self.feature_columns).apply(
            pd.to_numeric, errors='coerce'
        ).copy()
        X['missing_count'] = X.isna().sum(axis=1)
        X = X.fillna(X.median())
        y = np.log1p(engineered_df['disposals'])
        
        # Store training feature columns
        self.training_feature_columns = X.columns.tolist()
        
        # Train model and evaluate
        cv_score = self.train_optimized_model(X, y)
        print(f"Optimized model CV score: {cv_score:.4f}")
        
        return cv_score

    def predict_disposals(self, player_file, n_games=1):
        """Predict disposals for a player for the next n_games."""
        player_data = self.load_player_optimized(player_file)
        engineered_data = self._engineer_features_vectorized(player_data)
        
        # Select current season data
        current_season = engineered_data[engineered_data['year'] == str(self.target_year)].copy()
        
        # Prepare prediction data with consistent columns
        X_pred = current_season.reindex(columns=self.training_feature_columns).apply(
            pd.to_numeric, errors='coerce'
        ).copy()
        X_pred['missing_count'] = X_pred.isna().sum(axis=1)
        X_pred = X_pred.fillna(X_pred.median())
        
        # Make predictions
        log_preds = self.model.predict(X_pred)
        preds = np.expm1(log_preds)  # Reverse log1p transformation
        
        # Return last n_games predictions
        return preds[-n_games:].tolist()

# Example usage
if __name__ == "__main__":
    predictor = DisposalPredictor(rolling_window=5, within_season_window=3)
    player_files = ['player1.csv', 'player2.csv']  # Replace with actual file paths
    predictor.run_optimized(player_files, target_year=2023)
    
    # Predict for a specific player
    preds = predictor.predict_disposals('player1.csv', n_games=1)
    print(f"Predicted disposals: {preds}")