import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from cuml.ensemble import RandomForestRegressor as cuRFRegressor
from cuml.linear_model import Ridge as cuRidge
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')


class DisposalPredictionModel:
    """Predict AFL player disposals using GPU-accelerated models."""

    def __init__(self, data_dir: str = "./data/player_data", model_dir: str = "./models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.models: Dict[str, object] = {}
        self.scalers: Dict[str, object] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: List[str] = []
        self.target_column = 'disposals'
        os.makedirs(model_dir, exist_ok=True)

        self.model_configs = {
            'random_forest': {
                'model': cuRFRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': XGBRegressor(random_state=42, tree_method='gpu_hist'),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'ridge_regression': {
                'model': cuRidge(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['eig', 'svd']
                }
            }
        }
        logging.info("DisposalPredictionModel initialised")

    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Validating and cleaning data")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(r'[\u2191\u2193\u2192\u2190]', '', regex=True)
                df[col] = df[col].str.replace(r'[^\w\s.-]', '', regex=True)
        return df

    def load_and_prepare_data(self) -> pd.DataFrame:
        logging.info("Loading player performance data")
        all_data: List[pd.DataFrame] = []
        current_date = datetime.now()
        three_years_ago = current_date - timedelta(days=1095)
        for filename in os.listdir(self.data_dir):
            if filename.endswith("_performance_details.csv"):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    player_id = "_".join(filename.split("_")[:-2])
                    df = pd.read_csv(filepath, low_memory=False)
                    if df.empty or 'disposals' not in df.columns:
                        continue
                    df = self.validate_and_clean_data(df)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        df.dropna(subset=['date'], inplace=True)
                        recent_games = df[df['date'] >= three_years_ago]
                        if recent_games.empty:
                            continue
                        df['player_id'] = player_id
                        df['year'] = df['date'].dt.year
                        df['month'] = df['date'].dt.month
                        df['day_of_year'] = df['date'].dt.dayofyear
                    all_data.append(df)
                except Exception as exc:
                    logging.warning(f"Error processing {filename}: {exc}")
        if not all_data:
            raise ValueError("No valid player data found")
        combined = pd.concat(all_data, ignore_index=True)
        return self._engineer_features(combined)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Engineering features")
        numeric_columns = [
            'kicks', 'marks', 'handballs', 'disposals', 'goals', 'behinds',
            'hit_outs', 'tackles', 'rebound_50s', 'inside_50s', 'clearances',
            'clangers', 'free_kicks_for', 'free_kicks_against', 'brownlow_votes',
            'contested_possessions', 'uncontested_possessions', 'contested_marks',
            'marks_inside_50', 'one_percenters', 'bounces', 'goal_assist'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                df[col] = df[col].replace('', '0')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df = df[df['disposals'] > 0].copy()
        df.sort_values(['player_id', 'date'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        for window in [3, 5, 10]:
            for stat in ['kicks', 'handballs', 'marks', 'tackles']:
                if stat in df.columns:
                    df[f'{stat}_avg_{window}'] = df.groupby('player_id')[stat].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
        if 'kicks' in df.columns and 'handballs' in df.columns:
            df['kick_handball_ratio'] = df['kicks'] / (df['handballs'] + 1)
        if 'contested_possessions' in df.columns and 'uncontested_possessions' in df.columns:
            total_poss = df['contested_possessions'] + df['uncontested_possessions']
            df['contested_possession_rate'] = df['contested_possessions'] / (total_poss + 1)
        if 'hit_outs' in df.columns:
            df['likely_ruck'] = (df['hit_outs'] > 10).astype(int)
        if 'goals' in df.columns:
            df['likely_forward'] = (df['goals'] > 1).astype(int)
        if 'rebound_50s' in df.columns:
            df['likely_defender'] = (df['rebound_50s'] > 2).astype(int)
        if 'year' in df.columns:
            df['games_this_season'] = df.groupby(['player_id', 'year']).cumcount() + 1
        df['career_games'] = df.groupby('player_id').cumcount() + 1
        df['recent_form_disposals'] = df.groupby('player_id')['disposals'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        return df

    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        exclude_cols = ['disposals', 'player_id', 'date', 'team', 'opponent', 'round', 'result']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        categorical = ['year', 'month']
        for col in categorical:
            if col in feature_cols:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.encoders[col].transform(df[col].astype(str))
        X = df[feature_cols].copy()
        y = df[self.target_column].copy()
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                X[col] = X[col].replace('', '0')
                X[col] = pd.to_numeric(X[col], errors='coerce')
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        X = X.fillna(0)
        self.feature_columns = feature_cols
        return X, y

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        logging.info("Training models")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        results: Dict[str, Dict] = {}
        for name, config in self.model_configs.items():
            try:
                grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                if 'ridge' in name:
                    grid.fit(X_train_scaled, y_train)
                    preds = grid.predict(X_test_scaled)
                else:
                    grid.fit(X_train, y_train)
                    preds = grid.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                self.models[name] = grid.best_estimator_
                results[name] = {
                    'model': grid.best_estimator_,
                    'best_params': grid.best_params_,
                    'cv_score': -grid.best_score_,
                    'test_rmse': rmse,
                    'test_mae': mae,
                    'test_r2': r2,
                    'feature_importance': self._get_feature_importance(grid.best_estimator_, name)
                }
                logging.info(f"{name} - RMSE: {rmse:.3f}, R2: {r2:.3f}")
            except Exception as exc:
                logging.error(f"Error training {name}: {exc}")
        return results

    def _get_feature_importance(self, model, name: str) -> Optional[pd.Series]:
        try:
            if hasattr(model, 'feature_importances_'):
                return pd.Series(model.feature_importances_, index=self.feature_columns)
            if hasattr(model, 'coef_'):
                return pd.Series(np.abs(model.coef_), index=self.feature_columns)
            return None
        except Exception as exc:
            logging.warning(f"Could not extract feature importance for {name}: {exc}")
            return None

    def save_models(self) -> None:
        logging.info("Saving models")
        for name, model in self.models.items():
            path = os.path.join(self.model_dir, f"{name}_disposal_model.joblib")
            joblib.dump(model, path)
        joblib.dump(self.scalers, os.path.join(self.model_dir, "scalers.joblib"))
        joblib.dump(self.encoders, os.path.join(self.model_dir, "encoders.joblib"))
        joblib.dump(self.feature_columns, os.path.join(self.model_dir, "feature_columns.joblib"))

    def load_models(self) -> None:
        logging.info("Loading models")
        for name in self.model_configs.keys():
            path = os.path.join(self.model_dir, f"{name}_disposal_model.joblib")
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
        self.scalers = joblib.load(os.path.join(self.model_dir, "scalers.joblib"))
        self.encoders = joblib.load(os.path.join(self.model_dir, "encoders.joblib"))
        self.feature_columns = joblib.load(os.path.join(self.model_dir, "feature_columns.joblib"))

    def predict_current_season_disposals(self, output_file: str = "./data/predicted_disposals_ranking.csv") -> pd.DataFrame:
        logging.info("Predicting disposals for current season")
        if not self.models:
            raise ValueError("No trained models available")
        best_name = self._select_best_model()
        best_model = self.models[best_name]
        current_year = datetime.now().year
        predictions: List[Dict[str, object]] = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith("_performance_details.csv"):
                try:
                    player_id = "_".join(filename.split("_")[:-2])
                    filepath = os.path.join(self.data_dir, filename)
                    df = pd.read_csv(filepath, low_memory=False)
                    if df.empty:
                        continue
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        df.dropna(subset=['date'], inplace=True)
                        recent_cutoff = datetime.now() - timedelta(days=730)
                        df = df[df['date'] >= recent_cutoff]
                    if df.empty:
                        continue
                    features = self._prepare_prediction_features(df, player_id, current_year)
                    if features is not None:
                        if 'ridge' in best_name:
                            features_scaled = self.scalers['standard'].transform(features.reshape(1, -1))
                            pred = best_model.predict(features_scaled)[0]
                        else:
                            pred = best_model.predict(features.reshape(1, -1))[0]
                        recent_avg = df['disposals'].tail(10).mean() if 'disposals' in df.columns else 0
                        predictions.append({
                            'player_id': player_id,
                            'predicted_disposals': round(pred, 2),
                            'recent_avg_disposals': round(recent_avg, 2),
                            'prediction_confidence': self._calculate_confidence(df),
                            'games_analyzed': len(df),
                            'model_used': best_name
                        })
                except Exception as exc:
                    logging.warning(f"Error predicting for {filename}: {exc}")
        ranking_df = pd.DataFrame(predictions)
        if ranking_df.empty:
            logging.warning("No predictions generated")
            return ranking_df
        ranking_df.sort_values('predicted_disposals', ascending=False, inplace=True)
        ranking_df.reset_index(drop=True, inplace=True)
        ranking_df['rank'] = ranking_df.index + 1
        ranking_df['disposal_category'] = pd.cut(
            ranking_df['predicted_disposals'],
            bins=[0, 15, 20, 25, 30, float('inf')],
            labels=['Low', 'Medium', 'High', 'Elite', 'Exceptional']
        )
        ranking_df.to_csv(output_file, index=False)
        return ranking_df

    def _select_best_model(self) -> str:
        if 'random_forest' in self.models:
            return 'random_forest'
        if 'gradient_boosting' in self.models:
            return 'gradient_boosting'
        return next(iter(self.models))

    def _prepare_prediction_features(self, df: pd.DataFrame, player_id: str, year: int) -> Optional[np.ndarray]:
        try:
            df = df.copy()
            df['player_id'] = player_id
            df['year'] = year
            df['month'] = datetime.now().month
            df['day_of_year'] = datetime.now().timetuple().tm_yday
            df = self._engineer_features(df)
            latest = df.iloc[-1:].copy()
            missing = set(self.feature_columns) - set(latest.columns)
            for feat in missing:
                latest[feat] = 0
            return latest[self.feature_columns].values[0]
        except Exception as exc:
            logging.warning(f"Error preparing features for {player_id}: {exc}")
            return None

    def _calculate_confidence(self, df: pd.DataFrame) -> float:
        recent_games = len(df)
        completeness = df.notna().mean().mean()
        if 'date' in df.columns and not df.empty:
            latest_date = pd.to_datetime(df['date']).max()
            days_since = (datetime.now() - latest_date).days
            recency = max(0, 1 - (days_since / 365))
        else:
            recency = 0.5
        confidence = min(1.0, (recent_games / 50) * completeness * recency)
        return round(confidence, 3)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    predictor = DisposalPredictionModel()
    model_files = [f for f in os.listdir(predictor.model_dir) if f.endswith('.joblib')]
    if model_files:
        logging.info("Loading pre-trained models")
        predictor.load_models()
    else:
        logging.info("Training new models")
        data = predictor.load_and_prepare_data()
        X, y = predictor.prepare_features_and_target(data)
        predictor.train_models(X, y)
        predictor.save_models()
    rankings = predictor.predict_current_season_disposals()
    print("Top predictions:\n", rankings.head(20))


if __name__ == "__main__":
    main()
