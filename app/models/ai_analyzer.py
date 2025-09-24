"""
AI Stock Analyzer - Advanced machine learning model for stock analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import ta
from typing import Dict, List, Tuple, Optional
import logging
import joblib
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIStockAnalyzer:
    """Advanced AI model for stock analysis using multiple technical indicators"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.price_model = None
        self.trend_model = None
        self.volatility_model = None
        self.model_dir = "data/models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load existing models if available
        self.load_models()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        try:
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Need: {required_cols}")
            
            # Price-based indicators
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
            df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
            
            # Momentum indicators
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            df['RSI_30'] = ta.momentum.rsi(df['Close'], window=30)
            df['Stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
            df['Stoch_Signal'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
            df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
            
            # Trend indicators
            df['MACD'] = ta.trend.macd(df['Close'])
            df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
            df['MACD_Histogram'] = ta.trend.macd_diff(df['Close'])
            df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
            df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
            
            # Volatility indicators
            df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'])
            df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'])
            df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'])
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            df['Keltner_Upper'] = ta.volatility.keltner_channel_hband(df['High'], df['Low'], df['Close'])
            df['Keltner_Lower'] = ta.volatility.keltner_channel_lband(df['High'], df['Low'], df['Close'])
            
            # Volume indicators
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
            df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Price patterns (simplified)
            df['Doji'] = np.where((df['Open'] == df['Close']) & (df['High'] != df['Low']), 1, 0)
            df['Hammer'] = np.where((df['Close'] > df['Open']) & ((df['Open'] - df['Low']) > 2 * (df['Close'] - df['Open'])), 1, 0)
            df['Shooting_Star'] = np.where((df['Open'] > df['Close']) & ((df['High'] - df['Open']) > 2 * (df['Open'] - df['Close'])), 1, 0)
            
            # Custom indicators
            df['Price_Change'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Close_Open_Ratio'] = df['Close'] / df['Open']
            
            # Moving average crossovers
            df['SMA_20_50_Crossover'] = np.where(df['SMA_20'] > df['SMA_50'], 1, 0)
            df['SMA_50_200_Crossover'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)
            
            # Bollinger Band position
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # RSI levels
            df['RSI_Overbought'] = np.where(df['RSI'] > 70, 1, 0)
            df['RSI_Oversold'] = np.where(df['RSI'] < 30, 1, 0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix for ML models"""
        try:
            # Calculate technical indicators
            df_with_indicators = self.calculate_technical_indicators(df.copy())
            
            # Select features for ML
            feature_columns = [
                'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
                'RSI', 'RSI_30', 'Stoch', 'Williams_R',
                'MACD', 'MACD_Signal', 'MACD_Histogram', 'ADX', 'CCI',
                'BB_Upper', 'BB_Lower', 'BB_Middle', 'ATR',
                'OBV', 'MFI', 'VWAP',
                'Price_Change', 'Volume_Change', 'High_Low_Ratio', 'Close_Open_Ratio',
                'SMA_20_50_Crossover', 'SMA_50_200_Crossover',
                'BB_Position', 'RSI_Overbought', 'RSI_Oversold'
            ]
            
            # Create feature matrix
            features_df = df_with_indicators[feature_columns].copy()
            
            # Handle NaN values using modern pandas syntax
            features_df = features_df.ffill().bfill().fillna(0)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            raise
    
    def create_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create target variables for different prediction tasks"""
        try:
            targets = {}
            
            # Price direction (1: up, 0: down) - look ahead 5 days
            price_future = df['Close'].shift(-5)
            targets['price_direction'] = (price_future > df['Close']).astype(int)
            
            # Price change magnitude (regression) - 5-day return
            targets['price_change'] = (price_future / df['Close'] - 1) * 100  # Convert to percentage
            
            # Volatility prediction - 10-day rolling volatility
            targets['volatility'] = df['Close'].rolling(10).std() / df['Close'].rolling(10).mean() * 100
            
            # Trend strength - 20-day slope
            def calculate_slope(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                try:
                    slope, _ = np.polyfit(x, series, 1)
                    return slope
                except:
                    return 0
            
            targets['trend_strength'] = df['Close'].rolling(20).apply(calculate_slope, raw=True)
            
            # Remove NaN values from all targets
            for key in targets:
                targets[key] = targets[key].fillna(0)
            
            return targets
            
        except Exception as e:
            logger.error(f"Error creating targets: {str(e)}")
            raise
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train ML models for stock prediction"""
        try:
            logger.info("Training AI models...")
            
            # Create features and targets
            features_df = self.create_features(df)
            targets = self.create_targets(df)
            
            # Remove rows with NaN targets and ensure we have enough data
            valid_indices = ~targets['price_direction'].isna() & ~targets['price_change'].isna() & ~targets['volatility'].isna()
            
            if valid_indices.sum() < 50:  # Need at least 50 valid samples
                logger.warning(f"Not enough valid data for training: {valid_indices.sum()} samples")
                return {'error': 'Insufficient data for training'}
            
            X = features_df[valid_indices]
            y_direction = targets['price_direction'][valid_indices]
            y_change = targets['price_change'][valid_indices]
            y_volatility = targets['volatility'][valid_indices]
            
            # Ensure targets are finite
            y_direction = y_direction.fillna(0)
            y_change = y_change.fillna(0)
            y_volatility = y_volatility.fillna(0)
            
            # Split data
            X_train, X_test, y_train_dir, y_test_dir = train_test_split(
                X, y_direction, test_size=0.2, random_state=42
            )
            
            _, _, y_train_change, y_test_change = train_test_split(
                X, y_change, test_size=0.2, random_state=42
            )
            
            _, _, y_train_vol, y_test_vol = train_test_split(
                X, y_volatility, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train price direction model (classification)
            self.price_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.price_model.fit(X_train_scaled, y_train_dir)
            
            # Train price change model (regression)
            self.trend_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            self.trend_model.fit(X_train_scaled, y_train_change)
            
            # Train volatility model (regression)
            self.volatility_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            self.volatility_model.fit(X_train_scaled, y_train_vol)
            
            # Evaluate models
            price_pred = self.price_model.predict(X_test_scaled)
            trend_pred = self.trend_model.predict(X_test_scaled)
            vol_pred = self.volatility_model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test_dir, price_pred)
            trend_mse = mean_squared_error(y_test_change, trend_pred)
            vol_mse = mean_squared_error(y_test_vol, vol_pred)
            
            # Save models
            self.save_models()
            
            logger.info(f"Models trained successfully. Accuracy: {accuracy:.3f}")
            
            return {
                'price_direction_accuracy': accuracy,
                'trend_prediction_mse': trend_mse,
                'volatility_prediction_mse': vol_mse
            }
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        """Make predictions using trained models"""
        try:
            if not all([self.price_model, self.trend_model, self.volatility_model]):
                logger.warning("Models not trained. Training with available data...")
                self.train_models(df)
            
            # Create features for prediction
            features_df = self.create_features(df)
            
            # Use the most recent data point
            latest_features = features_df.iloc[-1:].fillna(method='ffill').fillna(method='bfill')
            
            # Scale features
            features_scaled = self.scaler.transform(latest_features)
            
            # Make predictions
            price_direction_prob = self.price_model.predict_proba(features_scaled)[0]
            trend_prediction = self.trend_model.predict(features_scaled)[0]
            volatility_prediction = self.volatility_model.predict(features_scaled)[0]
            
            # Calculate confidence scores
            confidence = max(price_direction_prob)
            
            return {
                'price_direction_probability': price_direction_prob[1],  # Probability of price going up
                'trend_prediction': trend_prediction,
                'volatility_prediction': volatility_prediction,
                'confidence': confidence,
                'prediction_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {
                'error': str(e),
                'price_direction_probability': 0.5,
                'trend_prediction': 0.0,
                'volatility_prediction': 0.0,
                'confidence': 0.0
            }
    
    def generate_recommendation(self, predictions: Dict, stock_info: Dict) -> Dict:
        """Generate investment recommendation based on predictions and fundamentals"""
        try:
            price_prob = predictions.get('price_direction_probability', 0.5)
            trend_pred = predictions.get('trend_prediction', 0.0)
            volatility_pred = predictions.get('volatility_prediction', 0.0)
            confidence = predictions.get('confidence', 0.0)
            
            # Fundamental analysis factors
            pe_ratio = stock_info.get('pe_ratio', 0)
            peg_ratio = stock_info.get('peg_ratio', 0)
            beta = stock_info.get('beta', 1.0)
            profit_margin = stock_info.get('profit_margin', 0)
            debt_to_equity = stock_info.get('debt_to_equity', 0)
            current_price = stock_info.get('current_price', 0)
            market_cap = stock_info.get('market_cap', 0)
            
            # Advanced scoring system
            score = 50  # Start neutral
            reasons = []
            
            # Technical analysis score (35% weight)
            if price_prob > 0.7:
                score += 25
                reasons.append("Strong bullish technical momentum")
            elif price_prob > 0.6:
                score += 15
                reasons.append("Positive technical indicators")
            elif price_prob > 0.4:
                score += 5
                reasons.append("Mixed technical signals")
            elif price_prob > 0.3:
                score -= 10
                reasons.append("Bearish technical indicators")
            else:
                score -= 20
                reasons.append("Strong bearish technical signals")
            
            # Trend analysis score (25% weight)
            if trend_pred > 2.0:  # 2% expected gain
                score += 20
                reasons.append("Strong positive trend momentum")
            elif trend_pred > 0.5:
                score += 10
                reasons.append("Positive trend momentum")
            elif trend_pred > 0:
                score += 5
                reasons.append("Slight positive trend")
            elif trend_pred > -0.5:
                score -= 5
                reasons.append("Slight negative trend")
            else:
                score -= 15
                reasons.append("Negative trend momentum")
            
            # Fundamental analysis score (25% weight)
            if pe_ratio > 0:  # Only if we have valid P/E data
                if 8 < pe_ratio < 20 and peg_ratio < 1.5:
                    score += 20
                    reasons.append("Excellent valuation metrics")
                elif 5 < pe_ratio < 25 and peg_ratio < 2.0:
                    score += 15
                    reasons.append("Attractive valuation")
                elif pe_ratio < 8:
                    score += 10
                    reasons.append("Potentially undervalued")
                elif pe_ratio > 30:
                    score -= 15
                    reasons.append("Overvalued based on P/E ratio")
                elif pe_ratio > 50:
                    score -= 25
                    reasons.append("Significantly overvalued")
            
            # Financial health score (15% weight)
            if profit_margin > 0.15:  # 15% profit margin
                score += 10
                reasons.append("Strong profitability")
            elif profit_margin > 0.10:
                score += 5
                reasons.append("Good profitability")
            elif profit_margin < 0.05:
                score -= 10
                reasons.append("Low profitability")
            
            if debt_to_equity < 0.5:
                score += 5
                reasons.append("Low debt levels")
            elif debt_to_equity > 2.0:
                score -= 10
                reasons.append("High debt levels")
            
            # Risk assessment (10% weight)
            if beta < 0.8 and volatility_pred < 1.0:
                score += 8
                reasons.append("Low risk profile")
            elif beta < 1.2 and volatility_pred < 2.0:
                score += 3
                reasons.append("Moderate risk profile")
            elif beta > 1.5:
                score -= 5
                reasons.append("High beta - increased volatility")
            
            # Market cap consideration
            if market_cap > 100000000000:  # Large cap (>$100B)
                score += 3
                reasons.append("Large cap stability")
            elif market_cap < 1000000000:  # Small cap (<$1B)
                score -= 5
                reasons.append("Small cap volatility")
            
            # Ensure score is within bounds
            score = max(0, min(100, score))
            
            # Generate recommendation with better thresholds
            if score >= 75:
                recommendation = "STRONG BUY"
                risk_level = "LOW"
            elif score >= 60:
                recommendation = "BUY"
                risk_level = "MEDIUM"
            elif score >= 45:
                recommendation = "HOLD"
                risk_level = "MEDIUM"
            elif score >= 30:
                recommendation = "SELL"
                risk_level = "HIGH"
            else:
                recommendation = "STRONG SELL"
                risk_level = "HIGH"
            
            return {
                'recommendation': recommendation,
                'score': int(score),
                'risk_level': risk_level,
                'confidence': min(confidence, 0.95),  # Cap confidence
                'reasons': reasons,
                'expected_return': trend_pred,
                'volatility': volatility_pred,
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            return {
                'recommendation': 'HOLD',
                'score': 50,
                'risk_level': 'MEDIUM',
                'confidence': 0.0,
                'reasons': ['Analysis error - using default recommendation'],
                'error': str(e)
            }
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            if self.price_model:
                joblib.dump(self.price_model, os.path.join(self.model_dir, 'price_model.pkl'))
            if self.trend_model:
                joblib.dump(self.trend_model, os.path.join(self.model_dir, 'trend_model.pkl'))
            if self.volatility_model:
                joblib.dump(self.volatility_model, os.path.join(self.model_dir, 'volatility_model.pkl'))
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            price_model_path = os.path.join(self.model_dir, 'price_model.pkl')
            trend_model_path = os.path.join(self.model_dir, 'trend_model.pkl')
            volatility_model_path = os.path.join(self.model_dir, 'volatility_model.pkl')
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            
            if os.path.exists(price_model_path):
                self.price_model = joblib.load(price_model_path)
            if os.path.exists(trend_model_path):
                self.trend_model = joblib.load(trend_model_path)
            if os.path.exists(volatility_model_path):
                self.volatility_model = joblib.load(volatility_model_path)
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load models: {str(e)}")
