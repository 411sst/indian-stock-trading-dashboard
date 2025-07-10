"""
Utility functions for ML models and data preprocessing
"""

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Data preprocessing utilities for stock market data"""
    
    @staticmethod
    def clean_data(data):
        """Clean and validate stock data"""
        if data.empty:
            return data
            
        # Remove any infinite or NaN values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers (prices that are 5+ standard deviations away)
        for column in ['Open', 'High', 'Low', 'Close']:
            if column in data.columns:
                mean = data[column].mean()
                std = data[column].std()
                data = data[abs(data[column] - mean) <= 5 * std]
        
        return data
    
    @staticmethod
    def add_technical_indicators(data):
        """Add technical indicators to the dataset"""
        if len(data) < 20:
            return data
        
        try:
            # Moving averages
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            data['SMA_10'] = data['Close'].rolling(window=10).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            
            # Price changes
            data['Price_Change'] = data['Close'].pct_change()
            data['High_Low_Ratio'] = data['High'] / data['Low']
            data['Open_Close_Ratio'] = data['Open'] / data['Close']
            
        except Exception as e:
            st.warning(f"Could not calculate all technical indicators: {e}")
        
        return data
    
    @staticmethod
    def create_features(data, lookback_period=10):
        """Create feature matrix for ML models"""
        if len(data) < lookback_period + 5:
            return None, None
        
        features = []
        targets = []
        
        # Use technical indicators as features
        feature_columns = [
            'Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 
            'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Volume_Ratio'
        ]
        
        # Remove rows with NaN values
        clean_data = data.dropna()
        
        for i in range(lookback_period, len(clean_data) - 1):
            # Features: last lookback_period days of indicators
            feature_row = []
            for col in feature_columns:
                if col in clean_data.columns:
                    feature_row.extend(clean_data[col].iloc[i-lookback_period:i].tolist())
            
            if len(feature_row) > 0:
                features.append(feature_row)
                targets.append(clean_data['Close'].iloc[i + 1])  # Next day's close price
        
        return np.array(features), np.array(targets)

class ModelEvaluator:
    """Evaluation utilities for ML models"""
    
    @staticmethod
    def calculate_accuracy_metrics(actual, predicted):
        """Calculate comprehensive accuracy metrics"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        if len(actual) != len(predicted) or len(actual) == 0:
            return {}
        
        # Basic error metrics
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        
        # Percentage errors
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Direction accuracy
        actual_direction = np.diff(actual) > 0
        pred_direction = np.diff(predicted) > 0
        
        if len(actual_direction) > 0:
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            direction_accuracy = 50
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy,
            'R_Squared': r_squared,
            'Accuracy_Score': max(0, min(100, 100 - mape))  # Simple accuracy score
        }
    
    @staticmethod
    def validate_predictions(predictions, current_price, max_change_percent=50):
        """Validate that predictions are reasonable"""
        if len(predictions) == 0:
            return False, "No predictions to validate"
        
        # Check for NaN or infinite values
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            return False, "Predictions contain invalid values"
        
        # Check for unreasonable price changes
        max_change = np.max(np.abs((predictions - current_price) / current_price)) * 100
        if max_change > max_change_percent:
            return False, f"Predicted change too large: {max_change:.1f}%"
        
        # Check for negative prices
        if np.any(predictions <= 0):
            return False, "Predictions contain negative prices"
        
        return True, "Predictions are valid"

class RiskAnalyzer:
    """Risk analysis utilities"""
    
    @staticmethod
    def calculate_volatility(price_data, window=20):
        """Calculate rolling volatility"""
        returns = price_data.pct_change().dropna()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        return volatility.iloc[-1] if len(volatility) > 0 else 0.2
    
    @staticmethod
    def calculate_var(returns, confidence_level=0.05):
        """Calculate Value at Risk"""
        if len(returns) < 30:
            return 0.05  # Default 5% VaR
        
        returns = returns.dropna()
        var = np.percentile(returns, confidence_level * 100)
        return abs(var)
    
    @staticmethod
    def risk_adjusted_prediction(predictions, risk_level='moderate'):
        """Adjust predictions based on risk level"""
        risk_multipliers = {
            'conservative': 0.7,
            'moderate': 1.0,
            'aggressive': 1.3
        }
        
        multiplier = risk_multipliers.get(risk_level, 1.0)
        
        # Adjust the magnitude of price changes
        baseline = predictions[0] if len(predictions) > 0 else 100
        changes = predictions - baseline
        adjusted_changes = changes * multiplier
        
        return baseline + adjusted_changes

class MarketRegimeDetector:
    """Detect market conditions for better predictions"""
    
    @staticmethod
    def detect_trend(price_data, window=20):
        """Detect market trend"""
        if len(price_data) < window:
            return 'sideways'
        
        recent_prices = price_data.tail(window)
        slope, _ = np.polyfit(range(len(recent_prices)), recent_prices, 1)
        
        # Normalize slope by price level
        normalized_slope = slope / recent_prices.mean()
        
        if normalized_slope > 0.002:  # 0.2% daily trend
            return 'uptrend'
        elif normalized_slope < -0.002:
            return 'downtrend'
        else:
            return 'sideways'
    
    @staticmethod
    def detect_volatility_regime(price_data, window=20):
        """Detect volatility regime"""
        if len(price_data) < window:
            return 'normal'
        
        returns = price_data.pct_change().dropna()
        recent_vol = returns.tail(window).std() * np.sqrt(252)
        historical_vol = returns.std() * np.sqrt(252)
        
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1
        
        if vol_ratio > 1.5:
            return 'high_volatility'
        elif vol_ratio < 0.7:
            return 'low_volatility'
        else:
            return 'normal'
    
    @staticmethod
    def get_market_confidence_adjustment(trend, volatility_regime):
        """Adjust model confidence based on market conditions"""
        base_confidence = 1.0
        
        # Trend adjustments
        if trend == 'uptrend' or trend == 'downtrend':
            base_confidence *= 1.1  # Trending markets are more predictable
        else:
            base_confidence *= 0.9  # Sideways markets are harder to predict
        
        # Volatility adjustments
        if volatility_regime == 'high_volatility':
            base_confidence *= 0.8  # High volatility reduces confidence
        elif volatility_regime == 'low_volatility':
            base_confidence *= 1.1  # Low volatility increases confidence
        
        return min(1.2, max(0.6, base_confidence))  # Cap between 60% and 120%

# Utility functions for data fetching and caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_and_preprocess_data(symbol, period='1y'):
    """Fetch and preprocess stock data with caching"""
    try:
        import yfinance as yf
        
        # Fetch data
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            return None
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        data = preprocessor.clean_data(data)
        data = preprocessor.add_technical_indicators(data)
        
        return data
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def generate_prediction_report(symbol, prediction_result, risk_analysis=True):
    """Generate a comprehensive prediction report"""
    report = {
        'symbol': symbol,
        'timestamp': datetime.now(),
        'prediction_summary': prediction_result,
        'risk_metrics': {},
        'market_conditions': {},
        'recommendations': []
    }
    
    if risk_analysis and 'predictions' in prediction_result:
        # Add risk analysis
        predictions = prediction_result['predictions']
        current_price = prediction_result.get('current_price', predictions[0])
        
        # Calculate risk metrics
        price_changes = [(p - current_price) / current_price for p in predictions]
        max_loss = min(price_changes) * 100
        max_gain = max(price_changes) * 100
        
        report['risk_metrics'] = {
            'max_potential_loss': max_loss,
            'max_potential_gain': max_gain,
            'volatility_estimate': np.std(price_changes) * 100,
            'risk_reward_ratio': abs(max_gain / max_loss) if max_loss != 0 else 0
        }
        
        # Generate recommendations
        confidence = prediction_result.get('confidence', 0.5)
        
        if confidence > 0.8:
            report['recommendations'].append("High confidence prediction - suitable for position sizing")
        elif confidence > 0.6:
            report['recommendations'].append("Moderate confidence - consider smaller position sizes")
        else:
            report['recommendations'].append("Low confidence - wait for better signals or avoid trading")
        
        if abs(max_loss) > 10:
            report['recommendations'].append("High risk detected - implement strict stop losses")
        
        if abs(max_gain) > 15:
            report['recommendations'].append("High potential upside - consider profit-taking levels")
    
    return report
