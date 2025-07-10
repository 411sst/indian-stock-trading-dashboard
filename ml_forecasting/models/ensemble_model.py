import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnsembleModel:
    """
    Ensemble model combining multiple prediction methods for stock forecasting.
    Uses moving averages, linear trends, and statistical methods for robust predictions.
    """
    
    def __init__(self):
        self.models = {
            'moving_average': self._moving_average_model,
            'linear_trend': self._linear_trend_model,
            'seasonal_naive': self._seasonal_naive_model,
            'exponential_smoothing': self._exponential_smoothing_model
        }
        self.weights = {
            'moving_average': 0.3,
            'linear_trend': 0.25,
            'seasonal_naive': 0.2,
            'exponential_smoothing': 0.25
        }
        self.confidence_base = 0.65
        
    def predict(self, data, steps=7, symbol=None):
        """
        Generate ensemble predictions for stock prices
        
        Args:
            data: pandas Series of historical prices
            steps: number of future periods to predict
            symbol: stock symbol for context
            
        Returns:
            dict with predictions, confidence, and metadata
        """
        try:
            if len(data) < 20:
                return self._simple_prediction(data, steps)
                
            # Generate predictions from each model
            predictions = {}
            confidences = {}
            
            for model_name, model_func in self.models.items():
                try:
                    pred, conf = model_func(data, steps)
                    predictions[model_name] = pred
                    confidences[model_name] = conf
                except Exception as e:
                    st.warning(f"Model {model_name} failed: {str(e)}")
                    predictions[model_name] = np.full(steps, data.iloc[-1])
                    confidences[model_name] = 0.3
            
            # Calculate weighted ensemble prediction
            ensemble_pred = np.zeros(steps)
            total_weight = 0
            
            for model_name, weight in self.weights.items():
                if model_name in predictions:
                    ensemble_pred += predictions[model_name] * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_pred = ensemble_pred / total_weight
            else:
                ensemble_pred = np.full(steps, data.iloc[-1])
            
            # Calculate overall confidence
            weighted_confidence = sum(
                confidences.get(name, 0.3) * weight 
                for name, weight in self.weights.items()
            ) / sum(self.weights.values())
            
            # Adjust confidence based on data quality and volatility
            volatility = data.pct_change().std()
            if volatility > 0.05:  # High volatility
                weighted_confidence *= 0.8
            elif volatility < 0.02:  # Low volatility
                weighted_confidence *= 1.1
                
            weighted_confidence = min(0.95, max(0.35, weighted_confidence))
            
            # Generate prediction dates
            last_date = data.index[-1]
            pred_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=steps,
                freq='B'  # Business days
            )
            
            # Calculate metrics
            current_price = data.iloc[-1]
            predicted_price = ensemble_pred[-1]
            price_change = ((predicted_price - current_price) / current_price) * 100
            
            return {
                'predictions': ensemble_pred,
                'dates': pred_dates,
                'confidence': weighted_confidence,
                'method': 'Ensemble (MA + Trend + Seasonal + ES)',
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_percent': price_change,
                'individual_predictions': predictions,
                'individual_confidences': confidences,
                'volatility': volatility,
                'data_points': len(data),
                'symbol': symbol or 'Unknown'
            }
            
        except Exception as e:
            st.error(f"Ensemble prediction failed: {str(e)}")
            return self._simple_prediction(data, steps)
    
    def _moving_average_model(self, data, steps):
        """Moving average based prediction"""
        # Use multiple MA periods for robustness
        ma_5 = data.rolling(window=5).mean().iloc[-1]
        ma_10 = data.rolling(window=10).mean().iloc[-1]
        ma_20 = data.rolling(window=20).mean().iloc[-1]
        
        # Weight recent MAs more heavily
        weighted_ma = (ma_5 * 0.5 + ma_10 * 0.3 + ma_20 * 0.2)
        
        # Calculate trend from MA slope
        ma_values = data.rolling(window=10).mean().tail(10)
        trend = (ma_values.iloc[-1] - ma_values.iloc[0]) / 9
        
        predictions = []
        current = weighted_ma
        
        for i in range(steps):
            # Add trend with decay
            trend_factor = 1 - (i * 0.1)  # Trend decays over time
            next_price = current + (trend * trend_factor)
            predictions.append(next_price)
            current = next_price
            
        confidence = 0.7 if len(data) >= 50 else 0.6
        return np.array(predictions), confidence
    
    def _linear_trend_model(self, data, steps):
        """Linear trend extrapolation"""
        # Use last 30 days for trend calculation
        recent_data = data.tail(min(30, len(data)))
        x = np.arange(len(recent_data))
        
        # Fit linear trend
        coeffs = np.polyfit(x, recent_data.values, 1)
        slope, intercept = coeffs
        
        # Project trend forward
        predictions = []
        last_x = len(recent_data) - 1
        
        for i in range(1, steps + 1):
            pred = slope * (last_x + i) + intercept
            predictions.append(pred)
            
        # Confidence based on R-squared
        trend_line = slope * x + intercept
        ss_res = np.sum((recent_data.values - trend_line) ** 2)
        ss_tot = np.sum((recent_data.values - np.mean(recent_data.values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        confidence = max(0.4, min(0.8, r_squared))
        return np.array(predictions), confidence
    
    def _seasonal_naive_model(self, data, steps):
        """Seasonal naive forecast (using weekly patterns)"""
        # Use last 5 trading days as seasonal pattern
        if len(data) >= 5:
            seasonal_pattern = data.tail(5).values
        else:
            seasonal_pattern = [data.iloc[-1]] * 5
            
        predictions = []
        for i in range(steps):
            # Cycle through the seasonal pattern
            seasonal_value = seasonal_pattern[i % len(seasonal_pattern)]
            
            # Add small random variation
            noise = np.random.normal(0, data.std() * 0.02)
            predictions.append(seasonal_value + noise)
            
        confidence = 0.55 if len(data) >= 20 else 0.45
        return np.array(predictions), confidence
    
    def _exponential_smoothing_model(self, data, steps):
        """Simple exponential smoothing"""
        alpha = 0.3  # Smoothing parameter
        
        # Initialize with first value
        smoothed = [data.iloc[0]]
        
        # Calculate exponentially smoothed values
        for i in range(1, len(data)):
            s_t = alpha * data.iloc[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(s_t)
            
        # Use last smoothed value for prediction
        last_smoothed = smoothed[-1]
        
        # Calculate trend from recent smoothed values
        if len(smoothed) >= 10:
            recent_smooth = smoothed[-10:]
            trend = (recent_smooth[-1] - recent_smooth[0]) / 9
        else:
            trend = 0
            
        predictions = []
        for i in range(steps):
            # Apply trend with diminishing effect
            trend_effect = trend * (0.9 ** i)
            pred = last_smoothed + trend_effect
            predictions.append(pred)
            
        confidence = 0.65
        return np.array(predictions), confidence
    
    def _simple_prediction(self, data, steps):
        """Fallback simple prediction for insufficient data"""
        if len(data) == 0:
            return {
                'predictions': np.array([100] * steps),
                'confidence': 0.3,
                'method': 'Default (insufficient data)'
            }
            
        current_price = data.iloc[-1]
        
        # Simple random walk with slight upward bias
        predictions = []
        price = current_price
        
        daily_return_std = data.pct_change().std() if len(data) > 1 else 0.02
        
        for i in range(steps):
            # Small upward bias with random variation
            change = np.random.normal(0.001, daily_return_std)  # 0.1% daily bias
            price = price * (1 + change)
            predictions.append(price)
            
        return {
            'predictions': np.array(predictions),
            'confidence': 0.4,
            'method': 'Simple Random Walk',
            'current_price': current_price,
            'predicted_price': predictions[-1],
            'data_points': len(data)
        }
    
    def get_prediction_summary(self, result):
        """Generate a human-readable summary of the prediction"""
        if not isinstance(result, dict):
            return "Prediction failed"
            
        confidence = result.get('confidence', 0) * 100
        price_change = result.get('price_change_percent', 0)
        method = result.get('method', 'Unknown')
        
        direction = "📈 increase" if price_change > 0 else "📉 decrease"
        strength = "strong" if abs(price_change) > 5 else "moderate" if abs(price_change) > 2 else "slight"
        
        summary = f"""
        **Prediction Summary:**
        - Expected {direction} of {abs(price_change):.1f}% ({strength})
        - Model confidence: {confidence:.0f}%
        - Method: {method}
        - Data quality: {'Good' if result.get('data_points', 0) > 50 else 'Limited'}
        """
        
        return summary.strip()
    
    def validate_prediction(self, result):
        """Validate prediction results for quality checks"""
        checks = {
            'has_predictions': 'predictions' in result and len(result['predictions']) > 0,
            'reasonable_prices': True,
            'confidence_range': 0.2 <= result.get('confidence', 0) <= 1.0,
            'no_infinite_values': True
        }
        
        if checks['has_predictions']:
            predictions = result['predictions']
            current_price = result.get('current_price', predictions[0] if len(predictions) > 0 else 100)
            
            # Check for reasonable price changes (not more than 50% in a week)
            max_change = max(abs(p - current_price) / current_price for p in predictions)
            checks['reasonable_prices'] = max_change < 0.5
            
            # Check for infinite or NaN values
            checks['no_infinite_values'] = not (np.any(np.isinf(predictions)) or np.any(np.isnan(predictions)))
        
        return checks, all(checks.values())

# Additional utility functions for the ensemble model
class TechnicalIndicators:
    """Helper class for technical analysis indicators"""
    
    @staticmethod
    def rsi(data, period=14):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

class PredictionMetrics:
    """Class for calculating prediction accuracy metrics"""
    
    @staticmethod
    def calculate_accuracy_metrics(actual, predicted):
        """Calculate various accuracy metrics"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Direction accuracy
        actual_direction = np.diff(actual) > 0
        pred_direction = np.diff(predicted) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }
