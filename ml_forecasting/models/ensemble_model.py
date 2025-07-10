import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnsembleModel:
    """
    Fixed ensemble model for stock forecasting with proper error handling.
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
        Generate ensemble predictions for stock prices with improved error handling
        """
        try:
            if data is None or len(data) < 5:
                return self._simple_prediction(steps)
                
            # Ensure data is a pandas Series
            if isinstance(data, pd.DataFrame):
                if 'Close' in data.columns:
                    data = data['Close']
                else:
                    data = data.iloc[:, -1]  # Use last column
            
            # Convert to Series if it's not already
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            
            # Remove any NaN values
            data = data.dropna()
            
            if len(data) < 5:
                return self._simple_prediction(steps)
                
            # Generate predictions from each model
            predictions = {}
            confidences = {}
            
            for model_name, model_func in self.models.items():
                try:
                    pred, conf = model_func(data, steps)
                    # Ensure predictions are 1D array
                    pred = np.array(pred).flatten()
                    if len(pred) == steps:
                        predictions[model_name] = pred
                        confidences[model_name] = conf
                    else:
                        # Fallback if wrong length
                        predictions[model_name] = np.full(steps, data.iloc[-1])
                        confidences[model_name] = 0.3
                except Exception as e:
                    st.warning(f"Model {model_name} failed: {str(e)}")
                    predictions[model_name] = np.full(steps, data.iloc[-1])
                    confidences[model_name] = 0.3
            
            # Calculate weighted ensemble prediction with proper array handling
            ensemble_pred = np.zeros(steps)
            total_weight = 0
            
            for model_name, weight in self.weights.items():
                if model_name in predictions:
                    pred_array = np.array(predictions[model_name]).flatten()
                    if len(pred_array) == steps:
                        ensemble_pred += pred_array * weight
                        total_weight += weight
            
            if total_weight > 0:
                ensemble_pred = ensemble_pred / total_weight
            else:
                ensemble_pred = np.full(steps, data.iloc[-1])
            
            # Calculate overall confidence
            if confidences:
                weighted_confidence = sum(
                    confidences.get(name, 0.3) * weight 
                    for name, weight in self.weights.items()
                ) / sum(self.weights.values())
            else:
                weighted_confidence = 0.5
            
            # Adjust confidence based on data quality and volatility
            try:
                volatility = data.pct_change().std()
                if not np.isnan(volatility):
                    if volatility > 0.05:  # High volatility
                        weighted_confidence *= 0.8
                    elif volatility < 0.02:  # Low volatility
                        weighted_confidence *= 1.1
            except:
                volatility = 0.03  # Default volatility
                
            weighted_confidence = min(0.95, max(0.35, weighted_confidence))
            
            # Generate prediction dates
            try:
                if hasattr(data, 'index') and len(data.index) > 0:
                    last_date = data.index[-1]
                else:
                    last_date = datetime.now()
                    
                pred_dates = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=steps,
                    freq='B'  # Business days
                )
            except:
                pred_dates = pd.date_range(
                    start=datetime.now() + timedelta(days=1),
                    periods=steps,
                    freq='D'
                )
            
            # Calculate metrics safely
            current_price = float(data.iloc[-1])
            predicted_price = float(ensemble_pred[-1])
            
            # Fix the price_change_percent calculation
            if current_price != 0:
                price_change = ((predicted_price - current_price) / current_price) * 100
            else:
                price_change = 0.0
            
            return {
                'predictions': ensemble_pred,
                'dates': pred_dates,
                'confidence': float(weighted_confidence),
                'method': 'Ensemble (MA + Trend + Seasonal + ES)',
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_percent': price_change,
                'individual_predictions': predictions,
                'individual_confidences': confidences,
                'volatility': float(volatility) if not np.isnan(volatility) else 0.03,
                'data_points': len(data),
                'symbol': symbol or 'Unknown'
            }
            
        except Exception as e:
            st.error(f"Ensemble prediction failed: {str(e)}")
            return self._simple_prediction(steps, symbol)
    
    def _moving_average_model(self, data, steps):
        """Moving average based prediction with better error handling"""
        try:
            # Use different MA periods based on data length
            ma_periods = [min(5, len(data)//2), min(10, len(data)//2), min(20, len(data))]
            ma_periods = [p for p in ma_periods if p > 0]
            
            if not ma_periods:
                return np.full(steps, data.iloc[-1]), 0.4
            
            # Calculate moving averages
            mas = []
            for period in ma_periods:
                if len(data) >= period:
                    ma = data.rolling(window=period).mean().iloc[-1]
                    if not np.isnan(ma):
                        mas.append(ma)
            
            if not mas:
                return np.full(steps, data.iloc[-1]), 0.4
                
            # Weight recent MAs more heavily
            if len(mas) >= 3:
                weighted_ma = mas[0] * 0.5 + mas[1] * 0.3 + mas[2] * 0.2
            elif len(mas) == 2:
                weighted_ma = mas[0] * 0.7 + mas[1] * 0.3
            else:
                weighted_ma = mas[0]
            
            # Calculate trend
            if len(data) >= 10:
                recent_data = data.tail(10)
                trend = (recent_data.iloc[-1] - recent_data.iloc[0]) / 9
            else:
                trend = 0
            
            predictions = []
            current = weighted_ma
            
            for i in range(steps):
                # Add trend with decay
                trend_factor = max(0.1, 1 - (i * 0.1))
                next_price = current + (trend * trend_factor)
                predictions.append(max(0.01, next_price))  # Ensure positive prices
                current = next_price
                
            confidence = 0.7 if len(data) >= 50 else 0.6
            return np.array(predictions), confidence
            
        except Exception as e:
            return np.full(steps, data.iloc[-1]), 0.4
    
    def _linear_trend_model(self, data, steps):
        """Linear trend extrapolation with improved robustness"""
        try:
            # Use appropriate window size
            window_size = min(30, max(5, len(data)))
            recent_data = data.tail(window_size)
            
            if len(recent_data) < 3:
                return np.full(steps, data.iloc[-1]), 0.4
            
            x = np.arange(len(recent_data))
            y = recent_data.values
            
            # Fit linear trend
            coeffs = np.polyfit(x, y, 1)
            slope, intercept = coeffs
            
            # Project trend forward
            predictions = []
            last_x = len(recent_data) - 1
            
            for i in range(1, steps + 1):
                pred = slope * (last_x + i) + intercept
                predictions.append(max(0.01, pred))  # Ensure positive
                
            # Calculate R-squared for confidence
            trend_line = slope * x + intercept
            ss_res = np.sum((y - trend_line) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            if ss_tot != 0:
                r_squared = max(0, 1 - (ss_res / ss_tot))
                confidence = max(0.4, min(0.8, r_squared))
            else:
                confidence = 0.5
            
            return np.array(predictions), confidence
            
        except Exception as e:
            return np.full(steps, data.iloc[-1]), 0.4
    
    def _seasonal_naive_model(self, data, steps):
        """Seasonal naive forecast with better handling"""
        try:
            # Use last few days as seasonal pattern
            pattern_length = min(5, len(data))
            seasonal_pattern = data.tail(pattern_length).values
            
            predictions = []
            for i in range(steps):
                # Cycle through the seasonal pattern
                seasonal_value = seasonal_pattern[i % len(seasonal_pattern)]
                
                # Add small random variation
                noise_std = data.std() * 0.02 if len(data) > 1 else 0
                noise = np.random.normal(0, noise_std)
                
                pred = max(0.01, seasonal_value + noise)
                predictions.append(pred)
                
            confidence = 0.55 if len(data) >= 20 else 0.45
            return np.array(predictions), confidence
            
        except Exception as e:
            return np.full(steps, data.iloc[-1]), 0.4
    
    def _exponential_smoothing_model(self, data, steps):
        """Simple exponential smoothing with error handling"""
        try:
            alpha = 0.3  # Smoothing parameter
            
            # Initialize with first value
            smoothed = [data.iloc[0]]
            
            # Calculate exponentially smoothed values
            for i in range(1, len(data)):
                s_t = alpha * data.iloc[i] + (1 - alpha) * smoothed[-1]
                smoothed.append(s_t)
                
            # Use last smoothed value for prediction
            last_smoothed = smoothed[-1]
            
            # Calculate simple trend
            if len(smoothed) >= 10:
                recent_smooth = smoothed[-10:]
                trend = (recent_smooth[-1] - recent_smooth[0]) / 9
            else:
                trend = 0
                
            predictions = []
            for i in range(steps):
                # Apply trend with diminishing effect
                trend_effect = trend * (0.9 ** i)
                pred = max(0.01, last_smoothed + trend_effect)
                predictions.append(pred)
                
            confidence = 0.65
            return np.array(predictions), confidence
            
        except Exception as e:
            return np.full(steps, data.iloc[-1]), 0.4
    
    def _simple_prediction(self, steps, symbol=None):
        """Fallback simple prediction for insufficient data"""
        current_price = 100.0  # Default price
        
        # Simple random walk with slight upward bias
        predictions = []
        price = current_price
        
        for i in range(steps):
            # Small upward bias with random variation
            change = np.random.normal(0.001, 0.02)  # 0.1% daily bias
            price = max(0.01, price * (1 + change))
            predictions.append(price)
            
        return {
            'predictions': np.array(predictions),
            'dates': pd.date_range(start=datetime.now() + timedelta(days=1), periods=steps, freq='D'),
            'confidence': 0.4,
            'method': 'Simple Random Walk (Fallback)',
            'current_price': current_price,
            'predicted_price': predictions[-1],
            'price_change_percent': ((predictions[-1] - current_price) / current_price) * 100,
            'individual_predictions': {},
            'individual_confidences': {},
            'data_points': 0,
            'symbol': symbol or 'Unknown',
            'volatility': 0.02
        }
    
    def get_prediction_summary(self, result):
        """Generate a human-readable summary of the prediction"""
        if not isinstance(result, dict) or 'price_change_percent' not in result:
            return "Prediction analysis unavailable"
            
        try:
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
            
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def validate_prediction(self, result):
        """Validate prediction results for quality checks"""
        try:
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
                if current_price > 0:
                    max_change = max(abs(p - current_price) / current_price for p in predictions)
                    checks['reasonable_prices'] = max_change < 0.5
                
                # Check for infinite or NaN values
                checks['no_infinite_values'] = not (np.any(np.isinf(predictions)) or np.any(np.isnan(predictions)))
            
            return checks, all(checks.values())
            
        except Exception as e:
            return {'validation_error': str(e)}, False
