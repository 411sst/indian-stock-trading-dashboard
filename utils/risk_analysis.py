import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

class RiskAnalyzer:
    """
    Advanced risk analysis tools for stock predictions and portfolio management
    """
    
    def __init__(self):
        self.risk_free_rate = 0.06  # 6% annual risk-free rate (typical for India)
        
    def calculate_var(self, returns, confidence_level=0.05, method='historical'):
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: pandas Series of stock returns
            confidence_level: VaR confidence level (default 5%)
            method: 'historical', 'parametric', or 'monte_carlo'
        """
        try:
            returns = returns.dropna()
            if len(returns) < 30:
                return {
                    'var_1d': 0.05,
                    'var_5d': 0.11,
                    'var_10d': 0.16,
                    'method': 'insufficient_data'
                }
            
            if method == 'historical':
                var_1d = np.percentile(returns, confidence_level * 100)
                
            elif method == 'parametric':
                mu = returns.mean()
                sigma = returns.std()
                var_1d = stats.norm.ppf(confidence_level, mu, sigma)
                
            elif method == 'monte_carlo':
                # Monte Carlo simulation
                mu = returns.mean()
                sigma = returns.std()
                simulated_returns = np.random.normal(mu, sigma, 10000)
                var_1d = np.percentile(simulated_returns, confidence_level * 100)
            
            # Scale to different time horizons
            var_5d = var_1d * np.sqrt(5)
            var_10d = var_1d * np.sqrt(10)
            
            return {
                'var_1d': abs(var_1d),
                'var_5d': abs(var_5d),
                'var_10d': abs(var_10d),
                'method': method,
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            st.warning(f"VaR calculation failed: {str(e)}")
            return {
                'var_1d': 0.05,
                'var_5d': 0.11,
                'var_10d': 0.16,
                'method': 'fallback'
            }
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=None):
        """Calculate Sharpe ratio for risk-adjusted returns"""
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate / 252  # Daily risk-free rate
                
            returns = returns.dropna()
            if len(returns) < 30:
                return 0.0
                
            excess_returns = returns - risk_free_rate
            sharpe = excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
            
            # Annualize
            return sharpe * np.sqrt(252)
            
        except Exception:
            return 0.0
    
    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        try:
            if len(prices) < 2:
                return 0.0
                
            # Calculate cumulative returns
            cumulative = (1 + prices.pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            return abs(drawdown.min())
            
        except Exception:
            return 0.0
    
    def calculate_beta(self, stock_returns, market_returns):
        """Calculate stock beta relative to market"""
        try:
            stock_returns = stock_returns.dropna()
            market_returns = market_returns.dropna()
            
            # Align dates
            common_dates = stock_returns.index.intersection(market_returns.index)
            if len(common_dates) < 30:
                return 1.0  # Default beta
                
            stock_aligned = stock_returns.loc[common_dates]
            market_aligned = market_returns.loc[common_dates]
            
            covariance = np.cov(stock_aligned, market_aligned)[0][1]
            market_variance = np.var(market_aligned)
            
            beta = covariance / market_variance if market_variance != 0 else 1.0
            return beta
            
        except Exception:
            return 1.0
    
    def volatility_regime_detection(self, returns, window=20):
        """Detect current volatility regime"""
        try:
            returns = returns.dropna()
            if len(returns) < window * 2:
                return {'regime': 'normal', 'current_vol': 0.02, 'historical_vol': 0.02}
            
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            
            current_vol = rolling_vol.iloc[-1]
            historical_vol = rolling_vol.mean()
            
            # Classify regime
            vol_ratio = current_vol / historical_vol if historical_vol != 0 else 1
            
            if vol_ratio > 1.5:
                regime = 'high_volatility'
            elif vol_ratio < 0.7:
                regime = 'low_volatility'
            else:
                regime = 'normal'
                
            return {
                'regime': regime,
                'current_vol': current_vol,
                'historical_vol': historical_vol,
                'vol_ratio': vol_ratio
            }
            
        except Exception:
            return {'regime': 'normal', 'current_vol': 0.02, 'historical_vol': 0.02}
    
    def stress_test_scenarios(self, current_price, predictions):
        """Generate stress test scenarios"""
        try:
            scenarios = {
                'bull_market': predictions * 1.2,  # 20% better than predicted
                'bear_market': predictions * 0.8,  # 20% worse than predicted
                'market_crash': predictions * 0.7,  # 30% crash scenario
                'black_swan': predictions * 0.5,   # 50% extreme event
                'base_case': predictions            # Original prediction
            }
            
            scenario_returns = {}
            for name, prices in scenarios.items():
                final_return = ((prices[-1] - current_price) / current_price) * 100
                scenario_returns[name] = {
                    'final_price': prices[-1],
                    'total_return': final_return,
                    'prices': prices
                }
            
            return scenario_returns
            
        except Exception as e:
            return {'error': str(e)}
    
    def risk_metrics_dashboard(self, stock_data, predictions):
        """Generate comprehensive risk metrics"""
        try:
            returns = stock_data.pct_change().dropna()
            current_price = stock_data.iloc[-1]
            
            # Calculate all risk metrics
            var_metrics = self.calculate_var(returns)
            sharpe = self.calculate_sharpe_ratio(returns)
            max_dd = self.calculate_max_drawdown(stock_data)
            vol_regime = self.volatility_regime_detection(returns)
            stress_scenarios = self.stress_test_scenarios(current_price, predictions)
            
            # Prediction risk assessment
            pred_volatility = np.std(predictions) / np.mean(predictions) if np.mean(predictions) != 0 else 0
            
            return {
                'var_metrics': var_metrics,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'volatility_regime': vol_regime,
                'stress_scenarios': stress_scenarios,
                'prediction_volatility': pred_volatility,
                'risk_score': self._calculate_risk_score(var_metrics, vol_regime, max_dd)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_risk_score(self, var_metrics, vol_regime, max_drawdown):
        """Calculate overall risk score (0-100, higher = riskier)"""
        try:
            # VaR component (0-40 points)
            var_score = min(40, var_metrics['var_1d'] * 100 * 4)
            
            # Volatility regime component (0-30 points)
            vol_scores = {'low_volatility': 10, 'normal': 20, 'high_volatility': 30}
            vol_score = vol_scores.get(vol_regime['regime'], 20)
            
            # Max drawdown component (0-30 points)
            dd_score = min(30, max_drawdown * 100)
            
            total_score = var_score + vol_score + dd_score
            return min(100, max(0, total_score))
            
        except Exception:
            return 50  # Default medium risk

class PerformanceAnalyzer:
    """
    Performance analysis and backtesting tools
    """
    
    def __init__(self):
        self.benchmark_symbols = {
            'NIFTY': '^NSEI',
            'SENSEX': '^BSESN',
            'NIFTY_BANK': '^NSEBANK'
        }
    
    def calculate_performance_metrics(self, prices, benchmark_prices=None):
        """Calculate comprehensive performance metrics"""
        try:
            returns = prices.pct_change().dropna()
            
            metrics = {
                'total_return': ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100,
                'annualized_return': self._annualized_return(prices),
                'volatility': returns.std() * np.sqrt(252) * 100,
                'sharpe_ratio': self._sharpe_ratio(returns),
                'max_drawdown': self._max_drawdown(prices) * 100,
                'win_rate': (returns > 0).mean() * 100,
                'avg_gain': returns[returns > 0].mean() * 100,
                'avg_loss': returns[returns < 0].mean() * 100,
                'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else 0
            }
            
            if benchmark_prices is not None:
                benchmark_returns = benchmark_prices.pct_change().dropna()
                metrics['beta'] = self._calculate_beta(returns, benchmark_returns)
                metrics['alpha'] = metrics['annualized_return'] - (0.06 + metrics['beta'] * (benchmark_returns.mean() * 252 * 100 - 6))
                metrics['information_ratio'] = self._information_ratio(returns, benchmark_returns)
            
            return metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def _annualized_return(self, prices):
        """Calculate annualized return"""
        try:
            total_days = (prices.index[-1] - prices.index[0]).days
            total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
            return ((1 + total_return) ** (365.25 / total_days) - 1) * 100
        except:
            return 0.0
    
    def _sharpe_ratio(self, returns, risk_free_rate=0.06):
        """Calculate Sharpe ratio"""
        try:
            daily_rf = risk_free_rate / 252
            excess_returns = returns - daily_rf
            return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0
        except:
            return 0.0
    
    def _max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        try:
            cumulative = prices / prices.iloc[0]
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except:
            return 0.0
    
    def _calculate_beta(self, stock_returns, market_returns):
        """Calculate beta"""
        try:
            covariance = np.cov(stock_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)
            return covariance / market_variance if market_variance != 0 else 1.0
        except:
            return 1.0
    
    def _information_ratio(self, portfolio_returns, benchmark_returns):
        """Calculate information ratio"""
        try:
            active_returns = portfolio_returns - benchmark_returns
            return active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() != 0 else 0
        except:
            return 0.0

def create_risk_dashboard(risk_metrics):
    """Create a visual risk dashboard"""
    try:
        # Risk Score Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_metrics['risk_score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        fig_gauge.update_layout(height=300, template='plotly_dark')
        
        return fig_gauge
        
    except Exception as e:
        st.error(f"Risk dashboard creation failed: {str(e)}")
        return None

def create_stress_test_chart(stress_scenarios, current_price):
    """Create stress test visualization"""
    try:
        if 'error' in stress_scenarios:
            return None
            
        scenario_names = list(stress_scenarios.keys())
        final_prices = [stress_scenarios[name]['final_price'] for name in scenario_names]
        returns = [stress_scenarios[name]['total_return'] for name in scenario_names]
        
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        fig = go.Figure(data=[
            go.Bar(x=scenario_names, y=returns, marker_color=colors)
        ])
        
        fig.update_layout(
            title="Stress Test Scenarios",
            xaxis_title="Scenario",
            yaxis_title="Total Return (%)",
            template='plotly_dark',
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Stress test chart creation failed: {str(e)}")
        return None
