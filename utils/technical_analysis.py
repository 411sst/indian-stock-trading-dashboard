# utils/technical_analysis.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

def create_candlestick_chart(df, title="Price Chart"):
    """Create an interactive candlestick chart with technical indicators"""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       subplot_titles=('Price & Indicators', 'RSI', 'MACD', 'Volume'),
                       row_heights=[0.5, 0.2, 0.2, 0.1])

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ), row=1, col=1)

    # Moving Averages
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA20'],
        name='SMA 20',
        line=dict(color='orange', width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA50'],
        name='EMA 50',
        line=dict(color='blue', width=1)
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_upper'],
        name='BB Upper',
        line=dict(color='gray', width=1, dash='dot')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_lower'],
        name='BB Lower',
        line=dict(color='gray', width=1, dash='dot'),
        fill='tonexty'
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple')
    ), row=2, col=1)
    
    # Add overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # MACD
    colors = ['green' if val >= 0 else 'red' for val in df['MACD'] - df['MACD_signal']]
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=(df['MACD'] - df['MACD_signal']),
        name='MACD Histogram',
        marker_color=colors
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue')
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD_signal'],
        name='Signal',
        line=dict(color='orange')
    ), row=3, col=1)

    # Volume
    volume_colors = ['green' if row['Open'] - row['Close'] >= 0 
                    else 'red' for _, row in df.iterrows()]
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color=volume_colors
    ), row=4, col=1)

    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)

    return fig

def create_rsi_chart(df):
    """Create RSI indicator chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple')
    ))
    
    # Add overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
    
    fig.update_layout(
        title="Relative Strength Index (RSI)",
        yaxis_range=[0,100],
        template='plotly_dark',
        height=300
    )
    
    return fig

def create_macd_chart(df):
    """Create MACD indicator chart"""
    fig = go.Figure()
    
    # MACD Line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue')
    ))
    
    # Signal Line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD_signal'],
        name='Signal',
        line=dict(color='orange')
    ))
    
    # Histogram
    colors = ['green' if val >= 0 else 'red' for val in df['MACD'] - df['MACD_signal']]
    fig.add_trace(go.Bar(
        x=df.index,
        y=(df['MACD'] - df['MACD_signal']),
        name='Histogram',
        marker_color=colors
    ))
    
    fig.update_layout(
        title="MACD Indicator",
        template='plotly_dark',
        height=300,
        showlegend=False
    )
    
    return fig

def generate_buy_sell_signals(df, symbol):
    """Generate buy/sell signals based on technical analysis"""
    if df.empty or len(df) < 50:
        return []
    
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    signals = []
    
    # RSI Signal
    if latest['RSI'] < 30:
        signals.append({
            "type": "BUY",
            "reason": f"{symbol} is oversold (RSI: {latest['RSI']:.2f})",
            "strength": "Strong" if latest['RSI'] < 25 else "Moderate"
        })
    elif latest['RSI'] > 70:
        signals.append({
            "type": "SELL",
            "reason": f"{symbol} is overbought (RSI: {latest['RSI']:.2f})",
            "strength": "Strong" if latest['RSI'] > 75 else "Moderate"
        })
    
    # MACD Signal
    if latest['MACD'] > latest['MACD_signal'] and previous['MACD'] <= previous['MACD_signal']:
        signals.append({
            "type": "BUY",
            "reason": "MACD bullish crossover detected",
            "strength": "Moderate"
        })
    elif latest['MACD'] < latest['MACD_signal'] and previous['MACD'] >= previous['MACD_signal']:
        signals.append({
            "type": "SELL",
            "reason": "MACD bearish crossover detected",
            "strength": "Moderate"
        })
    
    # Moving Average Signal
    if latest['SMA20'] > latest['EMA50'] and previous['SMA20'] <= previous['EMA50']:
        signals.append({
            "type": "BUY",
            "reason": "SMA20 crossed above EMA50 (Golden Cross)",
            "strength": "Strong"
        })
    elif latest['SMA20'] < latest['EMA50'] and previous['SMA20'] >= previous['EMA50']:
        signals.append({
            "type": "SELL",
            "reason": "SMA20 crossed below EMA50 (Death Cross)",
            "strength": "Strong"
        })
    
    # Bollinger Band Signal
    if latest['Close'] < latest['BB_lower']:
        signals.append({
            "type": "BUY",
            "reason": "Price below lower Bollinger Band (potential reversal)",
            "strength": "Moderate"
        })
    elif latest['Close'] > latest['BB_upper']:
        signals.append({
            "type": "SELL",
            "reason": "Price above upper Bollinger Band (potential reversal)",
            "strength": "Moderate"
        })
    
    # Volume Confirmation
    vol_ratio = latest['Volume'] / latest['Vol_MA'] if latest['Vol_MA'] > 0 else 1
    if vol_ratio > 1.5:
        for signal in signals:
            signal['volume_confirmation'] = True
            signal['reason'] += f" (High volume: {vol_ratio:.1f}x average)"
    
    return signals

def calculate_support_resistance(df, window=20):
    """Calculate dynamic support and resistance levels"""
    if df.empty or len(df) < window * 2:
        return [], []
    
    highs = df['High'].rolling(window=window).max()
    lows = df['Low'].rolling(window=window).min()
    
    # Find recent support and resistance levels
    recent_data = df.tail(window * 2)
    resistance_levels = recent_data['High'].nlargest(3).tolist()
    support_levels = recent_data['Low'].nsmallest(3).tolist()
    
    # Remove duplicates and sort
    resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
    support_levels = sorted(list(set(support_levels)))
    
    return support_levels, resistance_levels

def analyze_trend(df, period=20):
    """Analyze the overall trend of the stock"""
    if df.empty or len(df) < period:
        return "Unknown"
    
    recent_data = df.tail(period)
    
    # Calculate slope of price movement
    prices = recent_data['Close'].values
    x = np.arange(len(prices))
    slope = np.polyfit(x, prices, 1)[0]
    
    # Calculate moving average trend
    sma_slope = np.polyfit(x, recent_data['SMA20'].values, 1)[0]
    
    if slope > 0 and sma_slope > 0:
        return "Uptrend"
    elif slope < 0 and sma_slope < 0:
        return "Downtrend"
    else:
        return "Sideways"

def calculate_volatility(df, period=20):
    """Calculate historical volatility"""
    if df.empty or len(df) < period:
        return 0
    
    returns = df['Close'].pct_change().dropna()
    volatility = returns.rolling(window=period).std().iloc[-1] * np.sqrt(252) * 100
    
    return volatility if not np.isnan(volatility) else 0
