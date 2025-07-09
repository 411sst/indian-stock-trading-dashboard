# utils/technical_analysis.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

def create_candlestick_chart(df, title="Price Chart"):
    """Create an interactive candlestick chart with technical indicators"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, subplot_titles=('', 'Volume'),
                       row_width=[0.2, 0.7])

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

    # Volume
    colors = ['green' if row['Open'] - row['Close'] >= 0 
             else 'red' for _, row in df.iterrows()]
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color=colors
    ), row=2, col=1)

    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600,
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
    fig.update_yaxes(title_text="Volume", row=2, col=1)

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
            "reason": f"{symbol} is oversold (RSI: {latest['RSI']:.2f})"
        })
    elif latest['RSI'] > 70:
        signals.append({
            "type": "SELL",
            "reason": f"{symbol} is overbought (RSI: {latest['RSI']:.2f})"
        })
    
    # MACD Signal
    if latest['MACD'] > latest['MACD_signal'] and previous['MACD'] <= previous['MACD_signal']:
        signals.append({
            "type": "BUY",
            "reason": "MACD bullish crossover detected"
        })
    elif latest['MACD'] < latest['MACD_signal'] and previous['MACD'] >= previous['MACD_signal']:
        signals.append({
            "type": "SELL",
            "reason": "MACD bearish crossover detected"
        })
    
    return signals