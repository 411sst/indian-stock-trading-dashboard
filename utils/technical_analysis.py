# utils/technical_analysis.py

def create_candlestick_chart(df, title="Price Chart"):
    """Create an interactive candlestick chart with technical indicators"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, subplot_titles=('', '', 'Volume'),
                       row_width=[0.2, 0.3, 0.5])

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
