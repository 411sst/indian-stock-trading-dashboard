# utils/portfolio_manager.py

def calculate_detailed_portfolio_value(portfolio):
    """Calculate detailed portfolio value including returns and risk metrics"""
    if not portfolio:
        return {
            "total_value": 0,
            "p_and_l": 0,
            "return_percentage": 0,
            "holdings": []
        }
    
    total_value = 0
    total_investment = 0
    holdings = []
    
    for holding in portfolio:
        try:
            stock = yf.Ticker(holding['symbol'])
            current_price = stock.history(period="1d")['Close'][-1]
        except:
            current_price = holding['buy_price']  # Use buy price if unable to fetch
            
        quantity = holding['quantity']
        buy_price = holding['buy_price']
        
        current_value = quantity * current_price
        investment_value = quantity * buy_price
        p_and_l = current_value - investment_value
        return_percentage = (p_and_l / investment_value) * 100 if investment_value > 0 else 0
        
        total_value += current_value
        total_investment += investment_value
        
        holdings.append({
            "symbol": holding['symbol'],
            "quantity": quantity,
            "buy_price": buy_price,
            "current_price": current_price,
            "investment_value": investment_value,
            "current_value": current_value,
            "p_and_l": p_and_l,
            "return_percentage": return_percentage
        })
    
    p_and_l = total_value - total_investment
    return_percentage = (p_and_l / total_investment) * 100 if total_investment > 0 else 0
    
    return {
        "total_value": total_value,
        "p_and_l": p_and_l,
        "return_percentage": return_percentage,
        "holdings": holdings
    }

def create_asset_allocation_chart(portfolio_value):
    """Create a pie chart showing asset allocation"""
    if not portfolio_value or not portfolio_value['holdings']:
        return None
        
    labels = [holding['symbol'] for holding in portfolio_value['holdings']]
    values = [holding['current_value'] for holding in portfolio_value['holdings']]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(
        title="Asset Allocation",
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_performance_comparison_chart(portfolio_history, benchmark_history):
    """Create a chart comparing portfolio performance against benchmark"""
    if not portfolio_history or not benchmark_history:
        return None
        
    df_portfolio = pd.DataFrame(list(portfolio_history.items()), columns=['Date', 'Portfolio Value'])
    df_benchmark = pd.DataFrame(list(benchmark_history.items()), columns=['Date', 'Benchmark Value'])
    
    # Normalize both series to start at 100
    initial_portfolio = df_portfolio.iloc[0]['Portfolio Value']
    initial_benchmark = df_benchmark.iloc[0]['Benchmark Value']
    
    df_portfolio['Portfolio Index'] = (df_portfolio['Portfolio Value'] / initial_portfolio) * 100
    df_benchmark['Benchmark Index'] = (df_benchmark['Benchmark Value'] / initial_benchmark) * 100
    
    # Merge dataframes
    df = pd.merge(df_portfolio, df_benchmark, on='Date', how='outer')
    df = df.sort_values('Date')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Portfolio Index'], name='Your Portfolio'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Benchmark Index'], name='NIFTY 50 Benchmark'))
    
    fig.update_layout(
        title="Portfolio vs Market Performance",
        xaxis_title="Date",
        yaxis_title="Normalized Value (Base: 100)",
        template='plotly_dark'
    )
    
    return fig
