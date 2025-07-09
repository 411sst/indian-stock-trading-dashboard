# utils/portfolio_manager.py
import pandas as pd
import streamlit as st

def load_sample_portfolio():
    """Load sample portfolio data from CSV"""
    try:
        df = pd.read_csv("sample_portfolio.csv")
        return df.to_dict('records')
    except Exception as e:
        st.error(f"Error loading sample portfolio: {str(e)}")
        return []

def add_to_portfolio(portfolio, new_entry):
    """Add a new holding to the portfolio"""
    portfolio.append(new_entry)
    return portfolio

def remove_from_portfolio(portfolio, index):
    """Remove a holding from the portfolio by index"""
    if 0 <= index < len(portfolio):
        portfolio.pop(index)
    return portfolio

def update_portfolio(portfolio, index, updated_entry):
    """Update a specific holding in the portfolio"""
    if 0 <= index < len(portfolio):
        portfolio[index] = updated_entry
    return portfolio

def export_to_csv(portfolio):
    """Export portfolio to CSV format"""
    df = pd.DataFrame(portfolio)
    return df.to_csv(index=False).encode('utf-8')

def create_pie_chart(portfolio_value):
    """Create a pie chart showing asset allocation"""
    if not portfolio_value:
        return None
        
    labels = [holding['symbol'] for holding in portfolio_value['holdings']]
    values = [holding['current_value'] for holding in portfolio_value['holdings']]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(
        title="Asset Allocation",
        template='plotly_dark'
    )
    
    return fig

def create_performance_chart(portfolio_history):
    """Create a chart showing portfolio performance over time"""
    if not portfolio_history:
        return None
        
    dates = list(portfolio_history.keys())
    values = list(portfolio_history.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers'))
    
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Value (INR)",
        template='plotly_dark'
    )
    
    return fig