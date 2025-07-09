# pages/portfolio_tracker.py
import yfinance as yf
from utils.indian_stocks import INDIAN_STOCKS
import streamlit as st
import pandas as pd
from utils.data_fetcher import calculate_portfolio_value
from utils.portfolio_manager import load_sample_portfolio, add_to_portfolio, remove_from_portfolio, export_to_csv

def portfolio_tracker_page(mode):
    st.header("Portfolio Tracker")
    
    # Initialize session state for portfolio if not exists
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = load_sample_portfolio()
    
    # Portfolio Actions
    action = st.radio("Select Action", ["View Portfolio", "Add Holding", "Import CSV"])
    
    if action == "Add Holding":
        st.subheader("Add New Holding")
        with st.form("add_holding_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                symbol = st.selectbox("Stock", options=list(INDIAN_STOCKS.keys()))
            
            with col2:
                quantity = st.number_input("Quantity", min_value=1, value=1)
            
            col3, col4 = st.columns(2)
            
            with col3:
                buy_price = st.number_input("Buy Price (₹)", min_value=0.0, value=100.0)
            
            with col4:
                buy_date = st.date_input("Buy Date")
            
            submit_button = st.form_submit_button("Add to Portfolio")
            
            if submit_button:
                new_entry = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "buy_price": buy_price,
                    "buy_date": str(buy_date)
                }
                st.session_state.portfolio = add_to_portfolio(st.session_state.portfolio, new_entry)
                st.success(f"Added {quantity} shares of {symbol} to your portfolio!")
    
    elif action == "Import CSV":
        st.subheader("Import Portfolio from CSV")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_columns = ["symbol", "quantity", "buy_price", "buy_date"]
                
                if all(col in df.columns for col in required_columns):
                    st.session_state.portfolio = df.to_dict('records')
                    st.success("Portfolio imported successfully!")
                else:
                    st.error(f"CSV must contain these columns: {', '.join(required_columns)}")
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    # View Portfolio
    st.subheader("Your Portfolio")
    
    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Add some holdings to get started!")
        return
    
    # Calculate portfolio value
    portfolio_value = calculate_portfolio_value(st.session_state.portfolio)
    
    # Summary Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Portfolio Value", f"₹{portfolio_value['total_value']:.2f}")
    col2.metric(
        "Profit/Loss", 
        f"₹{portfolio_value['p_and_l']:.2f}", 
        f"{portfolio_value['return_percentage']:.2f}%"
    )
    col3.metric("Benchmark", "NIFTY 50", "+0.66%")
    
    # Portfolio Table
    st.markdown("---")
    st.subheader("Portfolio Holdings")
    
    # Convert portfolio to DataFrame for display
    df = pd.DataFrame(st.session_state.portfolio)
    
    # Get current prices
    current_prices = {}
    for symbol in df['symbol'].unique():
        try:
            stock = yf.Ticker(symbol)
            current_prices[symbol] = stock.history(period="1d")['Close'][-1]
        except:
            current_prices[symbol] = "N/A"
    
    # Add current price and P&L to DataFrame
    df['current_price'] = df['symbol'].map(current_prices)
    df['current_value'] = df['quantity'] * df['current_price']
    df['investment_value'] = df['quantity'] * df['buy_price']
    df['p_and_l'] = df['current_value'] - df['investment_value']
    df['return_percentage'] = (df['p_and_l'] / df['investment_value']) * 100
    
    # Format numbers
    df_display = df.copy()
    df_display['buy_price'] = df_display['buy_price'].apply(lambda x: f"₹{x:.2f}")
    df_display['current_price'] = df_display['current_price'].apply(lambda x: f"₹{x:.2f}" if isinstance(x, float) else x)
    df_display['p_and_l'] = df_display['p_and_l'].apply(lambda x: f"₹{x:.2f}")
    df_display['return_percentage'] = df_display['return_percentage'].apply(lambda x: f"{x:.2f}%")
    
    # Display table with actions
    for i, row in df_display.iterrows():
        col1, col2, col3, col4, col5, col6, col7 = st.columns([3, 2, 2, 2, 2, 2, 1])
        
        col1.text(row['symbol'])
        col2.text(row['buy_price'])
        col3.text(row['current_price'])
        col4.text(row['p_and_l'])
        col5.text(row['return_percentage'])
        col6.text(row['quantity'])
        
        if col7.button("Delete", key=f"delete_{i}_{row['symbol']}"):
            st.session_state.portfolio = remove_from_portfolio(st.session_state.portfolio, i)
            st.experimental_rerun()
    
    # Export Portfolio
    st.markdown("---")
    st.subheader("Export Portfolio")
    csv = export_to_csv(st.session_state.portfolio)
    
    st.download_button(
        label="Download Portfolio as CSV",
        data=csv,
        file_name="my_portfolio.csv",
        mime="text/csv",
    )
    
    # Asset Allocation Chart
    st.markdown("---")
    st.subheader("Asset Allocation")
    
    # Placeholder for pie chart (would require matplotlib or plotly)
    st.info("Asset allocation visualization would appear here in a full implementation.")
    
    # Coming Soon Banner
    st.markdown("---")
    st.markdown("""
    <div style='padding: 10px; background-color: #e6f7ff; border-left: 4px solid #1a73e8;'>
        <strong>Coming Soon:</strong> Auto-sync with brokers functionality for automatic portfolio updates.
    </div>
    """, unsafe_allow_html=True)