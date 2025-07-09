# app.py
import streamlit as st
from components.market_overview_module import market_overview_page
from components.stock_analysis_module import stock_analysis_page  
from components.portfolio_tracker_module import portfolio_tracker_page
from components.news_sentiment_module import news_sentiment_page
from utils.indian_stocks import INDIAN_STOCKS

# Set page config
st.set_page_config(
    page_title="Indian Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .reportview-container {
        background-color: #0e1117;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #1a1c23;
    }
    .metric-container {
        background-color: #1a1c23;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton button {
        background-color: #1a73e8;
        color: white;
    }
    .stTextInput input {
        background-color: #1a1c23;
        color: white;
    }
    .stSelectbox select {
        background-color: #1a1c23;
        color: white;
    }
    .stNumberInput input {
        background-color: #1a1c23;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'mode' not in st.session_state:
    st.session_state.mode = "Beginner"

# Sidebar
with st.sidebar:
    st.title("📈 Indian Stock Dashboard")
    st.markdown("---")
    
    # Mode Toggle
    st.session_state.mode = st.selectbox(
        "Select Mode",
        ["Beginner", "Pro"],
        index=0 if st.session_state.mode == "Beginner" else 1
    )
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### Navigation")
    nav_options = ["Market Overview", "Stock Analysis", "Portfolio Tracker", "News & Sentiment"]
    selected_nav = st.radio("Navigation", nav_options)
    
    st.markdown("---")
    
    # About Section
    st.markdown("### About")
    st.markdown("A comprehensive Indian stock market trading dashboard built with Streamlit.")
    st.markdown("Tracks NIFTY 50, SENSEX, and major Indian stocks.")
    st.markdown("Includes technical analysis, sentiment analysis, and portfolio tracking.")

# Main content area
if selected_nav == "Market Overview":
    market_overview_page(st.session_state.mode)
elif selected_nav == "Stock Analysis":
    stock_analysis_page(st.session_state.mode)
elif selected_nav == "Portfolio Tracker":
    portfolio_tracker_page(st.session_state.mode)
elif selected_nav == "News & Sentiment":
    news_sentiment_page(st.session_state.mode)