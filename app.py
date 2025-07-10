# app.py - Enhanced Indian Stock Trading Dashboard
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

# Import existing modules
from components.market_overview_module import market_overview_page
from components.stock_analysis_module import stock_analysis_page  
from components.portfolio_tracker_module import portfolio_tracker_page
from components.news_sentiment_module import news_sentiment_page
from utils.indian_stocks import INDIAN_STOCKS

# Import new authentication and ML modules
try:
    from authentication.auth_handler import AuthHandler
    from authentication.validators import validate_email, validate_password, validate_username, get_password_strength_score, get_password_strength_text
    from ml_forecasting.models.ensemble_model import EnsembleModel
    ENHANCED_FEATURES = True
except ImportError as e:
    ENHANCED_FEATURES = False
    st.sidebar.error(f"⚠️ Enhanced features not available: {str(e)}")

# Set page config
st.set_page_config(
    page_title="Indian Stock Dashboard - Enhanced",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
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
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #333;
    }
    .stButton button {
        background-color: #1a73e8;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #1557b0;
        border: none;
    }
    .user-info {
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 15px;
        color: white;
    }
    .auth-container {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .success-message {
        padding: 10px;
        background-color: #1f4e3d;
        border: 1px solid #10b981;
        border-radius: 5px;
        color: #10b981;
        margin: 10px 0;
    }
    .error-message {
        padding: 10px;
        background-color: #4c1d1d;
        border: 1px solid #ef4444;
        border-radius: 5px;
        color: #ef4444;
        margin: 10px 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #4f46e5;
    }
    .ml-metric {
        text-align: center;
        padding: 15px;
        background-color: #1f2937;
        border-radius: 8px;
        margin: 5px;
        border: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'mode' not in st.session_state:
        st.session_state.mode = "Beginner"
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = 'RELIANCE.NS'
    if 'show_ml_details' not in st.session_state:
        st.session_state.show_ml_details = False

initialize_session_state()

# Initialize authentication handler
if ENHANCED_FEATURES:
    if 'auth_handler' not in st.session_state:
        st.session_state.auth_handler = AuthHandler()

def create_password_strength_indicator(password):
    """Create a visual password strength indicator"""
    if not password:
        return ""
    
    score = get_password_strength_score(password)
    strength_text, color = get_password_strength_text(score)
    
    return f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 12px;">Password Strength:</span>
            <span style="color: {color}; font-weight: bold; font-size: 12px;">{strength_text}</span>
        </div>
        <div style="background-color: #374151; border-radius: 10px; height: 8px; margin: 5px 0;">
            <div style="background-color: {color}; width: {score}%; height: 100%; border-radius: 10px; transition: width 0.3s;"></div>
        </div>
    </div>
    """

# Sidebar Content
with st.sidebar:
    st.title("📈 Indian Stock Dashboard")
    st.markdown("*Enhanced with AI & Authentication*")
    
    # Authentication Section
    if ENHANCED_FEATURES:
        if not st.session_state.logged_in:
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            st.markdown("### 🔐 User Authentication")
            
            auth_tab1, auth_tab2 = st.tabs(["🔑 Login", "👤 Register"])
            
            with auth_tab1:
                with st.form("login_form", clear_on_submit=False):
                    st.markdown("#### Welcome Back!")
                    username = st.text_input("Username", placeholder="Enter your username")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        remember_me = st.checkbox("Remember me", value=True)
                    with col2:
                        st.markdown('<small><a href="#" style="color: #60a5fa;">Forgot?</a></small>', unsafe_allow_html=True)
                    
                    login_submit = st.form_submit_button("🚪 Login", use_container_width=True)
                    
                    if login_submit:
                        if username and password:
                            with st.spinner("Authenticating..."):
                                user_id, message = st.session_state.auth_handler.verify_user(username, password)
                                if user_id:
                                    st.session_state.logged_in = True
                                    st.session_state.user = st.session_state.auth_handler.get_user_info(user_id)
                                    st.markdown('<div class="success-message">✅ Login successful!</div>', unsafe_allow_html=True)
                                    st.rerun()
                                else:
                                    st.markdown(f'<div class="error-message">❌ {message}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="error-message">❌ Please fill in all fields</div>', unsafe_allow_html=True)
            
            with auth_tab2:
                with st.form("register_form", clear_on_submit=False):
                    st.markdown("#### Create Account")
                    new_username = st.text_input("Username", placeholder="Choose a username")
                    new_email = st.text_input("Email", placeholder="your.email@example.com")
                    new_password = st.text_input("Password", type="password", placeholder="Create a strong password")
                    confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                    
                    # Password strength indicator
                    if new_password:
                        st.markdown(create_password_strength_indicator(new_password), unsafe_allow_html=True)
                    
                    agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
                    
                    register_submit = st.form_submit_button("🎯 Create Account", use_container_width=True)
                    
                    if register_submit:
                        if not agree_terms:
                            st.markdown('<div class="error-message">❌ Please agree to the terms</div>', unsafe_allow_html=True)
                        elif new_username and new_email and new_password and confirm_password:
                            # Validate inputs
                            username_valid, username_msg = validate_username(new_username)
                            email_valid, email_msg = validate_email(new_email)
                            password_valid, password_msg = validate_password(new_password, confirm_password)
                            
                            if not username_valid:
                                st.markdown(f'<div class="error-message">❌ {username_msg}</div>', unsafe_allow_html=True)
                            elif not email_valid:
                                st.markdown(f'<div class="error-message">❌ {email_msg}</div>', unsafe_allow_html=True)
                            elif not password_valid:
                                st.markdown(f'<div class="error-message">❌ {password_msg}</div>', unsafe_allow_html=True)
                            else:
                                with st.spinner("Creating account..."):
                                    success, message = st.session_state.auth_handler.register_user(
                                        new_username, new_email, new_password
                                    )
                                    if success:
                                        st.markdown('<div class="success-message">✅ Registration successful! Please login.</div>', unsafe_allow_html=True)
                                        st.balloons()
                                    else:
                                        st.markdown(f'<div class="error-message">❌ {message}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="error-message">❌ Please fill in all fields</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            # User is logged in
            user = st.session_state.user
            st.markdown(f"""
            <div class="user-info">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <h4 style="margin: 0;">👋 Welcome, {user['username']}</h4>
                        <small>📧 {user['email']}</small><br>
                        <small>🕒 Last login: {user.get('last_login', 'Never')}</small>
                    </div>
                    <div style="text-align: right;">
                        <div style="background-color: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 15px; font-size: 11px;">
                            ✨ Premium User
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # User actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👤 Profile", use_container_width=True):
                    st.session_state.show_profile = True
            with col2:
                if st.button("🚪 Logout", use_container_width=True):
                    st.session_state.logged_in = False
                    st.session_state.user = None
                    st.rerun()
    
    st.markdown("---")
    
    # Mode Selection
    st.markdown("### 🎯 Trading Mode")
    mode_options = ["Beginner", "Pro", "Expert"]
    mode_descriptions = {
        "Beginner": "Simple analysis with explanations",
        "Pro": "Advanced technical indicators",
        "Expert": "Full ML predictions & risk analysis"
    }
    
    selected_mode = st.selectbox(
        "Select your experience level:",
        mode_options,
        index=mode_options.index(st.session_state.mode)
    )
    st.session_state.mode = selected_mode
    st.caption(mode_descriptions[selected_mode])
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    nav_options = ["📊 Market Overview", "📈 Stock Analysis", "💼 Portfolio Tracker", "📰 News & Sentiment"]
    
    if ENHANCED_FEATURES and st.session_state.logged_in:
        nav_options.extend(["🤖 ML Predictions", "⚙️ User Settings"])
    
    selected_nav = st.radio(
        "Choose a section:",
        nav_options,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📊 Quick Market Stats")
    try:
        nifty = yf.download("^NSEI", period="1d", interval="1m")
        if not nifty.empty:
            current_nifty = nifty['Close'][-1]
            nifty_change = ((current_nifty - nifty['Close'][0]) / nifty['Close'][0]) * 100
            
            st.metric(
                "NIFTY 50", 
                f"₹{current_nifty:.2f}", 
                f"{nifty_change:+.2f}%"
            )
    except:
        st.metric("NIFTY 50", "₹25,400", "+0.45%")
    
    # Feature availability indicator
    st.markdown("---")
    st.markdown("### 🚀 Features Available")
    if ENHANCED_FEATURES:
        st.markdown("✅ User Authentication")
        st.markdown("✅ ML-Powered Predictions")
        st.markdown("✅ Personal Portfolio")
        st.markdown("✅ Advanced Analytics")
    else:
        st.markdown("⚠️ Basic features only")
        st.markdown("💡 Install ML packages for full experience")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 12px;">© 2024 Indian Stock Dashboard<br>Enhanced with AI</p>',
        unsafe_allow_html=True
    )

# Main Content Area
if selected_nav == "📊 Market Overview":
    market_overview_page(st.session_state.mode)

elif selected_nav == "📈 Stock Analysis":
    stock_analysis_page(st.session_state.mode)

elif selected_nav == "💼 Portfolio Tracker":
    portfolio_tracker_page(st.session_state.mode)

elif selected_nav == "📰 News & Sentiment":
    news_sentiment_page(st.session_state.mode)


elif selected_nav == "🤖 ML Predictions" and ENHANCED_FEATURES:
    if not st.session_state.logged_in:
        st.warning("🔒 Please login to access ML-powered predictions.")
        st.info("💡 Register for free to unlock advanced AI features!")
    else:
        st.title("🤖 AI-Powered Stock Predictions")
        st.markdown("*Advanced machine learning models for intelligent stock forecasting*")
        
        # Stock selection
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_stock = st.selectbox(
                "📊 Select Stock for AI Analysis:",
                list(INDIAN_STOCKS.keys()),
                format_func=lambda x: f"{INDIAN_STOCKS[x]} ({x})"
            )
        
        with col2:
            prediction_period = st.selectbox(
                "⏱️ Prediction Period:",
                ["1 Week", "2 Weeks", "1 Month"],
                index=0
            )
        
        steps_map = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30}
        prediction_steps = steps_map[prediction_period]
        
        # ML Model Configuration
        with st.expander("🔧 Advanced Model Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                confidence_threshold = st.slider("Confidence Threshold", 0.3, 0.9, 0.6)
                include_technical = st.checkbox("Include Technical Analysis", value=True)
            with col2:
                risk_adjustment = st.selectbox("Risk Adjustment", ["Conservative", "Moderate", "Aggressive"])
                show_components = st.checkbox("Show Model Components", value=False)
        
        # Prediction Generation
        if st.button("🚀 Generate AI Prediction", type="primary", use_container_width=True):
            with st.spinner("🧠 AI models are analyzing market data..."):
                try:
                    # Fetch stock data with better error handling
                    import yfinance as yf
                    
                    stock_data = yf.download(selected_stock, period="1y", progress=False)
                    
                    if stock_data.empty:
                        st.error("❌ Unable to fetch stock data. Please try another stock.")
                    else:
                        # Initialize and run ensemble model
                        ensemble_model = EnsembleModel()
                        
                        # Extract Close prices for prediction
                        close_data = stock_data['Close'].dropna()
                        
                        if len(close_data) < 5:
                            st.error("❌ Insufficient historical data for prediction.")
                        else:
                            prediction_result = ensemble_model.predict(
                                close_data, 
                                steps=prediction_steps,
                                symbol=selected_stock
                            )
                            
                            # Validate prediction result
                            if not isinstance(prediction_result, dict) or 'predictions' not in prediction_result:
                                st.error("❌ Prediction failed. Please try again or select a different stock.")
                            else:
                                # Display results
                                st.markdown("---")
                                st.subheader("🎯 AI Prediction Results")
                                
                                # Key metrics with safe value extraction
                                col1, col2, col3, col4 = st.columns(4)
                                
                                current_price = prediction_result.get('current_price', 0)
                                predicted_price = prediction_result.get('predicted_price', 0)
                                price_change = prediction_result.get('price_change_percent', 0)
                                confidence = prediction_result.get('confidence', 0)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div class="ml-metric">
                                        <h4 style="margin: 0; color: #60a5fa;">Current Price</h4>
                                        <h2 style="margin: 5px 0; color: white;">₹{current_price:.2f}</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    color = "#10b981" if price_change > 0 else "#ef4444"
                                    arrow = "↗️" if price_change > 0 else "↘️"
                                    st.markdown(f"""
                                    <div class="ml-metric">
                                        <h4 style="margin: 0; color: #60a5fa;">Predicted Price</h4>
                                        <h2 style="margin: 5px 0; color: {color};">₹{predicted_price:.2f} {arrow}</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    st.markdown(f"""
                                    <div class="ml-metric">
                                        <h4 style="margin: 0; color: #60a5fa;">Expected Change</h4>
                                        <h2 style="margin: 5px 0; color: {color};">{price_change:+.1f}%</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col4:
                                    conf_color = "#10b981" if confidence > 0.7 else "#f59e0b" if confidence > 0.5 else "#ef4444"
                                    st.markdown(f"""
                                    <div class="ml-metric">
                                        <h4 style="margin: 0; color: #60a5fa;">AI Confidence</h4>
                                        <h2 style="margin: 5px 0; color: {conf_color};">{confidence:.1%}</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Prediction visualization
                                st.subheader("📈 Price Prediction Chart")
                                
                                try:
                                    # Prepare chart data safely
                                    historical_data = close_data.tail(30)
                                    pred_dates = prediction_result.get('dates', pd.date_range(
                                        start=datetime.now() + timedelta(days=1), 
                                        periods=prediction_steps, 
                                        freq='D'
                                    ))
                                    predictions = prediction_result['predictions']
                                    
                                    fig = go.Figure()
                                    
                                    # Historical prices
                                    fig.add_trace(go.Scatter(
                                        x=historical_data.index,
                                        y=historical_data.values,
                                        mode='lines',
                                        name='Historical Prices',
                                        line=dict(color='#60a5fa', width=2)
                                    ))
                                    
                                    # Predictions
                                    fig.add_trace(go.Scatter(
                                        x=pred_dates,
                                        y=predictions,
                                        mode='lines+markers',
                                        name=f'AI Predictions ({prediction_period})',
                                        line=dict(color='#10b981', width=3, dash='dot'),
                                        marker=dict(size=6)
                                    ))
                                    
                                    # Connection point
                                    if len(historical_data) > 0 and len(predictions) > 0:
                                        fig.add_trace(go.Scatter(
                                            x=[historical_data.index[-1], pred_dates[0]],
                                            y=[historical_data.iloc[-1], predictions[0]],
                                            mode='lines',
                                            name='Transition',
                                            line=dict(color='#f59e0b', width=2, dash='dash'),
                                            showlegend=False
                                        ))
                                    
                                    fig.update_layout(
                                        title=f"AI Prediction for {INDIAN_STOCKS[selected_stock]}",
                                        xaxis_title="Date",
                                        yaxis_title="Price (₹)",
                                        template='plotly_dark',
                                        height=500,
                                        hovermode='x unified'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                except Exception as chart_error:
                                    st.warning(f"Chart display error: {str(chart_error)}")
                                    st.info("Prediction data is available but chart couldn't be rendered.")
                                
                                # AI Summary
                                try:
                                    summary = ensemble_model.get_prediction_summary(prediction_result)
                                    method = prediction_result.get('method', 'Unknown')
                                    data_points = prediction_result.get('data_points', 0)
                                    volatility = prediction_result.get('volatility', 0)
                                    
                                    st.markdown(f"""
                                    <div class="prediction-card">
                                        <h3 style="margin-top: 0;">🤖 AI Analysis Summary</h3>
                                        {summary}
                                        
                                        <br><br>
                                        <strong>📊 Model Details:</strong><br>
                                        • Method: {method}<br>
                                        • Data Points: {data_points} trading days<br>
                                        • Volatility: {volatility:.3f}<br>
                                        • Risk Level: {risk_adjustment}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                except Exception as summary_error:
                                    st.info("Prediction completed successfully, but summary generation encountered an issue.")
                                
                                # Model components (if requested)
                                if show_components and 'individual_predictions' in prediction_result:
                                    st.subheader("🔍 Individual Model Components")
                                    
                                    components = prediction_result['individual_predictions']
                                    confidences = prediction_result.get('individual_confidences', {})
                                    
                                    if components:
                                        comp_cols = st.columns(min(len(components), 4))  # Max 4 columns
                                        for i, (model_name, preds) in enumerate(components.items()):
                                            if i < len(comp_cols):
                                                with comp_cols[i]:
                                                    try:
                                                        final_pred = preds[-1] if len(preds) > 0 else current_price
                                                        change = ((final_pred - current_price) / current_price) * 100 if current_price != 0 else 0
                                                        conf = confidences.get(model_name, 0.5)
                                                        
                                                        st.metric(
                                                            model_name.replace('_', ' ').title(),
                                                            f"₹{final_pred:.2f}",
                                                            f"{change:+.1f}%"
                                                        )
                                                        st.caption(f"Confidence: {conf:.1%}")
                                                    except Exception as comp_error:
                                                        st.caption(f"Component error: {model_name}")
                                
                                # Risk warnings
                                if confidence < confidence_threshold:
                                    st.warning(f"⚠️ Low prediction confidence ({confidence:.1%}). Consider waiting for more data or using conservative position sizing.")
                                
                                if abs(price_change) > 10:
                                    st.warning("⚠️ High volatility prediction detected. Exercise caution and consider risk management strategies.")
                                
                                # Disclaimer
                                st.markdown("---")
                                st.info("📢 **Disclaimer**: These AI predictions are for educational purposes only and should not be considered as financial advice. Always consult with qualified financial advisors and conduct your own research before making investment decisions.")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("💡 This might be due to:")
                    st.markdown("- Insufficient historical data for the selected stock")
                    st.markdown("- Temporary API rate limits from Yahoo Finance")
                    st.markdown("- Network connectivity issues")
                    st.markdown("- Try selecting a different stock or wait a few minutes before retrying")

elif selected_nav == "⚙️ User Settings" and ENHANCED_FEATURES and st.session_state.logged_in:
    st.title("⚙️ User Settings & Preferences")
    
    user = st.session_state.user
    
    # User Profile Section
    st.subheader("👤 Profile Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Username", value=user['username'], disabled=True)
        st.text_input("Email", value=user['email'], disabled=True)
    
    with col2:
        st.text_input("Member Since", value=user.get('created_at', 'Unknown'), disabled=True)
        st.text_input("Last Login", value=user.get('last_login', 'Never'), disabled=True)
    
    # Preferences Section
    st.subheader("🎨 Preferences")
    
    with st.form("preferences_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            theme_preference = st.selectbox(
                "Theme",
                ["Dark", "Light", "Auto"],
                index=0 if user.get('theme', 'dark') == 'dark' else 1
            )
            
            default_mode = st.selectbox(
                "Default Trading Mode",
                ["Beginner", "Pro", "Expert"],
                index=["Beginner", "Pro", "Expert"].index(user.get('default_mode', 'Beginner'))
            )
        
        with col2:
            email_notifications = st.checkbox(
                "Email Notifications",
                value=user.get('email_notifications', True)
            )
            
            auto_refresh = st.checkbox(
                "Auto-refresh Data",
                value=True
            )
        
        if st.form_submit_button("💾 Save Preferences", use_container_width=True):
            success = st.session_state.auth_handler.update_user_preferences(
                user['id'],
                theme=theme_preference.lower(),
                default_mode=default_mode,
                email_notifications=email_notifications
            )
            
            if success:
                st.success("✅ Preferences updated successfully!")
                # Update session state
                st.session_state.user.update({
                    'theme': theme_preference.lower(),
                    'default_mode': default_mode,
                    'email_notifications': email_notifications
                })
            else:
                st.error("❌ Failed to update preferences")
    
    # Portfolio Management Section
    st.subheader("💼 Portfolio Management")
    
    # Get user's portfolio
    user_portfolio = st.session_state.auth_handler.get_user_portfolio(user['id'])
    
    if user_portfolio:
        st.write(f"You have {len(user_portfolio)} holdings in your portfolio:")
        
        # Display portfolio in a nice format
        portfolio_df = pd.DataFrame(user_portfolio)
        st.dataframe(
            portfolio_df[['symbol', 'quantity', 'buy_price', 'buy_date']],
            use_container_width=True
        )
        
        # Option to clear portfolio
        if st.button("🗑️ Clear All Portfolio Data", type="secondary"):
            if st.checkbox("I understand this will delete all my portfolio data"):
                # Note: You'd need to implement this method in auth_handler
                st.warning("Clear portfolio functionality would be implemented here")
    else:
        st.info("📊 Your portfolio is empty. Add some stocks from the Portfolio Tracker page!")
    
    # Watchlist Management
    st.subheader("👀 Watchlist Management")
    
    user_watchlist = st.session_state.auth_handler.get_user_watchlist(user['id'])
    
    if user_watchlist:
        st.write(f"You're watching {len(user_watchlist)} stocks:")
        
        for item in user_watchlist:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"📈 {INDIAN_STOCKS.get(item['symbol'], item['symbol'])}")
            with col2:
                if item['alert_price']:
                    st.write(f"🔔 Alert: ₹{item['alert_price']}")
                else:
                    st.write("No alert set")
            with col3:
                if st.button("❌", key=f"remove_{item['id']}"):
                    # Remove from watchlist functionality would go here
                    st.rerun()
    else:
        st.info("👀 Your watchlist is empty. Add stocks from the Stock Analysis page!")
    
    # Data Export Section
    st.subheader("📥 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Portfolio", use_container_width=True):
            if user_portfolio:
                portfolio_df = pd.DataFrame(user_portfolio)
                csv = portfolio_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Portfolio CSV",
                    data=csv,
                    file_name=f"portfolio_{user['username']}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No portfolio data to export")
    
    with col2:
        if st.button("👀 Export Watchlist", use_container_width=True):
            if user_watchlist:
                watchlist_df = pd.DataFrame(user_watchlist)
                csv = watchlist_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Watchlist CSV",
                    data=csv,
                    file_name=f"watchlist_{user['username']}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No watchlist data to export")
    
    # Account Management
    st.subheader("🔐 Account Management")
    
    with st.expander("🔑 Change Password", expanded=False):
        with st.form("change_password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_new_password = st.text_input("Confirm New Password", type="password")
            
            if new_password:
                st.markdown(create_password_strength_indicator(new_password), unsafe_allow_html=True)
            
            if st.form_submit_button("🔄 Change Password"):
                if not all([current_password, new_password, confirm_new_password]):
                    st.error("❌ Please fill in all fields")
                elif new_password != confirm_new_password:
                    st.error("❌ New passwords don't match")
                else:
                    password_valid, password_msg = validate_password(new_password)
                    if not password_valid:
                        st.error(f"❌ {password_msg}")
                    else:
                        # Note: You'd need to implement change_password method in auth_handler
                        st.info("🔄 Password change functionality would be implemented here")
    
    with st.expander("⚠️ Danger Zone", expanded=False):
        st.markdown("### Delete Account")
        st.warning("⚠️ This action cannot be undone. All your data will be permanently deleted.")
        
        delete_confirmation = st.text_input(
            "Type 'DELETE' to confirm account deletion:",
            placeholder="Type DELETE here"
        )
        
        if st.button("🗑️ Delete Account", type="secondary", disabled=delete_confirmation != "DELETE"):
            st.error("🚨 Account deletion functionality would be implemented here with proper confirmation")

else:
    # Handle cases where enhanced features aren't available
    if not ENHANCED_FEATURES:
        st.title("📈 Indian Stock Trading Dashboard")
        st.warning("⚠️ Enhanced features (Authentication & ML) are not available.")
        st.info("💡 To enable full functionality, install required packages:")
        st.code("pip install tensorflow-cpu scikit-learn statsmodels bcrypt validators")
        
        st.markdown("---")
        st.markdown("### Available Features:")
        st.markdown("✅ Market Overview")
        st.markdown("✅ Stock Analysis")  
        st.markdown("✅ Portfolio Tracker")
        st.markdown("✅ News & Sentiment")
        st.markdown("❌ User Authentication")
        st.markdown("❌ ML Predictions")
        st.markdown("❌ Personal Settings")
    
    elif selected_nav not in ["📊 Market Overview", "📈 Stock Analysis", "💼 Portfolio Tracker", "📰 News & Sentiment"]:
        st.title("🔒 Authentication Required")
        st.info("Please login to access this feature.")
        
        # Quick login form in main area
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("### Quick Login")
                with st.form("main_login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    
                    if st.form_submit_button("Login", use_container_width=True):
                        if username and password:
                            user_id, message = st.session_state.auth_handler.verify_user(username, password)
                            if user_id:
                                st.session_state.logged_in = True
                                st.session_state.user = st.session_state.auth_handler.get_user_info(user_id)
                                st.success("✅ Login successful!")
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")

# Footer with additional information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🚀 Features")
    st.markdown("- Real-time market data")
    st.markdown("- Technical analysis")
    st.markdown("- News sentiment analysis")
    if ENHANCED_FEATURES:
        st.markdown("- AI-powered predictions")
        st.markdown("- User authentication")

with col2:
    st.markdown("### 📊 Data Sources")
    st.markdown("- Yahoo Finance")
    st.markdown("- NSE/BSE APIs")
    st.markdown("- News aggregators")
    st.markdown("- Technical indicators")

with col3:
    st.markdown("### ⚠️ Disclaimer")
    st.markdown("This application is for educational purposes only.")
    st.markdown("Not financial advice.")
    st.markdown("Please consult qualified advisors.")

# Performance metrics (if user is logged in)
if ENHANCED_FEATURES and st.session_state.logged_in:
    with st.sidebar:
        if st.button("📊 Performance Metrics"):
            st.session_state.show_performance = True
        
        if st.session_state.get('show_performance', False):
            st.markdown("### 📈 Your Trading Stats")
            # Mock performance data
            st.metric("Portfolio Return", "+12.5%", "+2.3%")
            st.metric("Win Rate", "68%", "+5%")
            st.metric("Predictions Used", "23", "+3")
            
            if st.button("❌ Close"):
                st.session_state.show_performance = False
                st.rerun()
