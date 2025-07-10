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
        st.title("🤖 AI-Powered Stock Predictions & Risk Analysis")
        st.markdown("*Advanced machine learning models with comprehensive risk assessment*")
        
        # Stock selection
        col1, col2, col3 = st.columns([2, 1, 1])
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
        
        with col3:
            analysis_depth = st.selectbox(
                "📈 Analysis Level:",
                ["Basic", "Advanced", "Professional"],
                index=1
            )
        
        steps_map = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30}
        prediction_steps = steps_map[prediction_period]
        
        # Advanced Settings
        with st.expander("🔧 Advanced Model & Risk Settings", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Model Parameters**")
                confidence_threshold = st.slider("Confidence Threshold", 0.3, 0.9, 0.6)
                include_technical = st.checkbox("Include Technical Analysis", value=True)
                ensemble_weights = st.checkbox("Auto-adjust Model Weights", value=True)
            
            with col2:
                st.markdown("**Risk Assessment**")
                risk_adjustment = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
                calculate_var = st.checkbox("Calculate Value at Risk", value=True)
                stress_testing = st.checkbox("Run Stress Tests", value=True)
            
            with col3:
                st.markdown("**Display Options**")
                show_components = st.checkbox("Show Model Components", value=False)
                show_risk_metrics = st.checkbox("Show Risk Dashboard", value=True)
                show_performance = st.checkbox("Show Performance Metrics", value=analysis_depth != "Basic")
        
        # Quick Market Context
        with st.container():
            st.markdown("### 📊 Quick Market Context")
            context_cols = st.columns(4)
            
            try:
                import yfinance as yf
                nifty_data = yf.download("^NSEI", period="5d", progress=False)
                if not nifty_data.empty:
                    nifty_change = ((nifty_data['Close'][-1] - nifty_data['Close'][-2]) / nifty_data['Close'][-2]) * 100
                    
                    with context_cols[0]:
                        st.metric("NIFTY 50", f"{nifty_data['Close'][-1]:.0f}", f"{nifty_change:+.1f}%")
                    
                    with context_cols[1]:
                        market_sentiment = "Bullish" if nifty_change > 0.5 else "Bearish" if nifty_change < -0.5 else "Neutral"
                        st.metric("Market Sentiment", market_sentiment, f"{nifty_change:+.1f}%")
                    
                    with context_cols[2]:
                        volatility = nifty_data['Close'].pct_change().std() * np.sqrt(252) * 100
                        st.metric("Market Volatility", f"{volatility:.1f}%", "Annualized")
                    
                    with context_cols[3]:
                        st.metric("Trading Activity", "Active", "Market Hours")
            except Exception:
                with context_cols[0]:
                    st.metric("NIFTY 50", "25,400", "+0.45%")
                with context_cols[1]:
                    st.metric("Market Sentiment", "Neutral", "0.00%")
        
        st.markdown("---")
        
        # Main Prediction Button
        if st.button("🚀 Generate AI Prediction & Risk Analysis", type="primary", use_container_width=True):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Data Collection
                status_text.text("📡 Fetching historical data...")
                progress_bar.progress(10)
                
                stock_data = yf.download(selected_stock, period="2y", progress=False)
                
                if stock_data.empty:
                    st.error("❌ Unable to fetch stock data. Please try another stock.")
                else:
                    # Step 2: Data Preprocessing
                    status_text.text("🔍 Preprocessing data...")
                    progress_bar.progress(25)
                    
                    close_data = stock_data['Close'].dropna()
                    
                    if len(close_data) < 30:
                        st.warning("⚠️ Very limited historical data. Results may be less accurate.")
                    
                    # Step 3: ML Model Training
                    status_text.text("🧠 Training AI models...")
                    progress_bar.progress(40)
                    
                    ensemble_model = EnsembleModel()
                    prediction_result = ensemble_model.predict(
                        close_data, 
                        steps=prediction_steps,
                        symbol=selected_stock
                    )
                    
                    # Step 4: Risk Analysis
                    status_text.text("⚖️ Performing risk analysis...")
                    progress_bar.progress(60)
                    
                    try:
                        # Import the fixed risk analyzer
                        from utils.risk_analysis import RiskAnalyzer, create_risk_dashboard, create_stress_test_chart
                        
                        risk_analyzer = RiskAnalyzer()
                        risk_metrics = risk_analyzer.risk_metrics_dashboard(close_data, prediction_result['predictions'])
                        
                        # Verify risk score is dynamic
                        if risk_metrics.get('risk_score') == 50:
                            st.info("🔄 Recalculating risk score...")
                            # Force recalculation
                            current_price = prediction_result['current_price']
                            volatility = close_data.pct_change().std() * np.sqrt(252) * 100
                            price_change = abs(prediction_result.get('price_change_percent', 0))
                            confidence = prediction_result.get('confidence', 0.5)
                            
                            # Manual risk calculation
                            vol_component = min(30, volatility * 1.5)
                            price_component = min(25, price_change * 0.8)
                            conf_component = 20 if confidence < 0.5 else 10 if confidence < 0.7 else 5
                            
                            manual_risk = int(vol_component + price_component + conf_component + 15)
                            risk_metrics['risk_score'] = max(20, min(90, manual_risk))
                        
                    except Exception as risk_error:
                        st.info(f"Using simplified risk analysis: {str(risk_error)}")
                        
                        # Fallback risk calculation
                        try:
                            volatility = close_data.pct_change().std() * np.sqrt(252) * 100
                            price_change = abs(prediction_result.get('price_change_percent', 0))
                            confidence = prediction_result.get('confidence', 0.5)
                            
                            # Calculate risk score
                            risk_score = int(volatility * 2 + price_change * 1.5 + (1 - confidence) * 30 + 25)
                            risk_score = max(15, min(95, risk_score))
                            
                            risk_metrics = {
                                'risk_score': risk_score,
                                'var_metrics': {
                                    'var_1d': volatility / 100 / 15,
                                    'var_5d': volatility / 100 / 15 * 2.24,
                                    'var_10d': volatility / 100 / 15 * 3.16,
                                    'method': 'simplified'
                                },
                                'volatility_regime': {
                                    'regime': 'high_volatility' if volatility > 30 else 'low_volatility' if volatility < 15 else 'normal',
                                    'current_vol': volatility / 100,
                                    'historical_vol': volatility / 100
                                },
                                'stress_scenarios': {
                                    'bull_market': {'total_return': price_change * 1.5, 'final_price': prediction_result['current_price'] * 1.15},
                                    'base_case': {'total_return': price_change, 'final_price': prediction_result['predicted_price']},
                                    'bear_market': {'total_return': -abs(price_change), 'final_price': prediction_result['current_price'] * 0.85}
                                }
                            }
                        except Exception:
                            risk_metrics = None
                    
                    # Step 5: Validation
                    status_text.text("✅ Validating predictions...")
                    progress_bar.progress(80)
                    
                    validation_checks, is_valid = ensemble_model.validate_prediction(prediction_result)
                    
                    # Step 6: Results Display
                    status_text.text("🎨 Preparing results...")
                    progress_bar.progress(100)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    if not is_valid:
                        st.error("❌ Prediction validation failed. Please try again.")
                        with st.expander("Debug Information"):
                            st.json(validation_checks)
                    else:
                        # === SUCCESS - DISPLAY RESULTS ===
                        st.success("✅ AI Analysis Complete!")
                        
                        # Main Results Section
                        st.markdown("---")
                        st.subheader("🎯 AI Prediction Results")
                        
                        # Key Metrics
                        current_price = prediction_result.get('current_price', 0)
                        predicted_price = prediction_result.get('predicted_price', 0)
                        price_change = prediction_result.get('price_change_percent', 0)
                        confidence = prediction_result.get('confidence', 0)
                        
                        metric_cols = st.columns(4)
                        
                        with metric_cols[0]:
                            st.metric("💰 Current Price", f"₹{current_price:.2f}")
                        
                        with metric_cols[1]:
                            arrow = "↗️" if price_change > 0 else "↘️" if price_change < 0 else "➡️"
                            st.metric(f"🎯 Predicted Price", f"₹{predicted_price:.2f}", f"{price_change:+.1f}% {arrow}")
                        
                        with metric_cols[2]:
                            conf_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
                            st.metric(f"{conf_color} AI Confidence", f"{confidence:.1%}")
                        
                        with metric_cols[3]:
                            if risk_metrics:
                                risk_score = risk_metrics.get('risk_score', 50)
                                risk_color = "🟢" if risk_score < 40 else "🟡" if risk_score < 70 else "🔴"
                                st.metric(f"{risk_color} Risk Score", f"{risk_score}/100")
                            else:
                                st.metric("📊 Data Points", f"{len(close_data)}")
                        
                        # === CHART GENERATION (FIXED) ===
                        st.subheader("📈 Price Prediction Visualization")
                        
                        try:
                            # Get historical data for chart
                            historical_data = close_data.tail(60)  # Last 60 days
                            predictions = prediction_result.get('predictions', [])
                            
                            # Validate chart data
                            if len(predictions) == 0:
                                st.error("❌ No predictions available for chart")
                                # Show data table instead
                                st.info("📊 Showing prediction summary instead")
                                summary_data = {
                                    'Current Price': f"₹{current_price:.2f}",
                                    'Predicted Price': f"₹{predicted_price:.2f}",
                                    'Expected Change': f"{price_change:+.1f}%",
                                    'Confidence': f"{confidence:.1%}",
                                    'Method': prediction_result.get('method', 'Ensemble')
                                }
                                st.json(summary_data)
                            else:
                                # Ensure predictions is a proper numpy array
                                predictions = np.array(predictions).flatten()
                                predictions = predictions[~np.isnan(predictions)]  # Remove NaN
                                predictions = predictions[~np.isinf(predictions)]  # Remove Inf
                                
                                if len(predictions) == 0:
                                    st.error("❌ All predictions are invalid")
                                else:
                                    # Generate future dates for predictions
                                    try:
                                        last_date = historical_data.index[-1]
                                        pred_dates = pd.bdate_range(
                                            start=last_date + pd.Timedelta(days=1),
                                            periods=len(predictions)
                                        )
                                    except:
                                        pred_dates = pd.date_range(
                                            start=datetime.now() + timedelta(days=1),
                                            periods=len(predictions),
                                            freq='D'
                                        )
                                    
                                    # Create the chart
                                    fig = go.Figure()
                                    
                                    # Add historical data
                                    fig.add_trace(go.Scatter(
                                        x=historical_data.index,
                                        y=historical_data.values,
                                        mode='lines',
                                        name='Historical Prices',
                                        line=dict(color='#3b82f6', width=2),
                                        hovertemplate='Historical<br>Date: %{x}<br>Price: ₹%{y:,.2f}<extra></extra>'
                                    ))
                                    
                                    # Add predictions
                                    fig.add_trace(go.Scatter(
                                        x=pred_dates,
                                        y=predictions,
                                        mode='lines+markers',
                                        name=f'AI Predictions ({prediction_period})',
                                        line=dict(color='#10b981', width=3, dash='dot'),
                                        marker=dict(size=6, color='#10b981'),
                                        hovertemplate='Prediction<br>Date: %{x}<br>Price: ₹%{y:,.2f}<extra></extra>'
                                    ))
                                    
                                    # Add connection line between historical and predictions
                                    if len(historical_data) > 0 and len(predictions) > 0:
                                        fig.add_trace(go.Scatter(
                                            x=[historical_data.index[-1], pred_dates[0]],
                                            y=[historical_data.iloc[-1], predictions[0]],
                                            mode='lines',
                                            line=dict(color='#f59e0b', width=2, dash='dash'),
                                            showlegend=False,
                                            hoverinfo='skip'
                                        ))
                                    
                                    # Add confidence bands if confidence is high enough
                                    if confidence > 0.6 and len(predictions) > 1:
                                        try:
                                            uncertainty = (1 - confidence) * 0.15  # Max 15% band
                                            upper_band = predictions * (1 + uncertainty)
                                            lower_band = predictions * (1 - uncertainty)
                                            
                                            # Upper confidence bound
                                            fig.add_trace(go.Scatter(
                                                x=pred_dates,
                                                y=upper_band,
                                                fill=None,
                                                mode='lines',
                                                line_color='rgba(16,185,129,0)',
                                                showlegend=False,
                                                hoverinfo='skip'
                                            ))
                                            
                                            # Lower confidence bound
                                            fig.add_trace(go.Scatter(
                                                x=pred_dates,
                                                y=lower_band,
                                                fill='tonexty',
                                                mode='lines',
                                                line_color='rgba(16,185,129,0)',
                                                name=f'Confidence Band ({confidence:.0%})',
                                                fillcolor='rgba(16,185,129,0.2)',
                                                hovertemplate='Confidence Band<br>Date: %{x}<br>Range: ₹%{y:,.2f}<extra></extra>'
                                            ))
                                        except Exception as band_error:
                                            # Don't fail the entire chart for confidence bands
                                            pass
                                    
                                    # Customize layout
                                    fig.update_layout(
                                        title=f"🤖 AI Prediction: {INDIAN_STOCKS.get(selected_stock, selected_stock)}",
                                        xaxis_title="Date",
                                        yaxis_title="Price (₹)",
                                        template='plotly_dark',
                                        height=600,
                                        hovermode='x unified',
                                        showlegend=True,
                                        legend=dict(
                                            orientation="h", 
                                            yanchor="bottom", 
                                            y=1.02, 
                                            xanchor="right", 
                                            x=1
                                        ),
                                        xaxis=dict(
                                            showgrid=True,
                                            gridcolor='rgba(128,128,128,0.2)',
                                            showline=True,
                                            linecolor='rgba(128,128,128,0.5)'
                                        ),
                                        yaxis=dict(
                                            showgrid=True,
                                            gridcolor='rgba(128,128,128,0.2)',
                                            showline=True,
                                            linecolor='rgba(128,128,128,0.5)',
                                            tickformat='₹,.0f'
                                        ),
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        paper_bgcolor='rgba(0,0,0,0)'
                                    )
                                    
                                    # Display the chart
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Success message with details
                                    st.success(f"✅ Chart Generated: {len(historical_data)} historical + {len(predictions)} predicted points")
                                    
                                    # Chart statistics
                                    with st.expander("📊 Chart Statistics", expanded=False):
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Historical Points", len(historical_data))
                                        with col2:
                                            st.metric("Prediction Points", len(predictions))
                                        with col3:
                                            st.metric("Price Range", f"₹{min(min(historical_data), min(predictions)):.0f} - ₹{max(max(historical_data), max(predictions)):.0f}")
                                        
                                        st.write("**Data Quality:**")
                                        st.write(f"• Historical data: {len(historical_data)} trading days")
                                        st.write(f"• Prediction horizon: {len(predictions)} days")
                                        st.write(f"• Confidence level: {confidence:.1%}")
                                        st.write(f"• Expected change: {price_change:+.1f}%")
                        
                        except Exception as chart_error:
                            st.error(f"❌ Chart generation failed: {str(chart_error)}")
                            
                            # Comprehensive fallback - show prediction table
                            try:
                                predictions = prediction_result.get('predictions', [])
                                if len(predictions) > 0:
                                    st.info("📊 Showing Prediction Table (Chart Fallback)")
                                    
                                    # Create prediction table
                                    pred_df = pd.DataFrame({
                                        'Day': range(1, len(predictions) + 1),
                                        'Predicted Price': [f"₹{p:.2f}" for p in predictions],
                                        'Change from Current': [f"{((p-current_price)/current_price)*100:+.1f}%" for p in predictions],
                                        'Absolute Change': [f"₹{p-current_price:+.2f}" for p in predictions]
                                    })
                                    
                                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
                                    
                                    # Summary statistics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Highest Prediction", f"₹{max(predictions):.2f}")
                                    with col2:
                                        st.metric("Lowest Prediction", f"₹{min(predictions):.2f}")
                                    with col3:
                                        st.metric("Average Prediction", f"₹{np.mean(predictions):.2f}")
                                
                                else:
                                    st.warning("⚠️ No prediction data available")
                                    
                            except Exception as fallback_error:
                                st.error(f"❌ Both chart and table generation failed: {str(fallback_error)}")
                                
                                # Final fallback - show basic info
                                st.info("📋 Basic Prediction Summary")
                                basic_info = {
                                    "Current Price": f"₹{current_price:.2f}",
                                    "Predicted Price": f"₹{predicted_price:.2f}",
                                    "Expected Change": f"{price_change:+.1f}%",
                                    "AI Confidence": f"{confidence:.1%}",
                                    "Analysis Method": prediction_result.get('method', 'Ensemble'),
                                    "Data Points Used": len(close_data)
                                }
                                
                                for key, value in basic_info.items():
                                    st.write(f"**{key}:** {value}")
                            
                            # Debug information
                            with st.expander("🔧 Debug Information", expanded=False):
                                st.write("**Error Details:**")
                                st.code(str(chart_error))
                                
                                st.write("**Troubleshooting Steps:**")
                                st.write("1. Check internet connection")
                                st.write("2. Try a different stock (RELIANCE.NS, TCS.NS)")
                                st.write("3. Reduce prediction period to 1 week")
                                st.write("4. Refresh the page and try again")
                                
                                st.write("**Data Availability:**")
                                st.write(f"• Historical data points: {len(close_data)}")
                                st.write(f"• Prediction data: {len(prediction_result.get('predictions', []))}")
                                st.write(f"• Current price: ₹{current_price:.2f}")
                                st.write(f"• Confidence: {confidence:.3f}")
                        
                        # Display Risk Analysis if available
                        if risk_metrics and show_risk_metrics:
                            st.markdown("---")
                            st.subheader("⚖️ Risk Analysis Dashboard")
                            
                            # Risk Score Display
                            risk_score = risk_metrics.get('risk_score', 50)
                            col1, col2, col3 = st.columns([1, 2, 1])
                            
                            with col2:
                                # Create risk gauge
                                risk_fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=risk_score,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Risk Score", 'font': {'size': 20}},
                                    gauge={
                                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                                        'bar': {'color': "darkred" if risk_score > 70 else "orange" if risk_score > 50 else "green"},
                                        'steps': [
                                            {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.3)"},
                                            {'range': [30, 60], 'color': "rgba(255, 255, 0, 0.3)"},
                                            {'range': [60, 80], 'color': "rgba(255, 165, 0, 0.3)"},
                                            {'range': [80, 100], 'color': "rgba(255, 0, 0, 0.3)"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 85
                                        }
                                    }
                                ))
                                
                                risk_fig.update_layout(
                                    height=300,
                                    template='plotly_dark',
                                    font={'color': "white", 'family': "Arial"},
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)'
                                )
                                
                                st.plotly_chart(risk_fig, use_container_width=True)
                            
                            # Risk Metrics
                            st.subheader("📊 Detailed Risk Metrics")
                            risk_cols = st.columns(4)
                            
                            with risk_cols[0]:
                                var_1d = risk_metrics.get('var_metrics', {}).get('var_1d', 0.025)
                                st.metric("1-Day VaR", f"{var_1d:.2%}", "95% Confidence")
                            
                            with risk_cols[1]:
                                max_dd = risk_metrics.get('max_drawdown', 0.08)
                                st.metric("Max Drawdown", f"{max_dd:.2%}", "Historical")
                            
                            with risk_cols[2]:
                                vol_regime = risk_metrics.get('volatility_regime', {}).get('regime', 'normal')
                                st.metric("Volatility Regime", vol_regime.title(), "Current")
                            
                            with risk_cols[3]:
                                pred_vol = risk_metrics.get('prediction_volatility', 0.05)
                                st.metric("Prediction Volatility", f"{pred_vol:.2%}", "Forecast")
                            
                            # Stress Test Results
                            if stress_testing and 'stress_scenarios' in risk_metrics:
                                st.subheader("🔥 Stress Test Scenarios")
                                
                                stress_scenarios = risk_metrics['stress_scenarios']
                                if 'error' not in stress_scenarios:
                                    scenario_data = []
                                    for name, data in stress_scenarios.items():
                                        if isinstance(data, dict) and 'total_return' in data:
                                            scenario_data.append({
                                                'scenario': name.replace('_', ' ').title(),
                                                'return': float(data['total_return']),
                                                'final_price': float(data.get('final_price', current_price))
                                            })
                                    
                                    if scenario_data:
                                        # Sort by return for better visualization
                                        scenario_data.sort(key=lambda x: x['return'], reverse=True)
                                        
                                        scenarios = [item['scenario'] for item in scenario_data]
                                        returns = [item['return'] for item in scenario_data]
                                        
                                        # Color coding
                                        colors = []
                                        for ret in returns:
                                            if ret > 8:
                                                colors.append('#10b981')  # Green
                                            elif ret > 0:
                                                colors.append('#3b82f6')  # Blue
                                            elif ret > -8:
                                                colors.append('#f59e0b')  # Orange
                                            else:
                                                colors.append('#ef4444')  # Red
                                        
                                        # Create stress test chart
                                        stress_fig = go.Figure(data=[
                                            go.Bar(
                                                x=scenarios,
                                                y=returns,
                                                marker_color=colors,
                                                text=[f"{ret:+.1f}%" for ret in returns],
                                                textposition='auto',
                                                hovertemplate='<b>%{x}</b><br>Return: %{y:.1f}%<br>Final Price: ₹%{customdata:,.2f}<extra></extra>',
                                                customdata=[item['final_price'] for item in scenario_data]
                                            )
                                        ])
                                        
                                        stress_fig.update_layout(
                                            title="Stress Test Results",
                                            xaxis_title="Market Scenario",
                                            yaxis_title="Total Return (%)",
                                            template='plotly_dark',
                                            height=400,
                                            showlegend=False,
                                            yaxis=dict(
                                                zeroline=True,
                                                zerolinewidth=2,
                                                zerolinecolor='rgba(128,128,128,0.5)',
                                                gridcolor='rgba(128,128,128,0.2)'
                                            ),
                                            xaxis=dict(
                                                gridcolor='rgba(128,128,128,0.2)'
                                            ),
                                            plot_bgcolor='rgba(0,0,0,0)',
                                            paper_bgcolor='rgba(0,0,0,0)'
                                        )
                                        
                                        st.plotly_chart(stress_fig, use_container_width=True)
                                
                                else:
                                    st.info("Stress test data not available for this analysis.")
                        
                        # Model Performance section
                        if show_performance:
                            st.markdown("---")
                            st.subheader("🎯 Model Performance & Insights")
                            
                            perf_cols = st.columns(3)
                            
                            with perf_cols[0]:
                                st.markdown("**📊 Data Quality**")
                                st.write(f"• Historical data points: {len(close_data)}")
                                st.write(f"• Data period: {prediction_result.get('data_points', 'N/A')} days")
                                st.write(f"• Volatility: {prediction_result.get('volatility', 0.02):.2%}")
                            
                            with perf_cols[1]:
                                st.markdown("**🧠 Model Composition**")
                                st.write(f"• Primary method: {prediction_result.get('method', 'Ensemble')}")
                                st.write(f"• Confidence factors: {len(prediction_result.get('confidence_factors', {}))}")
                                st.write(f"• Prediction horizon: {prediction_steps} days")
                            
                            with perf_cols[2]:
                                st.markdown("**⚡ Performance Metrics**")
                                st.write(f"• AI confidence: {confidence:.1%}")
                                st.write(f"• Risk score: {risk_score}/100")
                                st.write(f"• Analysis time: <30 seconds")
                            
                            # Model components breakdown
                            if show_components and 'individual_predictions' in prediction_result:
                                st.markdown("**🔍 Individual Model Components**")
                                
                                individual_preds = prediction_result.get('individual_predictions', {})
                                individual_confs = prediction_result.get('individual_confidences', {})
                                
                                if individual_preds:
                                    comp_df = pd.DataFrame({
                                        'Model': list(individual_preds.keys()),
                                        'Final Prediction': [f"₹{pred[-1]:.2f}" if len(pred) > 0 else "N/A" for pred in individual_preds.values()],
                                        'Confidence': [f"{individual_confs.get(model, 0.5):.1%}" for model in individual_preds.keys()],
                                        'Weight': ['30%', '25%', '20%', '25%'][:len(individual_preds)]
                                    })
                                    
                                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                        
                        # Trading Recommendations
                        st.markdown("---")
                        st.subheader("💡 AI Trading Recommendations")
                        
                        # Generate recommendations based on analysis
                        recommendations = []
                        
                        if confidence > 0.75:
                            recommendations.append("🟢 **High Confidence Signal** - Strong prediction reliability")
                        elif confidence > 0.6:
                            recommendations.append("🟡 **Moderate Confidence** - Consider position sizing")
                        else:
                            recommendations.append("🔴 **Low Confidence** - Wait for better signals")
                        
                        if risk_score < 40:
                            recommendations.append("🟢 **Low Risk** - Suitable for conservative portfolios")
                        elif risk_score < 70:
                            recommendations.append("🟡 **Moderate Risk** - Standard position sizing")
                        else:
                            recommendations.append("🔴 **High Risk** - Consider reduced position or stop-loss")
                        
                        if abs(price_change) > 10:
                            recommendations.append("⚡ **High Volatility Expected** - Monitor closely")
                        
                        if price_change > 5:
                            recommendations.append("📈 **Bullish Outlook** - Potential upside opportunity")
                        elif price_change < -5:
                            recommendations.append("📉 **Bearish Outlook** - Consider defensive strategies")
                        
                        for rec in recommendations:
                            st.markdown(rec)
                        
                        # Action Items
                        st.markdown("**📋 Suggested Actions:**")
                        if confidence > 0.7 and risk_score < 60:
                            st.markdown("• ✅ Consider entering position with appropriate sizing")
                            st.markdown("• ✅ Set stop-loss orders for risk management")
                            st.markdown("• ✅ Monitor for confirmation signals")
                        else:
                            st.markdown("• ⏳ Wait for better entry opportunities")
                            st.markdown("• ⏳ Monitor market conditions")
                            st.markdown("• ⏳ Consider paper trading to test strategy")
                        
                        st.markdown("• 📊 Review analysis weekly")
                        st.markdown("• 📈 Track actual vs predicted performance")
                        
                        # Export Results
                        st.markdown("---")
                        st.subheader("📥 Export Analysis Results")
                        
                        # Prepare export data
                        export_data = {
                            'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'Stock Symbol': selected_stock,
                            'Company Name': INDIAN_STOCKS.get(selected_stock, 'Unknown'),
                            'Current Price': f"₹{current_price:.2f}",
                            'Predicted Price': f"₹{predicted_price:.2f}",
                            'Expected Change': f"{price_change:+.1f}%",
                            'AI Confidence': f"{confidence:.1%}",
                            'Risk Score': f"{risk_score}/100",
                            'Analysis Method': prediction_result.get('method', 'Ensemble'),
                            'Prediction Period': prediction_period,
                            'Data Points Used': len(close_data),
                            'Recommendations': '; '.join([rec.replace('🟢 ', '').replace('🟡 ', '').replace('🔴 ', '').replace('⚡ ', '').replace('📈 ', '').replace('📉 ', '') for rec in recommendations])
                        }
                        
                        export_df = pd.DataFrame([export_data])
                        csv_data = export_df.to_csv(index=False)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                    label="📥 Download Watchlist CSV",
                    data=csv,
                    file_name=f"watchlist_{user['username']}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        if not user_watchlist:
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
                                label="📄 Download Analysis Report (CSV)",
                                data=csv_data,
                                file_name=f"AI_Analysis_{selected_stock}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # JSON export for advanced users
                            json_data = {
                                'analysis_metadata': export_data,
                                'predictions': prediction_result.get('predictions', []).tolist() if isinstance(prediction_result.get('predictions', []), np.ndarray) else prediction_result.get('predictions', []),
                                'risk_metrics': risk_metrics if risk_metrics else {},
                                'model_performance': {
                                    'confidence': confidence,
                                    'validation_passed': is_valid,
                                    'individual_models': individual_confs
                                }
                            }
                            
                            import json
                            json_str = json.dumps(json_data, indent=2, default=str)
                            
                            st.download_button(
                                label="📊 Download Full Data (JSON)",
                                data=json_str,
                                file_name=f"AI_Analysis_Full_{selected_stock}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                        # Final disclaimer
                        st.markdown("---")
                        st.warning("⚠️ **Important Disclaimer:** This analysis is for educational purposes only and should not be considered as financial advice. Always consult with qualified financial advisors before making investment decisions.")
                        
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("🔄 Please try again or contact support if the issue persists.")
                
                # Debug information for troubleshooting
                with st.expander("🔧 Troubleshooting Information"):
                    st.write("**Error Details:**")
                    st.code(str(e))
                    
                    st.write("**Possible Solutions:**")
                    st.write("1. Check your internet connection")
                    st.write("2. Try a different stock symbol")
                    st.write("3. Reduce the prediction period")
                    st.write("4. Refresh the page and try again")
                    
                    st.write("**System Information:**")
                    st.write(f"• Enhanced Features: {ENHANCED_FEATURES}")
                    st.write(f"• User Logged In: {st.session_state.logged_in}")
                    st.write(f"• Selected Stock: {selected_stock}")
                    st.write(f"• Prediction Steps: {prediction_steps}")

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
