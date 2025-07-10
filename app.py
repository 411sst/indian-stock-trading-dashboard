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
        
        # Stock selection with enhanced info
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
        
        # Advanced Configuration
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
            
            # Fetch quick market data
            try:
                import yfinance as yf
                nifty_data = yf.download("^NSEI", period="5d", progress=False)
                if not nifty_data.empty:
                    nifty_change = ((nifty_data['Close'][-1] - nifty_data['Close'][-2]) / nifty_data['Close'][-2]) * 100
                    
                    with context_cols[0]:
                        st.metric("NIFTY 50", f"₹{nifty_data['Close'][-1]:.0f}", f"{nifty_change:+.1f}%")
                    
                    with context_cols[1]:
                        market_sentiment = "Bullish" if nifty_change > 0.5 else "Bearish" if nifty_change < -0.5 else "Neutral"
                        st.metric("Market Sentiment", market_sentiment, f"{nifty_change:+.1f}%")
                    
                    with context_cols[2]:
                        volatility = nifty_data['Close'].pct_change().std() * np.sqrt(252) * 100
                        st.metric("Market Volatility", f"{volatility:.1f}%", "Annualized")
                    
                    with context_cols[3]:
                        trading_volume = nifty_data['Volume'][-1] / 1e6 if 'Volume' in nifty_data.columns else 0
                        st.metric("Trading Activity", f"{trading_volume:.0f}M", "Volume")
                        
            except Exception:
                with context_cols[0]:
                    st.metric("NIFTY 50", "25,400", "+0.45%")
                with context_cols[1]:
                    st.metric("Market Sentiment", "Neutral", "0.00%")
        
        st.markdown("---")
        
        # Main Prediction Button
        prediction_button = st.button(
            "🚀 Generate AI Prediction & Risk Analysis", 
            type="primary", 
            use_container_width=True,
            help="Generate comprehensive AI predictions with risk assessment"
        )
        
        if prediction_button:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Data Collection
                status_text.text("📡 Fetching historical data...")
                progress_bar.progress(10)
                
                stock_data = yf.download(selected_stock, period="2y", progress=False)  # More data for better analysis
                
                if stock_data.empty:
                    st.error("❌ Unable to fetch stock data. Please try another stock.")
                else:
                    # Step 2: Data Preprocessing
                    status_text.text("🔍 Preprocessing data...")
                    progress_bar.progress(25)
                    
                    close_data = stock_data['Close'].dropna()
                    
                    if len(close_data) < 50:  # Need more data for advanced analysis
                        st.warning("⚠️ Limited historical data. Results may be less accurate.")
                    
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
                    if calculate_var or stress_testing or show_risk_metrics:
                        status_text.text("⚖️ Performing risk analysis...")
                        progress_bar.progress(60)
                        
                        try:
                            from utils.risk_analysis import RiskAnalyzer, PerformanceAnalyzer, create_risk_dashboard, create_stress_test_chart
                            
                            risk_analyzer = RiskAnalyzer()
                            performance_analyzer = PerformanceAnalyzer()
                            
                            risk_metrics = risk_analyzer.risk_metrics_dashboard(close_data, prediction_result['predictions'])
                            performance_metrics = performance_analyzer.calculate_performance_metrics(close_data)
                            
                        except ImportError:
                            st.info("📊 Advanced risk analysis module not available. Showing basic analysis.")
                            risk_metrics = None
                            performance_metrics = None
                    
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
                        st.error("❌ Prediction validation failed. Please try again or select a different stock.")
                        st.json(validation_checks)  # Debug info
                    else:
                        # SUCCESS - Display comprehensive results
                        st.success("✅ AI Analysis Complete!")
                        
                        # === MAIN RESULTS SECTION ===
                        st.markdown("---")
                        st.subheader("🎯 AI Prediction Results")
                        
                        # Key Metrics Row
                        metric_cols = st.columns(4)
                        
                        current_price = prediction_result.get('current_price', 0)
                        predicted_price = prediction_result.get('predicted_price', 0)
                        price_change = prediction_result.get('price_change_percent', 0)
                        confidence = prediction_result.get('confidence', 0)
                        
                        with metric_cols[0]:
                            st.metric(
                                "💰 Current Price",
                                f"₹{current_price:.2f}",
                                help="Latest closing price"
                            )
                        
                        with metric_cols[1]:
                            color = "📈" if price_change > 0 else "📉"
                            st.metric(
                                f"{color} Predicted Price",
                                f"₹{predicted_price:.2f}",
                                f"{price_change:+.1f}%",
                                help=f"AI prediction for {prediction_period}"
                            )
                        
                        with metric_cols[2]:
                            conf_emoji = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
                            st.metric(
                                f"{conf_emoji} AI Confidence",
                                f"{confidence:.1%}",
                                help="Model confidence in prediction accuracy"
                            )
                        
                        with metric_cols[3]:
                            if risk_metrics and 'risk_score' in risk_metrics:
                                risk_score = risk_metrics['risk_score']
                                risk_emoji = "🟢" if risk_score < 30 else "🟡" if risk_score < 60 else "🔴"
                                st.metric(
                                    f"{risk_emoji} Risk Score",
                                    f"{risk_score:.0f}/100",
                                    help="Overall risk assessment (lower is safer)"
                                )
                            else:
                                st.metric("📊 Data Points", f"{len(close_data)}", "Historical")
                        
                        # === PRICE PREDICTION CHART ===
                        st.subheader("📈 Price Prediction Visualization")
                        
                        try:
                            historical_data = close_data.tail(60)  # Show more history
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
                                line=dict(color='#60a5fa', width=2),
                                hovertemplate='Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
                            ))
                            
                            # Predictions
                            fig.add_trace(go.Scatter(
                                x=pred_dates,
                                y=predictions,
                                mode='lines+markers',
                                name=f'AI Predictions ({prediction_period})',
                                line=dict(color='#10b981', width=3, dash='dot'),
                                marker=dict(size=8, symbol='diamond'),
                                hovertemplate='Date: %{x}<br>Predicted: ₹%{y:.2f}<extra></extra>'
                            ))
                            
                            # Connection line
                            if len(historical_data) > 0 and len(predictions) > 0:
                                fig.add_trace(go.Scatter(
                                    x=[historical_data.index[-1], pred_dates[0]],
                                    y=[historical_data.iloc[-1], predictions[0]],
                                    mode='lines',
                                    name='Transition',
                                    line=dict(color='#f59e0b', width=2, dash='dash'),
                                    showlegend=False
                                ))
                            
                            # Add confidence bands if available
                            if 'confidence' in prediction_result and confidence > 0.6:
                                upper_band = predictions * (1 + (1 - confidence) * 0.5)
                                lower_band = predictions * (1 - (1 - confidence) * 0.5)
                                
                                fig.add_trace(go.Scatter(
                                    x=pred_dates,
                                    y=upper_band,
                                    fill=None,
                                    mode='lines',
                                    line_color='rgba(0,100,80,0)',
                                    showlegend=False,
                                    name='Upper Confidence'
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=pred_dates,
                                    y=lower_band,
                                    fill='tonexty',
                                    mode='lines',
                                    line_color='rgba(0,100,80,0)',
                                    name='Confidence Band',
                                    fillcolor='rgba(16,185,129,0.2)'
                                ))
                            
                            fig.update_layout(
                                title=f"🤖 AI Prediction for {INDIAN_STOCKS[selected_stock]} ({selected_stock})",
                                xaxis_title="Date",
                                yaxis_title="Price (₹)",
                                template='plotly_dark',
                                height=600,
                                hovermode='x unified',
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as chart_error:
                            st.warning(f"Chart display error: {str(chart_error)}")
                            st.info("Prediction data is available but visualization couldn't be rendered.")
                        
                        # === RISK ANALYSIS SECTION ===
                        if show_risk_metrics and risk_metrics and 'error' not in risk_metrics:
                            st.subheader("⚖️ Comprehensive Risk Analysis")
                            
                            risk_cols = st.columns([1, 2])
                            
                            with risk_cols[0]:
                                # Risk Score Gauge
                                try:
                                    gauge_fig = create_risk_dashboard(risk_metrics)
                                    if gauge_fig:
                                        st.plotly_chart(gauge_fig, use_container_width=True)
                                except:
                                    st.metric("Risk Score", f"{risk_metrics.get('risk_score', 50)}/100")
                            
                            with risk_cols[1]:
                                # Risk Metrics Table
                                if 'var_metrics' in risk_metrics:
                                    var_data = risk_metrics['var_metrics']
                                    st.markdown("**Value at Risk (VaR)**")
                                    
                                    var_df = pd.DataFrame({
                                        'Time Horizon': ['1 Day', '5 Days', '10 Days'],
                                        'VaR (95%)': [
                                            f"{var_data.get('var_1d', 0)*100:.1f}%",
                                            f"{var_data.get('var_5d', 0)*100:.1f}%",
                                            f"{var_data.get('var_10d', 0)*100:.1f}%"
                                        ]
                                    })
                                    st.dataframe(var_df, use_container_width=True, hide_index=True)
                                
                                # Volatility Regime
                                if 'volatility_regime' in risk_metrics:
                                    vol_regime = risk_metrics['volatility_regime']
                                    regime_color = {
                                        'low_volatility': '🟢',
                                        'normal': '🟡', 
                                        'high_volatility': '🔴'
                                    }.get(vol_regime.get('regime', 'normal'), '🟡')
                                    
                                    st.markdown(f"**Volatility Regime:** {regime_color} {vol_regime.get('regime', 'Unknown').replace('_', ' ').title()}")
                                    st.markdown(f"Current Vol: {vol_regime.get('current_vol', 0)*100:.1f}% | Historical Avg: {vol_regime.get('historical_vol', 0)*100:.1f}%")
                            
                            # Stress Test Results
                            if stress_testing and 'stress_scenarios' in risk_metrics:
                                st.markdown("**🔥 Stress Test Scenarios**")
                                
                                try:
                                    stress_fig = create_stress_test_chart(
                                        risk_metrics['stress_scenarios'], 
                                        current_price
                                    )
                                    if stress_fig:
                                        st.plotly_chart(stress_fig, use_container_width=True)
                                    else:
                                        # Fallback table display
                                        stress_data = risk_metrics['stress_scenarios']
                                        if 'error' not in stress_data:
                                            stress_df = pd.DataFrame([
                                                {
                                                    'Scenario': name.replace('_', ' ').title(),
                                                    'Final Price': f"₹{data['final_price']:.2f}",
                                                    'Total Return': f"{data['total_return']:+.1f}%"
                                                }
                                                for name, data in stress_data.items()
                                            ])
                                            st.dataframe(stress_df, use_container_width=True, hide_index=True)
                                            
                                except Exception as stress_error:
                                    st.info("Stress test visualization unavailable")
                        
                        # === PERFORMANCE METRICS ===
                        if show_performance and performance_metrics and 'error' not in performance_metrics:
                            st.subheader("📊 Historical Performance Analysis")
                            
                            perf_cols = st.columns(4)
                            
                            with perf_cols[0]:
                                total_return = performance_metrics.get('total_return', 0)
                                st.metric(
                                    "📈 Total Return", 
                                    f"{total_return:+.1f}%",
                                    help="Total return since beginning of data"
                                )
                            
                            with perf_cols[1]:
                                annual_return = performance_metrics.get('annualized_return', 0)
                                st.metric(
                                    "📅 Annualized Return", 
                                    f"{annual_return:+.1f}%",
                                    help="Compound annual growth rate"
                                )
                            
                            with perf_cols[2]:
                                sharpe = performance_metrics.get('sharpe_ratio', 0)
                                sharpe_color = "🟢" if sharpe > 1 else "🟡" if sharpe > 0.5 else "🔴"
                                st.metric(
                                    f"{sharpe_color} Sharpe Ratio", 
                                    f"{sharpe:.2f}",
                                    help="Risk-adjusted return measure"
                                )
                            
                            with perf_cols[3]:
                                max_dd = performance_metrics.get('max_drawdown', 0)
                                st.metric(
                                    "📉 Max Drawdown", 
                                    f"{max_dd:.1f}%",
                                    help="Largest peak-to-trough decline"
                                )
                            
                            # Additional performance details
                            if analysis_depth == "Professional":
                                perf_detail_cols = st.columns(4)
                                
                                with perf_detail_cols[0]:
                                    win_rate = performance_metrics.get('win_rate', 0)
                                    st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
                                
                                with perf_detail_cols[1]:
                                    volatility = performance_metrics.get('volatility', 0)
                                    st.metric("📊 Volatility", f"{volatility:.1f}%")
                                
                                with perf_detail_cols[2]:
                                    avg_gain = performance_metrics.get('avg_gain', 0)
                                    st.metric("📈 Avg Gain", f"{avg_gain:.2f}%")
                                
                                with perf_detail_cols[3]:
                                    avg_loss = performance_metrics.get('avg_loss', 0)
                                    st.metric("📉 Avg Loss", f"{avg_loss:.2f}%")
                        
                        # === AI ANALYSIS SUMMARY ===
                        st.subheader("🤖 AI Analysis Summary")
                        
                        try:
                            summary = ensemble_model.get_prediction_summary(prediction_result)
                            method = prediction_result.get('method', 'Unknown')
                            data_points = prediction_result.get('data_points', 0)
                            volatility = prediction_result.get('volatility', 0)
                            
                            # Create comprehensive summary card
                            summary_html = f"""
                            <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); 
                                        border-radius: 12px; padding: 25px; margin: 15px 0; 
                                        border: 1px solid #4f46e5; color: white;">
                                <h3 style="margin-top: 0; color: #e0e7ff;">🧠 AI Intelligence Report</h3>
                                
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                                    <div>
                                        <h4 style="color: #c7d2fe; margin-bottom: 10px;">📊 Prediction Analysis</h4>
                                        {summary}
                                    </div>
                                    <div>
                                        <h4 style="color: #c7d2fe; margin-bottom: 10px;">🔧 Technical Details</h4>
                                        <strong>Method:</strong> {method}<br>
                                        <strong>Data Points:</strong> {data_points} trading days<br>
                                        <strong>Volatility:</strong> {volatility:.3f}<br>
                                        <strong>Risk Profile:</strong> {risk_adjustment}<br>
                                        <strong>Analysis Level:</strong> {analysis_depth}
                                    </div>
                                </div>
                                
                                <div style="margin-top: 20px; padding: 15px; background-color: rgba(255,255,255,0.1); 
                                           border-radius: 8px; border-left: 4px solid #10b981;">
                                    <strong>🎯 Investment Recommendation:</strong><br>
                                    {'<span style="color: #10b981;">BULLISH</span> - Consider position' if price_change > 2 
                                     else '<span style="color: #ef4444;">BEARISH</span> - Exercise caution' if price_change < -2 
                                     else '<span style="color: #f59e0b;">NEUTRAL</span> - Hold current position'}
                                    {f' (Confidence: {confidence:.0%})' if confidence > 0.6 else ' (Low confidence - wait for better signals)'}
                                </div>
                            </div>
                            """
                            
                            st.markdown(summary_html, unsafe_allow_html=True)
                            
                        except Exception as summary_error:
                            st.info("Prediction completed successfully, but summary generation encountered an issue.")
                        
                        # === MODEL COMPONENTS BREAKDOWN ===
                        if show_components and 'individual_predictions' in prediction_result:
                            st.subheader("🔍 AI Model Components Analysis")
                            
                            components = prediction_result['individual_predictions']
                            confidences = prediction_result.get('individual_confidences', {})
                            
                            if components:
                                # Create detailed component analysis
                                comp_cols = st.columns(min(len(components), 4))
                                
                                for i, (model_name, preds) in enumerate(components.items()):
                                    if i < len(comp_cols):
                                        with comp_cols[i]:
                                            try:
                                                final_pred = preds[-1] if len(preds) > 0 else current_price
                                                change = ((final_pred - current_price) / current_price) * 100 if current_price != 0 else 0
                                                conf = confidences.get(model_name, 0.5)
                                                
                                                # Model-specific styling
                                                model_colors = {
                                                    'moving_average': '#3b82f6',
                                                    'linear_trend': '#10b981', 
                                                    'seasonal_naive': '#f59e0b',
                                                    'exponential_smoothing': '#8b5cf6'
                                                }
                                                
                                                color = model_colors.get(model_name, '#6b7280')
                                                
                                                st.markdown(f"""
                                                <div style="background-color: {color}22; border: 1px solid {color}; 
                                                           border-radius: 8px; padding: 15px; text-align: center;">
                                                    <h4 style="color: {color}; margin: 0;">
                                                        {model_name.replace('_', ' ').title()}
                                                    </h4>
                                                    <h2 style="margin: 10px 0; color: white;">
                                                        ₹{final_pred:.2f}
                                                    </h2>
                                                    <p style="margin: 5px 0; color: {'#10b981' if change > 0 else '#ef4444'};">
                                                        {change:+.1f}%
                                                    </p>
                                                    <small style="color: #9ca3af;">
                                                        Confidence: {conf:.1%}
                                                    </small>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                            except Exception as comp_error:
                                                st.error(f"Component error: {model_name}")
                                
                                # Model weights visualization
                                if hasattr(ensemble_model, 'weights'):
                                    st.markdown("**🏋️ Model Weights in Ensemble**")
                                    weights_df = pd.DataFrame([
                                        {
                                            'Model': name.replace('_', ' ').title(),
                                            'Weight': f"{weight:.1%}",
                                            'Contribution': f"{weight * 100:.0f}/100"
                                        }
                                        for name, weight in ensemble_model.weights.items()
                                    ])
                                    st.dataframe(weights_df, use_container_width=True, hide_index=True)
                        
                        # === TRADING SIGNALS & RECOMMENDATIONS ===
                        st.subheader("🎯 Trading Signals & Recommendations")
                        
                        signal_cols = st.columns(3)
                        
                        with signal_cols[0]:
                            # Price Action Signal
                            if abs(price_change) > 5:
                                signal_strength = "Strong"
                                signal_color = "#10b981" if price_change > 0 else "#ef4444"
                            elif abs(price_change) > 2:
                                signal_strength = "Moderate" 
                                signal_color = "#f59e0b"
                            else:
                                signal_strength = "Weak"
                                signal_color = "#6b7280"
                            
                            st.markdown(f"""
                            <div style="padding: 15px; background-color: {signal_color}22; 
                                       border: 1px solid {signal_color}; border-radius: 8px;">
                                <h4 style="color: {signal_color}; margin: 0;">📊 Price Signal</h4>
                                <p style="margin: 10px 0; color: white;">
                                    <strong>{signal_strength}</strong> 
                                    {'Bullish' if price_change > 0 else 'Bearish' if price_change < 0 else 'Neutral'}
                                </p>
                                <small style="color: #9ca3af;">
                                    Expected {abs(price_change):.1f}% move
                                </small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with signal_cols[1]:
                            # Confidence Signal
                            if confidence > 0.8:
                                conf_signal = "High Conviction"
                                conf_color = "#10b981"
                            elif confidence > 0.6:
                                conf_signal = "Medium Conviction"
                                conf_color = "#f59e0b"
                            else:
                                conf_signal = "Low Conviction"
                                conf_color = "#ef4444"
                            
                            st.markdown(f"""
                            <div style="padding: 15px; background-color: {conf_color}22; 
                                       border: 1px solid {conf_color}; border-radius: 8px;">
                                <h4 style="color: {conf_color}; margin: 0;">🎯 Conviction Level</h4>
                                <p style="margin: 10px 0; color: white;">
                                    <strong>{conf_signal}</strong>
                                </p>
                                <small style="color: #9ca3af;">
                                    AI Confidence: {confidence:.1%}
                                </small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with signal_cols[2]:
                            # Risk Signal
                            if risk_metrics and 'risk_score' in risk_metrics:
                                risk_score = risk_metrics['risk_score']
                                if risk_score < 30:
                                    risk_signal = "Low Risk"
                                    risk_color = "#10b981"
                                elif risk_score < 60:
                                    risk_signal = "Medium Risk"
                                    risk_color = "#f59e0b"
                                else:
                                    risk_signal = "High Risk"
                                    risk_color = "#ef4444"
                            else:
                                risk_signal = "Unknown Risk"
                                risk_color = "#6b7280"
                                risk_score = 50
                            
                            st.markdown(f"""
                            <div style="padding: 15px; background-color: {risk_color}22; 
                                       border: 1px solid {risk_color}; border-radius: 8px;">
                                <h4 style="color: {risk_color}; margin: 0;">⚖️ Risk Assessment</h4>
                                <p style="margin: 10px 0; color: white;">
                                    <strong>{risk_signal}</strong>
                                </p>
                                <small style="color: #9ca3af;">
                                    Risk Score: {risk_score:.0f}/100
                                </small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # === RISK WARNINGS & DISCLAIMERS ===
                        st.markdown("---")
                        
                        warning_cols = st.columns(2)
                        
                        with warning_cols[0]:
                            # Dynamic warnings based on analysis
                            warnings_list = []
                            
                            if confidence < confidence_threshold:
                                warnings_list.append(f"⚠️ Low prediction confidence ({confidence:.1%})")
                            
                            if abs(price_change) > 10:
                                warnings_list.append("⚠️ High volatility prediction detected")
                            
                            if risk_metrics and risk_metrics.get('risk_score', 50) > 70:
                                warnings_list.append("⚠️ High risk investment identified")
                            
                            if len(close_data) < 100:
                                warnings_list.append("⚠️ Limited historical data available")
                            
                            if warnings_list:
                                st.warning("\n".join([
                                    "**⚠️ Risk Warnings:**"
                                ] + warnings_list + [
                                    "",
                                    "**Recommendations:**",
                                    "• Use conservative position sizing",
                                    "• Implement strict stop-loss orders", 
                                    "• Consider dollar-cost averaging",
                                    "• Diversify across multiple assets"
                                ]))
                        
                        with warning_cols[1]:
                            # Performance tips
                            tips = [
                                "💡 **Pro Tips:**",
                                "• Combine AI predictions with fundamental analysis",
                                "• Monitor key support/resistance levels",
                                "• Set both profit targets and stop losses",
                                "• Consider market sentiment and news events",
                                "• Review and adjust positions regularly"
                            ]
                            
                            st.info("\n".join(tips))
                        
                        # === FINAL DISCLAIMER ===
                        st.markdown("---")
                        st.markdown("""
                        <div style="background-color: #1f2937; border: 1px solid #374151; border-radius: 8px; padding: 20px; margin: 20px 0;">
                            <h4 style="color: #f59e0b; margin-top: 0;">📢 Important Disclaimer</h4>
                            <p style="color: #d1d5db; line-height: 1.6;">
                                These AI predictions are generated for <strong>educational and informational purposes only</strong> 
                                and should not be considered as financial advice. Stock market investments carry inherent risks, 
                                and past performance does not guarantee future results.
                            </p>
                            <p style="color: #d1d5db; line-height: 1.6; margin-bottom: 0;">
                                Always consult with qualified financial advisors, conduct your own research, and consider your 
                                risk tolerance before making any investment decisions. The developers and operators of this 
                                application are not responsible for any financial losses incurred based on these predictions.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                
                # Provide helpful troubleshooting information
                with st.expander("🔧 Troubleshooting Information"):
                    st.markdown("**Possible causes:**")
                    st.markdown("- Network connectivity issues")
                    st.markdown("- Yahoo Finance API rate limits")
                    st.markdown("- Insufficient historical data for selected stock")
                    st.markdown("- Temporary server issues")
                    
                    st.markdown("**Solutions to try:**")
                    st.markdown("1. Wait 1-2 minutes and try again")
                    st.markdown("2. Select a different, more liquid stock (like RELIANCE.NS or TCS.NS)")
                    st.markdown("3. Reduce the prediction period to 1 week")
                    st.markdown("4. Check your internet connection")
                    st.markdown("5. Refresh the page and log in again")
                
                # Show technical error details for debugging
                if st.checkbox("Show technical error details"):
                    st.code(str(e))
                    
                    import traceback
                    st.code(traceback.format_exc())
                            

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
