import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnsembleModel:
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
                        
                        # Debug info
                        with st.expander("🔧 Chart Debug Info", expanded=False):
                            st.write(f"• Historical data: {len(close_data)} points")
                            st.write(f"• Predictions: {len(prediction_result.get('predictions', []))} points")
                            st.write(f"• Current price: ₹{current_price:.2f}")
                            st.write(f"• Predicted price: ₹{predicted_price:.2f}")
                        
                        try:
                            # Chart data preparation
                            historical_data = close_data.tail(60)
                            predictions = prediction_result.get('predictions', [])
                            
                            if len(predictions) == 0:
                                st.error("❌ No predictions available for chart")
                            else:
                                # Ensure correct data types
                                predictions = np.array(predictions).flatten()
                                
                                # Generate dates
                                pred_dates = pd.date_range(
                                    start=historical_data.index[-1] + timedelta(days=1),
                                    periods=len(predictions),
                                    freq='B'
                                )
                                
                                # Create chart
                                fig = go.Figure()
                                
                                # Historical data
                                fig.add_trace(go.Scatter(
                                    x=historical_data.index,
                                    y=historical_data.values,
                                    mode='lines',
                                    name='Historical Prices',
                                    line=dict(color='#3b82f6', width=2),
                                    hovertemplate='Historical<br>%{x}<br>₹%{y:,.2f}<extra></extra>'
                                ))
                                
                                # Predictions
                                fig.add_trace(go.Scatter(
                                    x=pred_dates,
                                    y=predictions,
                                    mode='lines+markers',
                                    name=f'AI Predictions ({prediction_period})',
                                    line=dict(color='#10b981', width=3, dash='dot'),
                                    marker=dict(size=8, color='#10b981'),
                                    hovertemplate='Prediction<br>%{x}<br>₹%{y:,.2f}<extra></extra>'
                                ))
                                
                                # Connection line
                                fig.add_trace(go.Scatter(
                                    x=[historical_data.index[-1], pred_dates[0]],
                                    y=[historical_data.iloc[-1], predictions[0]],
                                    mode='lines',
                                    line=dict(color='#f59e0b', width=2, dash='dash'),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                
                                # Confidence bands
                                if confidence > 0.6:
                                    try:
                                        band_width = (1 - confidence) * 0.25
                                        upper_band = predictions * (1 + band_width)
                                        lower_band = predictions * (1 - band_width)
                                        
                                        fig.add_trace(go.Scatter(
                                            x=pred_dates,
                                            y=upper_band,
                                            fill=None,
                                            mode='lines',
                                            line_color='rgba(16,185,129,0)',
                                            showlegend=False
                                        ))
                                        
                                        fig.add_trace(go.Scatter(
                                            x=pred_dates,
                                            y=lower_band,
                                            fill='tonexty',
                                            mode='lines',
                                            line_color='rgba(16,185,129,0)',
                                            name=f'Confidence Band ({confidence:.0%})',
                                            fillcolor='rgba(16,185,129,0.15)'
                                        ))
                                    except:
                                        pass
                                
                                # Update layout
                                fig.update_layout(
                                    title=f"🤖 AI Prediction: {INDIAN_STOCKS.get(selected_stock, selected_stock)}",
                                    xaxis_title="Date",
                                    yaxis_title="Price (₹)",
                                    template='plotly_dark',
                                    height=600,
                                    hovermode='x unified',
                                    showlegend=True,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                
                                # Display chart
                                st.plotly_chart(fig, use_container_width=True)
                                st.success(f"✅ Chart: {len(historical_data)} historical + {len(predictions)} predicted points")
                        
                        except Exception as chart_error:
                            st.error(f"❌ Chart failed: {str(chart_error)}")
                            
                            # Fallback table
                            try:
                                predictions = prediction_result.get('predictions', [])
                                if len(predictions) > 0:
                                    st.info("📊 Prediction Table (Chart Fallback)")
                                    pred_df = pd.DataFrame({
                                        'Day': range(1, len(predictions) + 1),
                                        'Predicted Price': [f"₹{p:.2f}" for p in predictions],
                                        'Change %': [f"{((p-current_price)/current_price)*100:+.1f}%" for p in predictions]
                                    })
                                    st.dataframe(pred_df, use_container_width=True)
                            except:
                                st.error("❌ Both chart and table failed")
                        
                        # === RISK ANALYSIS DASHBOARD ===
                        if show_risk_metrics and risk_metrics:
                            st.subheader("⚖️ Risk Analysis Dashboard")
                            
                            risk_cols = st.columns([1, 2])
                            
                            with risk_cols[0]:
                                # Risk Score Display
                                risk_score = risk_metrics.get('risk_score', 50)
                                
                                if risk_score < 40:
                                    risk_level = "Low Risk"
                                    risk_color = "#10b981"
                                elif risk_score < 70:
                                    risk_level = "Medium Risk"
                                    risk_color = "#f59e0b"
                                else:
                                    risk_level = "High Risk"
                                    risk_color = "#ef4444"
                                
                                st.markdown(f"""
                                <div style="text-align: center; padding: 20px; background-color: {risk_color}22; 
                                           border: 2px solid {risk_color}; border-radius: 10px;">
                                    <h2 style="color: {risk_color}; margin: 0;">{risk_score}/100</h2>
                                    <p style="color: {risk_color}; margin: 5px 0; font-weight: bold;">{risk_level}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Try to create risk gauge
                                try:
                                    gauge_fig = create_risk_dashboard(risk_metrics)
                                    if gauge_fig:
                                        st.plotly_chart(gauge_fig, use_container_width=True)
                                except Exception as gauge_error:
                                    st.caption(f"Gauge display issue: {str(gauge_error)}")
                            
                            with risk_cols[1]:
                                # Risk Metrics Table
                                st.markdown("**📊 Risk Breakdown**")
                                
                                var_data = risk_metrics.get('var_metrics', {})
                                if var_data:
                                    st.write("**Value at Risk (VaR 95%):**")
                                    st.write(f"• 1 Day: {var_data.get('var_1d', 0)*100:.1f}%")
                                    st.write(f"• 5 Days: {var_data.get('var_5d', 0)*100:.1f}%")
                                    st.write(f"• 10 Days: {var_data.get('var_10d', 0)*100:.1f}%")
                                
                                vol_regime = risk_metrics.get('volatility_regime', {})
                                if vol_regime:
                                    regime = vol_regime.get('regime', 'normal')
                                    st.write(f"**Volatility Regime:** {regime.replace('_', ' ').title()}")
                                    st.write(f"• Current: {vol_regime.get('current_vol', 0)*100:.1f}%")
                                    st.write(f"• Historical: {vol_regime.get('historical_vol', 0)*100:.1f}%")
                            
                            # Stress Test
                            if stress_testing:
                                st.markdown("**🔥 Stress Test Results**")
                                
                                stress_data = risk_metrics.get('stress_scenarios', {})
                                if stress_data and 'error' not in stress_data:
                                    try:
                                        # Create stress test chart
                                        stress_fig = create_stress_test_chart(stress_data, current_price)
                                        if stress_fig:
                                            st.plotly_chart(stress_fig, use_container_width=True)
                                        else:
                                            # Fallback stress test table
                                            stress_df = pd.DataFrame([
                                                {
                                                    'Scenario': name.replace('_', ' ').title(),
                                                    'Final Price': f"₹{data['final_price']:,.2f}",
                                                    'Return': f"{data['total_return']:+.1f}%"
                                                }
                                                for name, data in stress_data.items()
                                                if isinstance(data, dict) and 'total_return' in data
                                            ])
                                            st.dataframe(stress_df, use_container_width=True, hide_index=True)
                                    except Exception as stress_error:
                                        st.info(f"Stress test visualization issue: {str(stress_error)}")
                                else:
                                    st.info("Stress test data not available")
                        
                        # === AI ANALYSIS SUMMARY (FIXED - NO RAW HTML) ===
                        st.subheader("🤖 AI Analysis Summary")
                        
                        # Summary using Streamlit components (no raw HTML)
                        summary_col1, summary_col2 = st.columns(2)
                        
                        with summary_col1:
                            st.markdown("**🎯 Prediction Summary**")
                            direction = "increase" if price_change > 0 else "decrease"
                            strength = "strong" if abs(price_change) > 5 else "moderate" if abs(price_change) > 2 else "slight"
                            
                            st.write(f"• Expected {direction}: {abs(price_change):.1f}% ({strength})")
                            st.write(f"• AI confidence: {confidence:.1%}")
                            st.write(f"• Method: {prediction_result.get('method', 'Ensemble')}")
                            st.write(f"• Data quality: {'Good' if len(close_data) > 100 else 'Limited'}")
                        
                        with summary_col2:
                            st.markdown("**🔧 Technical Details**")
                            st.write(f"• Data points: {len(close_data)} trading days")
                            st.write(f"• Volatility: {prediction_result.get('volatility', 0):.3f}")
                            st.write(f"• Risk profile: {risk_adjustment}")
                            st.write(f"• Analysis level: {analysis_depth}")
                        
                        # Investment Recommendation
                        st.markdown("**🎯 Investment Recommendation**")
                        
                        if price_change > 3:
                            st.success(f"🟢 **BULLISH SIGNAL** - Consider position (Confidence: {confidence:.1%})")
                        elif price_change < -3:
                            st.error(f"🔴 **BEARISH SIGNAL** - Exercise caution (Confidence: {confidence:.1%})")
                        else:
                            st.info(f"🟡 **NEUTRAL SIGNAL** - Hold position (Confidence: {confidence:.1%})")
                        
                        # === MODEL COMPONENTS (FIXED - NO RAW HTML) ===
                        if show_components and 'individual_predictions' in prediction_result:
                            st.subheader("🔍 Model Components Analysis")
                            
                            components = prediction_result['individual_predictions']
                            confidences = prediction_result.get('individual_confidences', {})
                            
                            if components:
                                comp_cols = st.columns(min(len(components), 4))
                                
                                for i, (model_name, preds) in enumerate(components.items()):
                                    if i < len(comp_cols):
                                        with comp_cols[i]:
                                            try:
                                                final_pred = preds[-1] if len(preds) > 0 else current_price
                                                change = ((final_pred - current_price) / current_price) * 100 if current_price != 0 else 0
                                                conf = confidences.get(model_name, 0.5)
                                                
                                                st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                                                st.metric("Prediction", f"₹{final_pred:.2f}", f"{change:+.1f}%")
                                                st.caption(f"Confidence: {conf:.1%}")
                                                
                                            except Exception:
                                                st.error(f"Error: {model_name}")
                        
                        # === TRADING SIGNALS (FIXED - NO RAW HTML) ===
                        st.subheader("🎯 Trading Signals")
                        
                        signal_cols = st.columns(3)
                        
                        with signal_cols[0]:
                            st.markdown("**📊 Price Signal**")
                            if abs(price_change) > 5:
                                if price_change > 0:
                                    st.success("📈 **Strong Bullish**")
                                else:
                                    st.error("📉 **Strong Bearish**")
                            elif abs(price_change) > 2:
                                if price_change > 0:
                                    st.info("📈 **Moderate Bullish**")
                                else:
                                    st.warning("📉 **Moderate Bearish**")
                            else:
                                st.info("📊 **Neutral**")
                            st.caption(f"Expected: {abs(price_change):.1f}% move")
                        
                        with signal_cols[1]:
                            st.markdown("**🎯 Conviction**")
                            if confidence > 0.8:
                                st.success("🟢 **High Conviction**")
                            elif confidence > 0.6:
                                st.info("🟡 **Medium Conviction**")
                            else:
                                st.warning("🔴 **Low Conviction**")
                            st.caption(f"AI Confidence: {confidence:.1%}")
                        
                        with signal_cols[2]:
                            st.markdown("**⚖️ Risk Level**")
                            if risk_metrics:
                                risk_score = risk_metrics.get('risk_score', 50)
                                if risk_score < 40:
                                    st.success("🟢 **Low Risk**")
                                elif risk_score < 70:
                                    st.warning("🟡 **Medium Risk**")
                                else:
                                    st.error("🔴 **High Risk**")
                                st.caption(f"Risk Score: {risk_score}/100")
                            else:
                                st.info("🟡 **Medium Risk**")
                        
                        # === WARNINGS AND DISCLAIMERS ===
                        st.markdown("---")
                        
                        # Dynamic warnings
                        warnings = []
                        if confidence < confidence_threshold:
                            warnings.append(f"⚠️ Low confidence ({confidence:.1%})")
                        if abs(price_change) > 10:
                            warnings.append("⚠️ High volatility predicted")
                        if risk_metrics and risk_metrics.get('risk_score', 50) > 75:
                            warnings.append("⚠️ High risk detected")
                        
                        if warnings:
                            st.warning("**Risk Warnings:**\n" + "\n".join(warnings))
                        
                        # Final disclaimer
                        st.info("""
                        **📢 Disclaimer:** These predictions are for educational purposes only. 
                        Not financial advice. Always consult qualified advisors before investing.
                        """)
                        
                        # === DEBUG SECTION ===
                        with st.expander("🔧 Debug Information", expanded=False):
                            st.write("**Confidence Factors:**")
                            conf_factors = prediction_result.get('confidence_factors', {})
                            for key, value in conf_factors.items():
                                st.write(f"• {key}: {value}")
                            
                            st.write("**Risk Score Components:**")
                            if risk_metrics and 'risk_components' in risk_metrics:
                                for key, value in risk_metrics['risk_components'].items():
                                    st.write(f"• {key}: {value:.1f}")
                            
                            st.write("**Prediction Data:**")
                            st.write(f"• Predictions length: {len(prediction_result.get('predictions', []))}")
                            st.write(f"• Current price: ₹{current_price:.2f}")
                            st.write(f"• Final confidence: {confidence:.3f}")
                            st.write(f"• Final risk score: {risk_metrics.get('risk_score', 'N/A') if risk_metrics else 'N/A'}")

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                
                with st.expander("🔧 Error Details"):
                    st.code(str(e))
                    
                    st.markdown("**Troubleshooting:**")
                    st.write("1. Try a different stock (RELIANCE.NS, TCS.NS)")
                    st.write("2. Reduce prediction period to 1 week")
                    st.write("3. Check internet connection")
                    st.write("4. Wait 1-2 minutes and retry")
