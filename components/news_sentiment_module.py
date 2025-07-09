# components/news_sentiment_module.py
import streamlit as st
from utils.data_fetcher import fetch_news_sources
from utils.sentiment_analysis import create_sentiment_trend_chart, create_sector_sentiment_chart, create_market_buzz_chart

def news_sentiment_page(mode):
    st.header("News & Sentiment Analysis")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        source_filter = st.selectbox("Filter by Source", ["All Sources", "Economic Times", "Business Standard", "Moneycontrol", "Live Mint"])
    
    with col2:
        sector_filter = st.selectbox("Filter by Sector", ["All Sectors", "Banking", "IT", "Pharma", "Auto", "FMCG", "Energy"])
    
    # Fetch news
    with st.spinner("Fetching latest news..."):
        news_data = fetch_news_sources()
    
    # Apply filters
    filtered_news = news_data
    
    if source_filter != "All Sources":
        filtered_news = [item for item in filtered_news if item['source'] == source_filter]
    
    if sector_filter != "All Sectors":
        # This is a simplified filter for demonstration
        sector_keywords = {
            "Banking": ["bank", "banking", "financial"],
            "IT": ["tech", "technology", "software", "hardware", "it"],
            "Pharma": ["pharma", "pharmaceutical", "drug", "medicine"],
            "Auto": ["auto", "car", "vehicle", "automobile"],
            "FMCG": ["consumer", "good", "product", "brand"],
            "Energy": ["energy", "oil", "gas", "power"]
        }.get(sector_filter, [])
        
        filtered_news = [
            item for item in filtered_news 
            if any(keyword.lower() in item['title'].lower() for keyword in sector_keywords)
        ]
    
    # Display news cards
    st.subheader(f"Latest News ({len(filtered_news)})")
    
    # Layout configuration for news cards
    for i, item in enumerate(filtered_news):
        # Determine colors based on sentiment
        if item['sentiment'] == 'positive':
            bg_color = "#1f2937"  # Dark gray
            border_color = "#10b981"  # Green
            sentiment_color = "#10b981"
            sentiment_bg = "#065f46"
        elif item['sentiment'] == 'negative':
            bg_color = "#1f2937"  # Dark gray
            border_color = "#ef4444"  # Red
            sentiment_color = "#ef4444"
            sentiment_bg = "#7f1d1d"
        else:
            bg_color = "#1f2937"  # Dark gray
            border_color = "#f59e0b"  # Yellow
            sentiment_color = "#f59e0b"
            sentiment_bg = "#78350f"
        
        # Create news card
        st.markdown(f"""
        <div style='
            background-color: {bg_color}; 
            border: 1px solid {border_color}; 
            border-radius: 8px; 
            padding: 20px; 
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        '>
            <h4 style='
                color: #ffffff; 
                margin-top: 0; 
                margin-bottom: 15px;
                font-size: 18px;
                line-height: 1.4;
            '>{item['title']}</h4>
            
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='display: flex; align-items: center; gap: 15px;'>
                    <span style='color: #9ca3af; font-size: 14px;'><strong>Source:</strong> {item['source']}</span>
                    <span style='color: #9ca3af; font-size: 14px;'><strong>Score:</strong> {item['score']:.2f}</span>
                </div>
                <div style='
                    background-color: {sentiment_bg}; 
                    color: {sentiment_color}; 
                    padding: 6px 12px; 
                    border-radius: 20px; 
                    font-weight: bold;
                    font-size: 12px;
                    text-transform: uppercase;
                '>
                    {item['sentiment']}
                </div>
            </div>
        </div>
    # Display news cards
    st.subheader(f"Latest News ({len(filtered_news)})")
    
    # Layout configuration for news cards
    for i, item in enumerate(filtered_news):
        # Determine colors based on sentiment
        if item['sentiment'] == 'positive':
            bg_color = "#1f2937"  # Dark gray
            border_color = "#10b981"  # Green
            sentiment_color = "#10b981"
            sentiment_bg = "#065f46"
        elif item['sentiment'] == 'negative':
            bg_color = "#1f2937"  # Dark gray
            border_color = "#ef4444"  # Red
            sentiment_color = "#ef4444"
            sentiment_bg = "#7f1d1d"
        else:
            bg_color = "#1f2937"  # Dark gray
            border_color = "#f59e0b"  # Yellow
            sentiment_color = "#f59e0b"
            sentiment_bg = "#78350f"
        
        # Create news card
        st.markdown(f"""
        <div style='
            background-color: {bg_color}; 
            border: 1px solid {border_color}; 
            border-radius: 8px; 
            padding: 20px; 
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        '>
            <h4 style='
                color: #ffffff; 
                margin-top: 0; 
                margin-bottom: 15px;
                font-size: 18px;
                line-height: 1.4;
            '>{item['title']}</h4>
            
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div style='display: flex; align-items: center; gap: 15px;'>
                    <span style='color: #9ca3af; font-size: 14px;'><strong>Source:</strong> {item['source']}</span>
                    <span style='color: #9ca3af; font-size: 14px;'><strong>Score:</strong> {item['score']:.2f}</span>
                </div>
                <div style='
                    background-color: {sentiment_bg}; 
                    color: {sentiment_color}; 
                    padding: 6px 12px; 
                    border-radius: 20px; 
                    font-weight: bold;
                    font-size: 12px;
                    text-transform: uppercase;
                '>
                    {item['sentiment']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sentiment Analysis Charts
    st.markdown("---")
    st.subheader("Market Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Overall Market Sentiment Trend")
        sentiment_trend_fig = create_sentiment_trend_chart(filtered_news)
        if sentiment_trend_fig:
            st.plotly_chart(sentiment_trend_fig, use_container_width=True)
        else:
            st.info("No sentiment data available for trend analysis.")
    
    with col2:
        st.markdown("### Sector-wise Sentiment Analysis")
        sector_sentiment_fig = create_sector_sentiment_chart(filtered_news)
        if sector_sentiment_fig:
            st.plotly_chart(sector_sentiment_fig, use_container_width=True)
        else:
            st.info("No sentiment data available for sector analysis.")
    
    # Market Buzz Indicator
    st.markdown("---")
    st.subheader("Market Buzz Index")
    
    market_buzz_fig = create_market_buzz_chart(filtered_news)
    if market_buzz_fig:
        st.plotly_chart(market_buzz_fig, use_container_width=True)
    else:
        st.info("No market buzz data available.")
    
    # Detailed Analysis (only visible in Pro mode)
    if mode == "Pro":
        st.markdown("---")
        st.subheader("Detailed Sentiment Analysis")
        
        st.markdown("#### How Sentiment Analysis Works")
        st.write("Our system uses natural language processing to analyze the sentiment of news articles. Here's how we classify them:")
        st.write("- Positive: Articles with positive sentiment score above +0.3")
        st.write("- Negative: Articles with negative sentiment score below -0.3")
        st.write("- Neutral: Articles with sentiment score between -0.3 and +0.3")
        
        st.markdown("#### Sentiment Score Distribution")
        st.info("Histogram showing distribution of sentiment scores would appear here in a full implementation.")
