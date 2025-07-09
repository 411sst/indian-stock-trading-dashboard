# pages/news_sentiment.py
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
    num_cols = 3 if len(filtered_news) > 2 else len(filtered_news) or 1
    cols = st.columns(num_cols)
    
    for i, item in enumerate(filtered_news):
        with cols[i % num_cols]:
            sentiment_color = {
                "positive": "#dcffe4",
                "negative": "#ffe5e5",
                "neutral": "#fff3cd"
            }.get(item['sentiment'], "#ffffff")
            
            sentiment_text_color = {
                "positive": "#1e8e3e",
                "negative": "#d93020",
                "neutral": "#926900"
            }.get(item['sentiment'], "#000000")
            
            st.markdown(f"""
            <div style='background-color: {sentiment_color}; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                <h4 style='margin-top: 0;'>{item['title']}</h4>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;'>
                    <span style='font-size: 0.9em; color: #666;'>{item['source']}</span>
                    <span style='font-size: 0.9em; color: #666;'>Score: {item['score']:.2f}</span>
                </div>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='background-color: {sentiment_text_color}66; color: white; padding: 3px 6px; border-radius: 3px; font-weight: bold;'>
                        {item['sentiment'].capitalize()}
                    </span>
                    <button style='background-color: #1a73e8; color: white; border: none; border-radius: 3px; padding: 5px 10px; cursor: pointer;'>
                        Read More →
                    </button>
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