# utils/sentiment_analysis.py
import pandas as pd
import plotly.express as px
from textblob import TextBlob

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.3:
        return {"sentiment": "positive", "score": polarity}
    elif polarity < -0.3:
        return {"sentiment": "negative", "score": polarity}
    else:
        return {"sentiment": "neutral", "score": polarity}

def create_sentiment_trend_chart(news_data):
    """Create a sentiment trend chart from news data"""
    if not news_data:
        return None
        
    df = pd.DataFrame(news_data)
    
    # Create time series for sentiment trend
    df['date'] = pd.to_datetime('now')  # Mock date for demo purposes
    
    fig = px.line(
        df,
        x='date',
        y='score',
        title='Market Sentiment Trend',
        labels={'score': 'Sentiment Score', 'date': 'Time'},
        template='plotly_dark'
    )
    
    # Add color based on sentiment
    fig.update_traces(line=dict(color='green', width=2), selector=dict(name='positive'))
    fig.update_traces(line=dict(color='red', width=2), selector=dict(name='negative'))
    fig.update_traces(line=dict(color='gray', width=2), selector=dict(name='neutral'))
    
    return fig

def create_sector_sentiment_chart(news_data):
    """Create a bar chart comparing sentiment across sectors"""
    if not news_data:
        return None
        
    # Mock sector mapping for demo purposes
    sector_mapping = {
        "bank": "Banking",
        "banking": "Banking",
        "stock": "General Market",
        "market": "General Market",
        "nifty": "General Market",
        "sensex": "General Market",
        "pharma": "Pharma",
        "pharmaceutical": "Pharma",
        "auto": "Auto",
        "car": "Auto",
        "vehicle": "Auto",
        "technology": "IT",
        "software": "IT",
        "hardware": "IT",
        "tech": "IT",
        "it": "IT",
        "cement": "Construction",
        "construction": "Construction",
        "real estate": "Construction",
        "energy": "Energy",
        "oil": "Energy",
        "gas": "Energy"
    }
    
    sector_scores = {}
    sector_counts = {}
    
    for item in news_data:
        score = item['score']
        words = item['title'].lower().split()
        
        sectors_found = set()
        for word in words:
            for key, sector in sector_mapping.items():
                if key in word:
                    sectors_found.add(sector)
        
        if not sectors_found:
            sectors_found.add("Other")
        
        for sector in sectors_found:
            sector_scores[sector] = sector_scores.get(sector, 0) + score
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    # Calculate average scores
    avg_scores = {sector: sector_scores[sector]/sector_counts[sector] for sector in sector_scores}
    
    df = pd.DataFrame({
        'Sector': list(avg_scores.keys()),
        'Sentiment Score': list(avg_scores.values())
    })
    
    fig = px.bar(
        df,
        x='Sector',
        y='Sentiment Score',
        title='Sector-wise Sentiment Analysis',
        color='Sentiment Score',
        color_continuous_scale=[
            [0, 'red'],
            [0.5, 'white'],
            [1, 'green']
        ],
        range_color=(-1, 1),
        template='plotly_dark'
    )
    
    return fig

def create_market_buzz_chart(news_data):
    """Create a market buzz indicator showing activity across sectors"""
    if not news_data:
        return None
        
    # Mock sector mapping for demo purposes
    sector_mapping = {
        "bank": "Banking",
        "banking": "Banking",
        "stock": "Market",
        "market": "Market",
        "nifty": "Market",
        "sensex": "Market",
        "pharma": "Pharma",
        "pharmaceutical": "Pharma",
        "auto": "Auto",
        "car": "Auto",
        "vehicle": "Auto",
        "technology": "IT",
        "software": "IT",
        "hardware": "IT",
        "tech": "IT",
        "it": "IT",
        "cement": "Construction",
        "construction": "Construction",
        "real estate": "Construction",
        "energy": "Energy",
        "oil": "Energy",
        "gas": "Energy"
    }
    
    sector_counts = {}
    
    for item in news_data:
        words = item['title'].lower().split()
        
        sectors_found = set()
        for word in words:
            for key, sector in sector_mapping.items():
                if key in word:
                    sectors_found.add(sector)
        
        if not sectors_found:
            sectors_found.add("Other")
        
        for sector in sectors_found:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    # Normalize to 0-100 scale
    max_count = max(sector_counts.values()) if sector_counts else 1
    
    normalized_counts = {
        sector: min(100, count/max_count*100)
        for sector, count in sector_counts.items()
    }
    
    df = pd.DataFrame({
        'Sector': list(normalized_counts.keys()),
        'Buzz Level': list(normalized_counts.values())
    })
    
    fig = px.bar(
        df,
        x='Sector',
        y='Buzz Level',
        title='Market Buzz Index',
        color='Buzz Level',
        color_continuous_scale=[
            [0, 'lightblue'],
            [0.5, 'blue'],
            [1, 'darkblue']
        ],
        range_color=(0, 100),
        template='plotly_dark'
    )
    
    return fig