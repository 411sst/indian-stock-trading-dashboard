# utils/data_fetcher.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import requests
from bs4 import BeautifulSoup
import feedparser
from textblob import TextBlob
import ta
import os
from dotenv import load_dotenv
from .indian_stocks import INDIAN_STOCKS, INDICES
# Load environment variables
load_dotenv()

def get_market_status():
    """Check if Indian markets are currently open"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Market hours: 9:15 AM - 3:30 PM IST
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    is_weekday = now.weekday() < 5  # Monday-Friday
    is_market_time = market_open <= now <= market_close
    
    return is_market_time and is_weekday, now.strftime('%Y-%m-%d %H:%M:%S IST')

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_index_data():
    """Fetch data for NIFTY 50 and SENSEX indices"""
    index_data = {}
    
    for ticker, name in INDICES.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            prev_close = stock.info['previousClose']
            
            current_price = hist['Close'][-1] if not hist.empty else prev_close
            change = current_price - prev_close
            percent_change = (change / prev_close) * 100
            
            index_data[ticker] = {
                'name': name,
                'value': current_price,
                'change': change,
                'percent_change': percent_change
            }
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            continue
            
    return index_data

@st.cache_data(ttl=300)
def fetch_gainers_losers():
    """Get top 5 gainers and losers from NSE"""
    # This is a simplified approach since yfinance doesn't directly provide this data
    gainers = []
    losers = []
    
    # Sample stocks for demonstration
    sample_stocks = list(INDIAN_STOCKS.keys())[:10]
    
    for symbol in sample_stocks:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d")
            if hist.empty:
                continue
                
            prev_close = hist['Close'][-2] if len(hist) > 1 else stock.info.get('previousClose', 0)
            current_price = hist['Close'][-1]
            
            change = current_price - prev_close
            percent_change = (change / prev_close) * 100 if prev_close != 0 else 0
            
            if change > 0:
                gainers.append({
                    'symbol': symbol,
                    'price': current_price,
                    'change': percent_change
                })
            else:
                losers.append({
                    'symbol': symbol,
                    'price': current_price,
                    'change': abs(percent_change)
                })
                
        except Exception as e:
            continue
            
    # Sort by percentage change
    gainers.sort(key=lambda x: x['change'], reverse=True)
    losers.sort(key=lambda x: x['change'], reverse=True)
    
    return gainers[:5], losers[:5]

@st.cache_data(ttl=300)
def fetch_sector_performance():
    """Mock sector performance data"""
    sectors = ['Banking', 'IT', 'Pharma', 'Auto', 'FMCG']
    performance = [np.random.uniform(-2, 3) for _ in range(len(sectors))]
    
    return [
        {'sector': sector, 'performance': round(perf, 2)}
        for sector, perf in zip(sectors, performance)
    ]

@st.cache_data(ttl=300)
def fetch_stock_data(symbol, period='1mo'):
    """Fetch historical data for a specific stock"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return None
            
        # Calculate technical indicators
        hist['SMA20'] = ta.trend.sma_indicator(hist['Close'], window=20)
        hist['EMA50'] = ta.trend.ema_indicator(hist['Close'], window=50)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(hist['Close'])
        hist['BB_upper'] = bb.bollinger_hband()
        hist['BB_middle'] = bb.bollinger_mavg()
        hist['BB_lower'] = bb.bollinger_lband()
        
        # RSI
        hist['RSI'] = ta.momentum.rsi(hist['Close'])
        
        # MACD
        macd = ta.trend.MACD(hist['Close'])
        hist['MACD'] = macd.macd()
        hist['MACD_signal'] = macd.macd_signal()
        
        # Volume moving average
        hist['Vol_MA'] = hist['Volume'].rolling(window=20).mean()
        
        return hist
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def analyze_technical_indicators(df):
    """Analyze technical indicators to generate signals"""
    if df.empty or len(df) < 50:
        return {"signal": "NEUTRAL", "confidence": 50, "reasons": []}
    
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    reasons = []
    confidence = 50
    
    # RSI Analysis
    rsi = latest['RSI']
    if rsi < 30:
        reasons.append(f"Oversold RSI(14)={rsi:.2f}")
        confidence += 20
    elif rsi > 70:
        reasons.append(f"Overbought RSI(14)={rsi:.2f}")
        confidence += 20
    
    # MACD Analysis
    if latest['MACD'] > latest['MACD_signal'] and previous['MACD'] <= previous['MACD_signal']:
        reasons.append("MACD bullish crossover")
        confidence += 15
    elif latest['MACD'] < latest['MACD_signal'] and previous['MACD'] >= previous['MACD_signal']:
        reasons.append("MACD bearish crossover")
        confidence += 15
    
    # Moving Average Crossover
    if latest['SMA20'] > latest['EMA50'] and previous['SMA20'] <= previous['EMA50']:
        reasons.append("SMA20 crossed above EMA50")
        confidence += 10
    elif latest['SMA20'] < latest['EMA50'] and previous['SMA20'] >= previous['EMA50']:
        reasons.append("SMA20 crossed below EMA50")
        confidence += 10
    
    # Volume Analysis
    vol_ratio = latest['Volume'] / latest['Vol_MA']
    if vol_ratio > 1.5:
        reasons.append(f"High volume ({vol_ratio:.1f}x average)")
        confidence += 10
    
    # Determine overall signal
    if len([r for r in reasons if "bullish" in r or "oversold" in r]) > len([r for r in reasons if "bearish" in r or "overbought" in r]):
        signal = "BUY"
        confidence = min(confidence, 90)
    elif len([r for r in reasons if "bearish" in r or "overbought" in r]) > len([r for r in reasons if "bullish" in r or "oversold" in r]):
        signal = "SELL"
        confidence = min(confidence, 90)
    else:
        signal = "HOLD"
        confidence = max(40, 100 - confidence)
    
    return {
        "signal": signal,
        "confidence": confidence,
        "reasons": reasons
    }

@st.cache_data(ttl=300)
def fetch_news_sources():
    """Fetch news from multiple Indian financial sources"""
    try:
        # NewsAPI integration
        newsapi_key = os.getenv("NEWS_API_KEY")
        news_items = []
        
        if newsapi_key:
            # Economic Times
            et_url = f"https://newsapi.org/v2/top-headlines?sources=economic-times&apiKey={newsapi_key}"
            response = requests.get(et_url)
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                for article in articles:
                    sentiment = analyze_sentiment(article["title"])
                    news_items.append({
                        "title": article["title"],
                        "source": "Economic Times",
                        "sentiment": sentiment["sentiment"],
                        "score": sentiment["score"]
                    })
                    
            # Business Standard
            bs_url = f" https://newsapi.org/v2/top-headlines?sources=business-standard&apiKey={newsapi_key}"
            response = requests.get(bs_url)
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                for article in articles:
                    sentiment = analyze_sentiment(article["title"])
                    news_items.append({
                        "title": article["title"],
                        "source": "Business Standard",
                        "sentiment": sentiment["sentiment"],
                        "score": sentiment["score"]
                    })
        
        # Moneycontrol RSS Feed
        mc_feed = feedparser.parse(" https://www.moneycontrol.com/rss/buzzingstocks.xml ")
        for entry in mc_feed.entries[:5]:
            sentiment = analyze_sentiment(entry.title)
            news_items.append({
                "title": entry.title,
                "source": "Moneycontrol",
                "sentiment": sentiment["sentiment"],
                "score": sentiment["score"]
            })
            
        # Livemint RSS Feed
        lm_feed = feedparser.parse("https://www.livemint.com/market/rssfeedlatest.cms ")
        for entry in lm_feed.entries[:5]:
            sentiment = analyze_sentiment(entry.title)
            news_items.append({
                "title": entry.title,
                "source": "Live Mint",
                "sentiment": sentiment["sentiment"],
                "score": sentiment["score"]
            })
            
        return news_items
        
    except Exception as e:
        st.warning(f"Error fetching news: {str(e)}. Using mock news data.")
        # Return mock data if API fails
        return [
            {"title": "Nifty 50 hits record high on FII inflows", "source": "Economic Times", "sentiment": "positive", "score": 0.85},
            {"title": "Banking sector shows strong Q2 results", "source": "Business Standard", "sentiment": "positive", "score": 0.78},
            {"title": "Crude oil prices rise affecting FMCG margins", "source": "Moneycontrol", "sentiment": "negative", "score": -0.65},
            {"title": "IT companies report steady growth in exports", "source": "The Hindu Business Line", "sentiment": "positive", "score": 0.82},
            {"title": "Auto sector faces supply chain challenges", "source": "Live Mint", "sentiment": "neutral", "score": 0.25},
            {"title": "Govt announces new infrastructure projects", "source": "Financial Express", "sentiment": "positive", "score": 0.91}
        ]

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

@st.cache_data(ttl=300)
def fetch_support_resistance_levels(df):
    """Calculate support and resistance levels from historical data"""
    if df.empty or len(df) < 50:
        return [], []
    
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    
    # Simple method to find support/resistance levels
    # This would be more sophisticated in production
    level_diff_threshold = np.std(closes[-50:]) * 0.5
    
    # Find significant price levels
    price_levels = []
    for i in range(5, len(closes)-5):
        if is_support(i, lows):
            price_levels.append(lows[i])
        elif is_resistance(i, highs):
            price_levels.append(highs[i])
    
    # Filter unique levels within reasonable range
    unique_levels = []
    for level in sorted(price_levels, reverse=True):
        if not unique_levels or abs(level - unique_levels[-1]) > level_diff_threshold:
            unique_levels.append(level)
    
    # Separate into support and resistance
    recent_price = closes[-1]
    supports = sorted([l for l in unique_levels if l < recent_price], reverse=True)[:3]
    resistances = sorted([l for l in unique_levels if l > recent_price])[:3]
    
    return supports, resistances

def is_support(i, lows):
    """Check if a point is a support level"""
    return lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]

def is_resistance(i, highs):
    """Check if a point is a resistance level"""
    return highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]

def calculate_portfolio_value(portfolio):
    """Calculate total portfolio value and returns"""
    total_value = 0
    total_investment = 0
    
    for holding in portfolio:
        try:
            stock = yf.Ticker(holding['symbol'])
            current_price = stock.history(period="1d")['Close'][-1]
        except:
            current_price = holding['buy_price']  # Use buy price if unable to fetch
            
        quantity = holding['quantity']
        buy_price = holding['buy_price']
        
        current_value = quantity * current_price
        investment_value = quantity * buy_price
        
        total_value += current_value
        total_investment += investment_value
    
    p_and_l = total_value - total_investment
    return_percentage = (p_and_l / total_investment) * 100 if total_investment > 0 else 0
    
    return {
        "total_value": total_value,
        "p_and_l": p_and_l,
        "return_percentage": return_percentage
    }