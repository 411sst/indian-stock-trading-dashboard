# 📈 Indian Stock Market Trading Dashboard

A comprehensive, professional-grade Indian stock market trading dashboard built with Streamlit. This application serves as both a practical trading tool for real market decisions and a showcase of advanced data analytics and machine learning capabilities.

## ✨ Key Features

### 🏛️ **Real-time Market Overview**
- Live NIFTY 50 and SENSEX index tracking
- Market status indicator (Open/Closed with IST time)
- Top gainers and losers from NSE
- Sector performance heatmap
- USD/INR impact analysis

### 📊 **Advanced Stock Analysis**
- Interactive candlestick charts with technical indicators
- RSI, MACD, Bollinger Bands, and Moving Averages
- Support and resistance level identification
- AI-powered buy/sell signals with confidence scores
- Historical data analysis with multiple timeframes

### 💼 **Portfolio Tracker**
- Manual portfolio entry and CSV import/export
- Real-time P&L calculations in INR
- Portfolio vs NIFTY 50 benchmark comparison
- Asset allocation visualization
- Individual stock performance tracking

### 📰 **News & Sentiment Analysis**
- Real-time news feed from multiple Indian financial sources
- ML-powered sentiment analysis using TextBlob
- Sector-wise sentiment distribution
- Market buzz indicator
- News filtering by source and sector

### 🎯 **Dual Mode Interface**
- **Beginner Mode**: Simplified signals and explanations
- **Pro Mode**: Detailed technical analysis with confidence metrics

## 🛠️ Technology Stack

- **Framework**: Streamlit
- **Data Source**: Yahoo Finance API (yfinance)
- **News APIs**: NewsAPI, RSS feeds (Economic Times, Moneycontrol, Live Mint)
- **Technical Analysis**: TA-Lib library
- **Sentiment Analysis**: TextBlob
- **Visualization**: Plotly
- **Deployment**: Streamlit Cloud

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NewsAPI key (free from [newsapi.org](https://newsapi.org/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/indian-stock-trading-dashboard.git
   cd indian-stock-trading-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   echo "NEWS_API_KEY=your_newsapi_key_here" > .env
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 📱 Usage

### Navigation
- Use the sidebar to switch between **Beginner** and **Pro** modes
- Navigate between four main sections using the radio buttons
- Monitor market status in real-time

### Trading Signals
- **Beginner Mode**: `📈 BUY Signal: Stock is oversold (RSI: 25) and recent news is positive`
- **Pro Mode**: `📈 BUY Signal (Confidence: 78%): RSI(14)=25, MACD bullish crossover, News sentiment: +0.65`

### Portfolio Management
- Add stocks manually or import from CSV
- View real-time P&L calculations
- Export portfolio data for external analysis

## 🏗️ Project Structure

```
indian-stock-trading-dashboard/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── sample_portfolio.csv      # Sample portfolio data
├── components/               # Page modules
│   ├── __init__.py
│   ├── market_overview_module.py
│   ├── stock_analysis_module.py
│   ├── portfolio_tracker_module.py
│   └── news_sentiment_module.py
└── utils/                    # Utility modules
    ├── __init__.py
    ├── data_fetcher.py       # Yahoo Finance & News APIs
    ├── technical_analysis.py # Technical indicators
    ├── sentiment_analysis.py # NLP sentiment analysis
    ├── portfolio_manager.py  # Portfolio operations
    └── indian_stocks.py      # Indian stock symbols
```

## 🔧 Configuration

### Supported Indian Stocks
- **NIFTY 50 components**: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, etc.
- **Major indices**: NIFTY 50 (^NSEI), SENSEX (^BSESN)
- **Trading hours**: 9:15 AM - 3:30 PM IST (Monday-Friday)

### News Sources
- Economic Times
- Business Standard
- Moneycontrol
- Live Mint
- Financial Express

## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add `NEWS_API_KEY` in app secrets
4. Deploy with one click

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
streamlit run app.py --server.runOnSave=true
```

## 📊 Performance Features

- **Caching**: 5-minute TTL for API calls to reduce latency
- **Error Handling**: Graceful fallbacks for API failures
- **Responsive Design**: Mobile-friendly interface
- **Real-time Updates**: Market hours detection for smart refresh rates

## 🔮 Future Enhancements

- [ ] Integration with broker APIs for live trading
- [ ] Machine learning price prediction models
- [ ] Options chain analysis
- [ ] Algorithmic trading strategies
- [ ] Advanced portfolio optimization
- [ ] Real-time alerts and notifications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔗 Connect

- **LinkedIn**: (https://linkedin.com/in/your-profile)
- **GitHub**: (https://www.linkedin.com/in/shrish-singh-thakur/)

## ⚠️ Disclaimer

This application is for educational and informational purposes only. It should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions.

## 🙏 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for providing stock data
- [NewsAPI](https://newsapi.org/) for news data
- [Streamlit](https://streamlit.io/) for the amazing framework
- [Plotly](https://plotly.com/) for interactive visualizations

---