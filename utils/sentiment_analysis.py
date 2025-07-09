# utils/sentiment_analysis.py

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob with enhanced classification"""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.5:
        return {"sentiment": "strong_positive", "score": polarity}
    elif polarity > 0.1:
        return {"sentiment": "positive", "score": polarity}
    elif polarity < -0.5:
        return {"sentiment": "strong_negative", "score": polarity}
    elif polarity < -0.1:
        return {"sentiment": "negative", "score": polarity}
    else:
        return {"sentiment": "neutral", "score": polarity}

def create_sector_sentiment_chart(news_data):
    """Create a bar chart comparing sentiment across sectors with improved visualization"""
    if not news_data:
        return None
        
    # Define sector mapping with weights and categories
    sector_mapping = {
        "bank": {"sector": "Banking", "weight": 0.9},
        "banking": {"sector": "Banking", "weight": 1.0},
        "stock": {"sector": "Market", "weight": 0.7},
        "market": {"sector": "Market", "weight": 1.0},
        "nifty": {"sector": "Market", "weight": 0.8},
        "sensex": {"sector": "Market", "weight": 0.8},
        "pharma": {"sector": "Pharma", "weight": 1.0},
        "pharmaceutical": {"sector": "Pharma", "weight": 0.9},
        "auto": {"sector": "Auto", "weight": 1.0},
        "car": {"sector": "Auto", "weight": 0.8},
        "vehicle": {"sector": "Auto", "weight": 0.7},
        "technology": {"sector": "IT", "weight": 1.0},
        "software": {"sector": "IT", "weight": 0.8},
        "hardware": {"sector": "IT", "weight": 0.7},
        "tech": {"sector": "IT", "weight": 0.9},
        "it": {"sector": "IT", "weight": 1.0},
        "cement": {"sector": "Construction", "weight": 1.0},
        "construction": {"sector": "Construction", "weight": 1.0},
        "real estate": {"sector": "Construction", "weight": 0.8},
        "energy": {"sector": "Energy", "weight": 1.0},
        "oil": {"sector": "Energy", "weight": 0.9},
        "gas": {"sector": "Energy", "weight": 0.8}
    }
    
    sector_scores = {}
    sector_weights = {}
    
    for item in news_data:
        score = item['score']
        words = item['title'].lower().split()
        
        sectors_found = set()
        word_weights = []
        
        for word in words:
            for key, mapping in sector_mapping.items():
                if key in word:
                    sector = mapping['sector']
                    weight = mapping['weight']
                    sectors_found.add((sector, weight))
                    word_weights.append(weight)
        
        if not sectors_found:
            sectors_found.add(("Other", 0.5))
            word_weights.append(0.5)
        
        # Calculate weighted score contribution
        avg_weight = sum(word_weights) / len(word_weights) if word_weights else 0.5
        
        for sector, weight in sectors_found:
            weighted_score = score * weight * avg_weight
            sector_scores[sector] = sector_scores.get(sector, 0) + weighted_score
            sector_weights[sector] = sector_weights.get(sector, 0) + weight * avg_weight
    
    # Calculate average scores
    avg_scores = {sector: sector_scores[sector]/sector_weights[sector] for sector in sector_scores}
    
    # Sort by absolute value
    sorted_sectors = sorted(avg_scores.keys(), key=lambda x: abs(avg_scores[x]), reverse=True)
    sorted_scores = [avg_scores[sector] for sector in sorted_sectors]
    
    # Create color gradient based on strength
    colors = [
        f"rgb({255-int(255*(score+1)/2)}, {255*int(score>0)*(score+1)/2}, 0)"
        if score != 0 else "rgb(128, 128, 128)"
        for score in sorted_scores
    ]
    
    fig = px.bar(
        x=sorted_sectors,
        y=sorted_scores,
        title='Sector-wise Sentiment Analysis',
        labels={'x': 'Sector', 'y': 'Sentiment Score'},
        color=sorted_scores,
        color_continuous_scale=[
            [0, 'red'],
            [0.5, 'white'],
            [1, 'green']
        ],
        range_color=(-1, 1),
        template='plotly_dark'
    )
    
    # Add custom annotations for strong sentiments
    for i, (sector, score) in enumerate(zip(sorted_sectors, sorted_scores)):
        if abs(score) > 0.7:
            fig.add_annotation(
                x=sector,
                y=score,
                text=f"{score:.2f}",
                showarrow=False,
                font=dict(size=12, color="white"),
                bgcolor="rgba(0,0,0,0.5)",
                borderpad=4
            )
    
    return fig
