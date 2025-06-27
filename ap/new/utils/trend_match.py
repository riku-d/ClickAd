from pytrends.request import TrendReq

pytrends = TrendReq()

def compute_trend_match(keywords, category=None):
    try:
        # Combine category with keywords for more precise trend search
        search_terms = [f"{category} {kw}" if category else kw for kw in keywords]
        
        pytrends.build_payload(search_terms, timeframe='now 7-d')  # Last 7 days trends
        trend_data = pytrends.interest_over_time()
        
        if trend_data.empty:
            return 0
        
        score = trend_data.mean().mean()
        return min(score, 100)
    except:
        return 0
