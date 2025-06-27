from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

app = FastAPI(title="Ad Insight API")

season_map = {
    "Diwali_Sale": [10, 11],
    "Winter_Wear": [11, 12, 1],
    "Summer_Wear": [4, 5],
    "Food": list(range(1, 13)),
    "Books": list(range(1, 13)),
    "Fashion": list(range(1, 13)),
    "Sports": list(range(1, 13)),
    "Electronics": list(range(1, 13)),
    "Generic": list(range(1, 13))
}

avg_conversion_rate = 0.05
avg_order_value = 300

MAX_BOOST = 0.05  # Maximum CTR boost per platform

class AdInput(BaseModel):
    age_level: Optional[int] = 25
    gender: Optional[str] = "unknown"
    budget: Optional[float] = 20000.0
    user_depth: Optional[int] = 1
    product_type: Optional[str] = "Generic"
    current_time: Optional[str] = None
    instagram_followers: Optional[int] = 0
    facebook_followers: Optional[int] = 0

@app.post("/analyze_ad")
def analyze_ad(input: AdInput):
    month = datetime.now().month
    relevant_months = season_map.get(input.product_type, season_map["Generic"])

    relevant = month in relevant_months
    relevance_msg = "Yes, this ad is relevant for current time." if relevant else "No, better to upload in another season."
    best_time = "Now" if relevant else ", ".join([datetime(2023, m, 1).strftime('%B') for m in relevant_months])

    # Followers normalized to millions, safe defaults
    insta_boost = min((input.instagram_followers or 0) / 1_000_000, MAX_BOOST)
    fb_boost = min((input.facebook_followers or 0) / 1_000_000, MAX_BOOST)

    ctr = (
        0.02
        + (input.age_level / 1000)
        + (0.01 if (input.gender or "").lower() == "female" else 0)
        + input.user_depth * 0.005
        + insta_boost
        + fb_boost
    )
    ctr = min(ctr, 0.25)  # Absolute CTR cap

    estimated_rev = round(ctr * input.budget * avg_conversion_rate * avg_order_value, 2)

    return {
        "relevance": relevance_msg,
        "best_time_to_upload": best_time,
        "predicted_ctr": round(ctr, 4),
        "estimated_revenue": estimated_rev
    }
