def build_feature_vector(
    *,
    trend_score,
    interest_match,
    category_encoded,
    time_of_day,
    gender_encoded,
    age_level,
    user_group_id,
    user_depth,
    city_development_index,
    day_of_week,
    product_encoded,
    user_interest_encoded,
    product_category_1,
    product_category_2,
    campaign_id,
    webpage_id
):
    """
    Build dynamic 16-feature vector for CTR prediction.
    """

    feature_vector = [
        product_encoded,
        campaign_id,
        webpage_id,
        product_category_1,
        product_category_2,
        gender_encoded,
        age_level,
        user_group_id,
        user_depth,
        city_development_index,
        time_of_day,
        day_of_week,
        user_interest_encoded,
        category_encoded,
        interest_match,
        trend_score
    ]

    return [feature_vector]
