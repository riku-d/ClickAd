import streamlit as st
import numpy as np
import joblib

# Load trained model and encoders
model = joblib.load(r"C:\Users\rohit\OneDrive\Desktop\ClickAd\adaboost_ctr_model.pkl")
encoders = joblib.load(r"C:\Users\rohit\OneDrive\Desktop\ClickAd\encoders.pkl")

product_encoder = encoders['product']
interest_encoder = encoders['user_interest']
ad_encoder = encoders['ad_category']

valid_products = list(product_encoder.classes_)
valid_interests = list(interest_encoder.classes_)
valid_ads = list(ad_encoder.classes_)

st.set_page_config(page_title="Ad Click Prediction", page_icon="🖱️")

st.title("🖱️ Advertisement Click Prediction with CTR Estimate")

# ---------- Field Explanations ----------
with st.expander("ℹ️ **Feature Descriptions** - Click to Expand"):
    st.markdown("""
    **Feature Explanations:**

    - **Product**: Select product name (as per training data).
    - **Campaign ID**: Unique campaign identifier.
    - **Webpage ID**: ID of webpage where the ad appears.
    - **Primary Product Category**: Main category (1 to 5).
    - **Secondary Product Category**: Optional sub-category.
    - **Gender**: User's gender.
    - **Age Level**: User's age bracket (1 = youngest, 5 = oldest).
    - **User Group ID**: Segment based on behavior/demographics.
    - **User Depth**: Engagement level (1 = low, 3 = high).
    - **City Development Index**: Development score of city (0 to 1).
    - **Hour of the Day**: Hour ad appears (0 to 23).
    - **Day of Week**: Day (0 = Monday, 6 = Sunday).
    - **User Interest**: User's known interest area.
    - **Ad Category**: Category the ad belongs to.
    - **Interest Match**: Does the ad match user's interest.
    """)

# ---------- Form Inputs ----------
st.header("📊 Prediction Form")

product_raw = st.selectbox("Product", valid_products)
campaign_id = st.number_input("Campaign ID", min_value=100000, max_value=999999, value=404347)
webpage_id = st.number_input("Webpage ID", min_value=10000, max_value=99999, value=53587)
product_category_1 = st.selectbox("Primary Product Category", [1, 2, 3, 4, 5])
product_category_2 = st.text_input("Secondary Product Category (Numeric, optional)", value="0")

gender = st.selectbox("Gender", ["Male", "Female"])
age_level = st.slider("Age Level (1 = Young, 5 = Oldest)", 1, 5, 3)
user_group_id = st.number_input("User Group ID", min_value=1, max_value=5, value=2)
user_depth = st.selectbox("User Depth (Engagement Level)", [1, 2, 3])

city_development_index = st.number_input("City Development Index (0-1)", min_value=0.0, max_value=1.0, value=0.5)
hour = st.slider("Hour of the Day", 0, 23, 12)
day_of_week = st.slider("Day of Week (0 = Mon, 6 = Sun)", 0, 6, 2)

user_interest_label = st.selectbox("User Interest", valid_interests)
ad_category_label = st.selectbox("Ad Category", valid_ads)
interest_match = st.radio("Does the Ad Match User's Interest?", ["Yes", "No"])

# ---------- Prediction ----------
if st.button("🔮 Predict Click"):

    product_encoded = product_encoder.transform([product_raw])[0]
    user_interest_encoded = interest_encoder.transform([user_interest_label])[0]
    ad_category_encoded = ad_encoder.transform([ad_category_label])[0]
    gender_encoded = 0 if gender == "Male" else 1
    interest_match_encoded = 1 if interest_match == "Yes" else 0

    try:
        product_category_2 = float(product_category_2)
    except:
        product_category_2 = 0.0

    input_data = np.array([[

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
        hour,
        day_of_week,
        user_interest_encoded,
        ad_category_encoded,
        interest_match_encoded,
        0  # Placeholder to match feature count
    ]])

    st.subheader("🎯 Prediction Result:")

    # CTR Probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0][1] * 100
        st.progress(int(proba))
        st.write(f"**Estimated CTR (Click Probability): {proba:.2f}%**")
    else:
        st.info("CTR percentage not available for this model.")

    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ Likely the user will CLICK the Ad!")
    else:
        st.warning("❌ Unlikely the user will click the Ad.")
