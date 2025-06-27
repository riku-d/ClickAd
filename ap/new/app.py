import streamlit as st
import joblib
import base64
import tempfile
import requests
from utils.transcription import transcribe_audio
from utils.keyword_extraction import extract_keywords
from utils.trend_match import compute_trend_match
from utils.feature_engineering import build_feature_vector

model = joblib.load("models/adaboost_ctr_model.pkl")
encoders = joblib.load("models/encoders.pkl")
category_encoder = encoders['ad_category']

FASTAPI_URL = "http://127.0.0.1:8000/analyze_ad"

# Background Video
def add_bg_video(video_file):
    video_bytes = open(video_file, "rb").read()
    encoded_video = base64.b64encode(video_bytes).decode()
    st.markdown(f"""
        <style>
        .stApp {{ background: transparent; }}
        video#bgvid {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            z-index: -1;
            object-fit: cover;
        }}
        .navbar {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background-color: rgba(0,0,0,0.9);
            z-index: 9999;
            padding: 10px;
            display: flex;
            justify-content: center;
            gap: 20px;
            border-bottom: 2px solid rgba(0,255,204,0.3);
            backdrop-filter: blur(10px);
        }}
        .nav-btn {{
            background-color: rgba(255,255,255,0.1);
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s ease;
        }}
        .nav-btn:hover {{
            background-color: rgba(0,255,204,0.2);
            color: #00ffcc;
            transform: translateY(-2px);
        }}
        .main-content {{
            margin-top: 80px;
            padding: 20px;
        }}
        .center-card {{
            background-color: rgba(0,0,0,0.85);
            padding: 30px;
            border-radius: 12px;
            max-width: 700px;
            color: white;
            margin: 0 auto;
            text-align: center;
            box-shadow: 0 0 25px rgba(0,0,0,0.7);
        }}
        </style>
        <video autoplay muted loop id="bgvid">
            <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
        </video>
    """, unsafe_allow_html=True)

# Navbar with routing
def navbar():
    st.markdown('<div class="navbar">', unsafe_allow_html=True)
    pages = {"🏠 Home": "home", "🎯 Predict": "predict", "📊 Analyze": "analyze", "ℹ️ About": "about"}

    for label, target in pages.items():
        if st.button(label, key=f"nav_{target}", use_container_width=True):
            st.query_params.update({"page": target})
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# Pages
def home_section():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.markdown('<div class="center-card">', unsafe_allow_html=True)
    st.title("🎬 Welcome to Ad Analyzer")
    st.write("""
        This tool helps you predict the Click Through Rate (CTR) of your advertisements based on AI-powered analysis. 
        Navigate to 'Predict' to try it out!
    """)
    st.markdown('</div></div>', unsafe_allow_html=True)


def about_section():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("ℹ️ About This Project")
    st.write("""
        This project helps advertisers assess ad content before publishing.
        Powered by Machine Learning and NLP for transcript, keyword extraction, and CTR prediction.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def predict_section():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("🎯 Ad CTR Prediction Tool")

    st.write("""
    Upload your advertisement (video or audio) and get:

    - 📃 Automatic transcript  
    - 🔑 Keyword extraction  
    - 🔥 Trend alignment score  
    - 🎯 Predicted click-through likelihood  
    - 📅 AI-Powered Publishing Recommendations  
    """)

    uploaded_file = st.file_uploader("Upload Ad File (mp4, mp3, wav)", type=["mp4", "mp3", "wav"])

    st.sidebar.header("Additional Ad Details")
    time_of_day = st.sidebar.slider("Time of Day (0-23)", 0, 23, 12)
    categories = ["Food", "Books", "Fashion", "Sports", "Electronics"]
    selected_category = st.sidebar.selectbox("Ad Category", options=categories)
    interest_match = st.sidebar.slider("Interest Match Score (0-1)", 0.0, 1.0, 1.0)
    budget = st.sidebar.number_input("Ad Budget (₹)", 5000.0, 100000.0, 20000.0)
    ad_company = st.sidebar.text_input("Ad Company Name")
    instagram_followers = st.sidebar.number_input("Instagram Followers", min_value=0, value=100000)
    facebook_followers = st.sidebar.number_input("Facebook Followers", min_value=0, value=100000)

    if uploaded_file and st.button("🔍 Analyze Ad"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        transcript = transcribe_audio(temp_path)
        keywords = extract_keywords(transcript)
        trend_score = compute_trend_match(keywords)

        encoded_category = category_encoder.transform([selected_category])[0]

        features = build_feature_vector(
            trend_score=trend_score,
            interest_match=interest_match,
            category_encoded=encoded_category,
            time_of_day=time_of_day,
            gender_encoded=1,
            age_level=3,
            user_group_id=2,
            user_depth=2,
            city_development_index=0.5,
            day_of_week=2,
            product_encoded=1,
            user_interest_encoded=1,
            product_category_1=3,
            product_category_2=0.0,
            campaign_id=404347,
            webpage_id=53587
        )

        click_prob = model.predict_proba(features)[0][1] * 100

        st.markdown('<div class="center-card">', unsafe_allow_html=True)
        st.text_area("📝 Transcript", transcript, height=150)
        st.write("🔑 **Extracted Keywords:**", keywords)

        # Simple Relevance Match for Company Name
        company_keywords = ad_company.lower().split()
        keyword_match_count = sum(1 for kw in keywords if kw.lower() in company_keywords)
        relevance_pct = (keyword_match_count / len(keywords)) * 100 if keywords else 0

        st.metric("🔥 Trend Alignment Score (%)", f"{trend_score:.2f}")
        st.metric("🤝 Company Relevance Match (%)", f"{relevance_pct:.2f}")
        st.success(f"📈 Predicted Click Probability: {click_prob:.2f}%")

        with st.spinner("Fetching AI-Powered Insights..."):
            payload = {
                "age_level": 25,
                "gender": "female",
                "budget": budget,
                "user_depth": 2,
                "product_type": selected_category,
                "current_time": None,
                "instagram_followers": instagram_followers,
                "facebook_followers": facebook_followers
            }
            try:
                res = requests.post(FASTAPI_URL, json=payload)
                if res.ok:
                    insights = res.json()
                    st.info(f"✅ Relevance: {insights['relevance']}")
                    st.info(f"📅 Best Time to Upload: {insights['best_time_to_upload']}")
                    st.metric("💰 Estimated Revenue", f"₹ {insights['estimated_revenue']}")
                else:
                    st.error("Failed to fetch deeper insights.")
            except:
                st.error("Could not connect to AI API.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def analyze_section():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("📊 Deep Analyze Your Ads with AI")

    st.write("Enter advertisement details to get AI-driven predictions about relevance, CTR, and estimated revenue.")

    with st.form("analyze_form"):
        age = st.slider("Age Level", 18, 60, 25)
        gender = st.selectbox("Gender", ["unknown", "male", "female"])
        budget = st.number_input("Ad Budget (₹)", 5000.0, 100000.0, 20000.0)
        depth = st.slider("User Depth", 1, 5, 1)
        product = st.selectbox("Product Type", ["Generic", "Winter_Wear", "Summer_Wear", "Diwali_Sale", "Food", "Books", "Fashion", "Sports", "Electronics"])
        time_input = st.text_input("Time (Optional - e.g., 'winter' or '2024-12-15T10:00:00')")
        ad_company = st.text_input("Ad Company Name")
        instagram_followers = st.number_input("Instagram Followers", min_value=0, value=1000)
        facebook_followers = st.number_input("Facebook Followers", min_value=0, value=1000)

        submitted = st.form_submit_button("🔍 Analyze Ad")

    if submitted:
        st.info("Submitting your data to AI API...")

        payload = {
            "age_level": age,
            "gender": gender,
            "budget": budget,
            "user_depth": depth,
            "product_type": product,
            "current_time": time_input or None,
            "instagram_followers": instagram_followers,
            "facebook_followers": facebook_followers
        }

        st.write("Payload being sent:", payload)

        with st.spinner("Contacting AI API..."):
            try:
                res = requests.post(FASTAPI_URL, json=payload)
                
                st.write("Raw API Response:", res.status_code, res.text)

                if res.ok:
                    insights = res.json()
                    st.markdown('<div class="center-card">', unsafe_allow_html=True)
                    
                    st.success(f"✅ Relevance: {insights['relevance']}")
                    st.info(f"📅 Best Time to Upload: {insights['best_time_to_upload']}")
                    st.metric("📈 Predicted CTR (%)", insights['predicted_ctr'] * 100)
                    st.metric("💸 Estimated Revenue (₹)", insights['estimated_revenue'])

                    # Optional: Compare Ad Company name with keywords (if provided by backend or from other logic)
                    if ad_company:
                        st.write(f"**Ad Company Provided:** `{ad_company}` - Compare manually with keywords extracted in Predict section.")

                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Failed to fetch insights from API. Status: {res.status_code}, Response: {res.text}")

            except Exception as e:
                st.error(f"Error connecting to AI API: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# Run App
st.set_page_config(page_title="Ad Analyzer", page_icon="🎬")
add_bg_video("backgrounds/bg_v.mp4")
navbar()

page = st.query_params.get("page", ["home"])


if page == "home":
    home_section()
elif page == "predict":
    predict_section()
elif page == "analyze":
    analyze_section()
elif page == "about":
    about_section()
else:
    home_section()
