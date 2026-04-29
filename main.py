from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
SMOKING_MODEL_PATH = MODEL_DIR / "smoking_random_forest.joblib"
ALCOHOL_MODEL_PATH = MODEL_DIR / "alcohol_random_forest.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.joblib"


st.set_page_config(
    page_title="Instagram Lifestyle Predictor",
    page_icon="IG",
    layout="wide",
)


@st.cache_resource
def load_models():
    smoking_model = joblib.load(SMOKING_MODEL_PATH)
    alcohol_model = joblib.load(ALCOHOL_MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return smoking_model, alcohol_model, metadata


def probability_for_positive_class(model, row):
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    return float(probabilities[classes.index(1)])


def risk_label(probability):
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def build_input_form():
    st.sidebar.header("Basic details")
    age = st.sidebar.slider("Age", 13, 80, 24)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Non-binary"])
    urban_rural = st.sidebar.selectbox("Area", ["Urban", "Rural"])
    income_level = st.sidebar.selectbox("Income level", ["Low", "Middle", "Upper-middle", "High"])
    employment_status = st.sidebar.selectbox(
        "Employment status",
        ["Student", "Full-time employed", "Part-time employed", "Unemployed", "Retired"],
    )
    education_level = st.sidebar.selectbox(
        "Education level",
        ["High school", "Bachelor's", "Master's", "PhD", "Other"],
    )
    relationship_status = st.sidebar.selectbox(
        "Relationship status",
        ["Single", "In a relationship", "Married", "Divorced"],
    )

    st.subheader("Instagram usage")
    usage_col, content_col, account_col = st.columns(3)

    with usage_col:
        daily_active_minutes_instagram = st.slider("Daily active minutes", 0, 300, 90)
        sessions_per_day = st.slider("Sessions per day", 1, 30, 8)
        average_session_length_minutes = st.slider("Average session length", 1.0, 60.0, 12.0, 0.5)
        reels_watched_per_day = st.slider("Reels watched per day", 0, 300, 80)
        stories_viewed_per_day = st.slider("Stories viewed per day", 0, 200, 60)

    with content_col:
        posts_created_per_week = st.slider("Posts created per week", 0, 30, 3)
        likes_given_per_day = st.slider("Likes given per day", 0, 250, 70)
        comments_written_per_day = st.slider("Comments written per day", 0, 80, 10)
        dms_sent_per_week = st.slider("DMs sent per week", 0, 100, 20)
        dms_received_per_week = st.slider("DMs received per week", 0, 100, 20)

    with account_col:
        followers_count = st.number_input("Followers", min_value=0, max_value=1_000_000, value=1200, step=100)
        following_count = st.number_input("Following", min_value=0, max_value=1_000_000, value=900, step=100)
        account_creation_year = st.slider("Account creation year", 2010, 2026, 2018)
        notification_response_rate = st.slider("Notification response rate", 0.0, 1.0, 0.55, 0.01)
        linked_accounts_count = st.slider("Linked accounts", 0, 10, 1)

    st.subheader("Content and app behavior")
    behavior_col, settings_col = st.columns(2)

    with behavior_col:
        ads_viewed_per_day = st.slider("Ads viewed per day", 0, 80, 12)
        ads_clicked_per_day = st.slider("Ads clicked per day", 0, 20, 1)
        time_on_feed_per_day = st.slider("Time on feed per day", 0, 180, 40)
        time_on_explore_per_day = st.slider("Time on explore per day", 0, 180, 25)
        time_on_messages_per_day = st.slider("Time on messages per day", 0, 180, 15)
        time_on_reels_per_day = st.slider("Time on reels per day", 0, 180, 35)

    with settings_col:
        uses_premium_features = st.selectbox("Uses premium features", ["No", "Yes"])
        content_type_preference = st.selectbox("Content type preference", ["Mixed", "Photos", "Videos", "Stories"])
        preferred_content_theme = st.selectbox("Preferred content theme", ["Tech", "Fashion", "Food", "Fitness", "Travel", "Other"])
        privacy_setting_level = st.selectbox("Privacy setting", ["Private", "Public", "Friends only"])
        two_factor_auth_enabled = st.selectbox("Two-factor authentication", ["Yes", "No"])
        biometric_login_used = st.selectbox("Biometric login", ["No", "Yes"])
        subscription_status = st.selectbox("Subscription status", ["Free", "Paid"])
        user_engagement_score = st.slider("Engagement score", 0.0, 10.0, 5.0, 0.1)

    return pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "urban_rural": urban_rural,
                "income_level": income_level,
                "employment_status": employment_status,
                "education_level": education_level,
                "relationship_status": relationship_status,
                "daily_active_minutes_instagram": daily_active_minutes_instagram,
                "sessions_per_day": sessions_per_day,
                "posts_created_per_week": posts_created_per_week,
                "reels_watched_per_day": reels_watched_per_day,
                "stories_viewed_per_day": stories_viewed_per_day,
                "likes_given_per_day": likes_given_per_day,
                "comments_written_per_day": comments_written_per_day,
                "dms_sent_per_week": dms_sent_per_week,
                "dms_received_per_week": dms_received_per_week,
                "ads_viewed_per_day": ads_viewed_per_day,
                "ads_clicked_per_day": ads_clicked_per_day,
                "time_on_feed_per_day": time_on_feed_per_day,
                "time_on_explore_per_day": time_on_explore_per_day,
                "time_on_messages_per_day": time_on_messages_per_day,
                "time_on_reels_per_day": time_on_reels_per_day,
                "followers_count": followers_count,
                "following_count": following_count,
                "uses_premium_features": uses_premium_features,
                "notification_response_rate": notification_response_rate,
                "account_creation_year": account_creation_year,
                "average_session_length_minutes": average_session_length_minutes,
                "content_type_preference": content_type_preference,
                "preferred_content_theme": preferred_content_theme,
                "privacy_setting_level": privacy_setting_level,
                "two_factor_auth_enabled": two_factor_auth_enabled,
                "biometric_login_used": biometric_login_used,
                "linked_accounts_count": linked_accounts_count,
                "subscription_status": subscription_status,
                "user_engagement_score": user_engagement_score,
            }
        ]
    )


st.title("Instagram Usage Lifestyle Predictor")
st.caption("Random Forest model for estimating smoking and alcohol-risk likelihood from Instagram behavior.")

if not (SMOKING_MODEL_PATH.exists() and ALCOHOL_MODEL_PATH.exists() and METADATA_PATH.exists()):
    st.error("Model files were not found. Run `python train_model.py` first.")
    st.stop()

smoking_model, alcohol_model, metadata = load_models()
input_data = build_input_form()
input_data = input_data[metadata["feature_columns"]]

if st.button("Predict lifestyle risk", type="primary", use_container_width=True):
    smoking_probability = probability_for_positive_class(smoking_model, input_data)
    alcohol_probability = probability_for_positive_class(alcohol_model, input_data)

    smoking_risk = risk_label(smoking_probability)
    alcohol_risk = risk_label(alcohol_probability)

    result_col_1, result_col_2 = st.columns(2)
    with result_col_1:
        st.metric("Smoking likelihood", f"{smoking_probability * 100:.1f}%", smoking_risk)
        st.progress(smoking_probability)
    with result_col_2:
        st.metric("Alcohol-risk likelihood", f"{alcohol_probability * 100:.1f}%", alcohol_risk)
        st.progress(alcohol_probability)

    st.info(
        "This app is for academic prediction only. It should not be used as a medical diagnosis "
        "or as proof that a person smokes or drinks."
    )

with st.expander("Model details"):
    st.write(f"Smoking model accuracy: **{metadata['smoking_accuracy']:.2%}**")
    st.write(f"Alcohol-risk model accuracy: **{metadata['alcohol_accuracy']:.2%}**")
    st.write(metadata["smoking_target_note"])
    st.write(metadata["alcohol_target_note"])
