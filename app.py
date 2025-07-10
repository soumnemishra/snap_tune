import streamlit as st
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load model and data
model = joblib.load('models/song_recommender.pkl')
songs_df = pd.read_csv('proceesed_songs/processed_songs2.csv')

# Image Mood Extraction
def get_dominant_colors(image_data, k=3):
    image = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshaped = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reshaped)
    colors = kmeans.cluster_centers_.astype(int)
    avg_brightness = np.mean(colors)
    mood = 'Upbeat' if avg_brightness > 150 else 'Sad' if avg_brightness < 75 else 'Neutral'
    return mood, colors

# Text Sentiment Analysis
def analyze_text_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)['compound']
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Recommend Songs
def recommend_songs(image_mood, text_sentiment, model, songs_df):
    mood_mapping = {
        'Upbeat': {'valence': 0.7, 'danceability': 0.8, 'energy': 0.9},
        'Neutral': {'valence': 0.5, 'danceability': 0.6, 'energy': 0.6},
        'Sad': {'valence': 0.3, 'danceability': 0.4, 'energy': 0.3}
    }

    if image_mood == 'Upbeat' and text_sentiment == 'Positive':
        features = mood_mapping['Upbeat']
    elif image_mood == 'Sad' or text_sentiment == 'Negative':
        features = mood_mapping['Sad']
    else:
        features = mood_mapping['Neutral']

    feature_df = pd.DataFrame([features])
    predicted_mood = model.predict(feature_df)[0]
    recommendations = songs_df[songs_df['mood'] == predicted_mood].sample(5)
    return recommendations, predicted_mood


# Streamlit App UI

st.set_page_config(page_title="SnapTune", layout="centered")
st.title(" SnapTune -  Song Recommender")

# Upload image
uploaded_image = st.file_uploader("Upload an image ", type=['jpg', 'png', 'jpeg'])

# Enter text
user_text = st.text_area("Enter your current thoughts or feelings (optional) ")

if st.button("Get Recommendations") and uploaded_image:
    # Predict mood from image
    image_mood, colors = get_dominant_colors(uploaded_image)

    # Predict mood from text
    text_mood = analyze_text_sentiment(user_text) if user_text.strip() != '' else 'Neutral'

    # Get recommended songs
    recommendations, predicted_mood = recommend_songs(image_mood, text_mood, model, songs_df)

    st.success(f" Predicted Mood: **{predicted_mood}** (Image: {image_mood}, Text: {text_mood})")

    st.subheader(" Recommended Songs")
    for idx, row in recommendations.iterrows():
        st.markdown(f"""
        ** {row['track_name']}**  
         *{row['artist_names']}*  
         *{row['genres']}*  
         Mood: *{row['mood']}*
        """)
else:
    st.info("Upload an image and (optionally) enter your thoughts to get started.")

