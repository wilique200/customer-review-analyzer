import streamlit as st
import nltk
import re
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import pandas as pd
import numpy as np

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Download TextBlob corpora
import subprocess
subprocess.run(['python', '-m', 'textblob.download_corpora'], 
               capture_output=True)
# --- Page Config ---
st.set_page_config(
    page_title="Customer Review Analyzer",
    page_icon="💬",
    layout="centered"
)

# --- Styling ---
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 0.5rem;
}
.sub-header {
    font-size: 1rem;
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}
.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    border-left: 4px solid #1f77b4;
}
.positive { border-left: 4px solid #28a745; }
.negative { border-left: 4px solid #dc3545; }
.neutral  { border-left: 4px solid #ffc107; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">💬 Customer Review Analyzer</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Paste any product review to instantly analyze sentiment,\
 extract key phrases and get business recommendations</div>',
            unsafe_allow_html=True)

# --- Text Cleaning ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
keep_words = {'not', 'no', 'never', 'very', 'but', 'however'}
stop_words = stop_words - keep_words

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# --- Sentiment Analysis ---
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        sentiment = 'Positive'
        confidence = min((polarity + 1) / 2 * 100, 99)
        emoji = '😊'
        color = 'positive'
        badge_color = '#28a745'
    elif polarity < -0.1:
        sentiment = 'Negative'
        confidence = min((abs(polarity) + 1) / 2 * 100, 99)
        emoji = '😞'
        color = 'negative'
        badge_color = '#dc3545'
    else:
        sentiment = 'Neutral'
        confidence = 60 + (1 - abs(polarity)) * 20
        emoji = '😐'
        color = 'neutral'
        badge_color = '#ffc107'

    return sentiment, confidence, polarity, subjectivity, emoji, color, badge_color

# --- Key Phrase Extraction ---
def extract_key_phrases(text, n=8):
    try:
        blob = TextBlob(text)
        noun_phrases = list(blob.noun_phrases)
    except Exception:
        noun_phrases = []

    cleaned = clean_text(text)
    words = cleaned.split()
    word_freq = pd.Series(words).value_counts()
    top_words = word_freq.head(n).index.tolist()

    phrases = list(set(noun_phrases + top_words))[:n]
    return phrases if phrases else top_words[:n]
    
# --- Business Recommendations ---
def get_recommendations(sentiment, polarity, subjectivity, text):
    text_lower = text.lower()
    recommendations = []

    if sentiment == 'Positive':
        recommendations = [
            "⭐ Feature this review prominently on your product page",
            "📣 Share on social media as customer testimonial",
            "🎁 Send a loyalty reward to this customer",
            "📊 Identify what drove this positive experience and replicate it",
            "💌 Ask this customer for a detailed case study"
        ]
    elif sentiment == 'Negative':
        recommendations = [
            "🚨 Flag for immediate customer service follow-up",
            "🔍 Investigate the specific issues mentioned",
            "💬 Respond publicly to show you care",
            "🔄 Offer replacement, refund or discount as goodwill",
            "📋 Log recurring complaints for product improvement"
        ]
        if 'shipping' in text_lower or 'delivery' in text_lower:
            recommendations.append("🚚 Review your shipping and logistics process")
        if 'quality' in text_lower or 'broke' in text_lower:
            recommendations.append("🔧 Escalate quality control review")
        if 'price' in text_lower or 'expensive' in text_lower:
            recommendations.append("💰 Review pricing strategy for this product")
    else:
        recommendations = [
            "📧 Follow up with customer for more detailed feedback",
            "🎯 Identify specific pain points in the review",
            "💡 Small improvements could convert this to a positive experience",
            "📊 Monitor if sentiment shifts with product updates",
            "🤝 Engage customer to build stronger relationship"
        ]

    if subjectivity > 0.7:
        recommendations.append("💭 Highly emotional review — prioritize personal outreach")

    return recommendations

# --- Sentiment Gauge ---
def create_gauge(polarity):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=polarity,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Score", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.1], 'color': "#ffcccc"},
                {'range': [-0.1, 0.1], 'color': "#fff3cd"},
                {'range': [0.1, 1], 'color': "#d4edda"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': polarity
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
    return fig

# --- Main App ---
st.markdown("### 📝 Enter Your Review")
review_text = st.text_area(
    "Paste a customer review here:",
    placeholder="e.g. This product exceeded my expectations! The quality is amazing and delivery was super fast...",
    height=150,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🔍 Analyze Review", use_container_width=True,
                             type="primary")

if analyze_btn:
    if not review_text.strip():
        st.warning("Please enter a review to analyze!")
    else:
        with st.spinner("Analyzing review..."):

            sentiment, confidence, polarity, subjectivity, emoji, color, badge_color = \
                analyze_sentiment(review_text)
            key_phrases = extract_key_phrases(review_text)
            recommendations = get_recommendations(
                sentiment, polarity, subjectivity, review_text)

        st.markdown("---")
        st.markdown("### 📊 Analysis Results")

        # Sentiment + Confidence
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card {color}">
                <h1>{emoji}</h1>
                <h3 style="color:{badge_color}">{sentiment}</h3>
                <p>Sentiment</p>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h1>🎯</h1>
                <h3>{confidence:.1f}%</h3>
                <p>Confidence</p>
            </div>""", unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h1>💭</h1>
                <h3>{subjectivity:.0%}</h3>
                <p>Subjectivity</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(create_gauge(polarity),
                           use_container_width=True)
        with col2:
            st.markdown("#### 🔑 Key Phrases")
            if key_phrases:
                for phrase in key_phrases[:6]:
                    st.markdown(f"• `{phrase}`")
            else:
                st.write("No key phrases detected")

        # Recommendations
        st.markdown("---")
        st.markdown("#### 💼 Business Recommendations")
        for rec in recommendations:
            st.markdown(rec)

        # Review Stats
        st.markdown("---")
        st.markdown("#### 📈 Review Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Word Count", len(review_text.split()))
        col2.metric("Characters", len(review_text))
        col3.metric("Polarity", f"{polarity:.3f}")
        col4.metric("Subjectivity", f"{subjectivity:.3f}")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.8rem'>
Built with ❤️ using Streamlit & NLP | Customer Review Analyzer v1.0
</div>""", unsafe_allow_html=True)
