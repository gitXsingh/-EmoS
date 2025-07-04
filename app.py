"""
Mental Health Risk Prediction Web App
A comprehensive tool for assessing mental health risk and providing personalized feedback
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from phq9 import PHQ9_QUESTIONS, PHQ9_OPTIONS, calculate_phq9_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🧠 EmoS - Mental Health Risk Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
    .wellness-score {
        font-size: 2rem;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        with open('mental_health_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run 'python train_model.py' first.")
        return None

def calculate_wellness_score(user_data):
    """Calculate mental wellness score (0-100) based on user inputs"""
    score = 0
    
    # Sleep duration (0-25 points)
    if user_data['sleep_duration'] >= 7:
        score += 25
    elif user_data['sleep_duration'] >= 6:
        score += 15
    elif user_data['sleep_duration'] >= 5:
        score += 5
    
    # Physical activity (0-20 points)
    if user_data['physical_activity'] == 'High':
        score += 20
    elif user_data['physical_activity'] == 'Moderate':
        score += 10
    
    # Stress level (0-25 points)
    if user_data['stress_level'] <= 3:
        score += 25
    elif user_data['stress_level'] <= 5:
        score += 15
    elif user_data['stress_level'] <= 7:
        score += 5
    
    # Mood swings (0-15 points)
    if not user_data['mood_swings']:
        score += 15
    
    # Screen time (0-15 points)
    if user_data['screen_time'] <= 6:
        score += 15
    elif user_data['screen_time'] <= 8:
        score += 8
    
    return min(score, 100)

def get_personalized_recommendations(user_data, risk_prediction):
    """Generate personalized recommendations based on user inputs"""
    recommendations = []
    
    # Sleep recommendations
    if user_data['sleep_duration'] < 7:
        recommendations.append("😴 **Sleep**: Aim for 7-9 hours of sleep per night for better mental health")
    
    # Stress recommendations
    if user_data['stress_level'] > 5:
        recommendations.append("🧘 **Stress Management**: Practice meditation, deep breathing, or yoga")
    
    # Physical activity recommendations
    if user_data['physical_activity'] != 'High':
        recommendations.append("🏃 **Exercise**: Increase physical activity to reduce stress and improve mood")
    
    # Mood swings recommendations
    if user_data['mood_swings']:
        recommendations.append("📝 **Mood Tracking**: Keep a mood journal to identify patterns and triggers")
    
    # Screen time recommendations
    if user_data['screen_time'] > 6:
        recommendations.append("📱 **Digital Wellness**: Reduce screen time and take regular breaks")
    
    # Social interaction recommendations
    if user_data['social_interactions'] < 5:
        recommendations.append("👥 **Social Connection**: Increase social interactions for better mental well-being")
    
    # Risk-specific recommendations
    if risk_prediction == 1:
        recommendations.append("🏥 **Professional Help**: Consider consulting a mental health professional")
        recommendations.append("📞 **Support**: Reach out to friends, family, or mental health hotlines")
    
    return recommendations

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 EmoS - Mental Health Risk Prediction</h1>', unsafe_allow_html=True)
    st.markdown("### Your AI-powered mental health assessment tool")
    
    # Load model
    model_data = load_model()
    if model_data is None:
        return
    
    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Main Assessment", "📋 PHQ-9 Depression Screening", "📊 About the Model"]
    )
    
    if page == "🏠 Main Assessment":
        show_main_assessment(model_data)
    elif page == "📋 PHQ-9 Depression Screening":
        show_phq9_screening()
    elif page == "📊 About the Model":
        show_model_info(model_data)

def show_main_assessment(model_data):
    """Display the main mental health assessment"""
    
    st.markdown("## 📊 Lifestyle & Mental Health Assessment")
    st.markdown("Please provide information about your lifestyle habits to assess your mental health risk.")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛏️ Sleep & Physical Health")
        sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 12.0, 7.0, 0.1)
        quality_of_sleep = st.slider("Sleep Quality (1-10)", 1, 10, 7)
        physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
        heart_rate = st.slider("Resting Heart Rate (bpm)", 50, 120, 75)
        daily_steps = st.slider("Daily Steps", 1000, 15000, 6000, 500)
    
    with col2:
        st.markdown("### 😰 Stress & Lifestyle")
        stress_level = st.slider("Stress Level (0-10)", 0, 10, 5)
        mood_swings = st.checkbox("Experience frequent mood swings?")
        screen_time = st.slider("Daily Screen Time (hours)", 1, 16, 6)
        social_interactions = st.slider("Daily Social Interactions (people)", 0, 20, 8)
    
    # Create feature vector for prediction
    user_data = {
        'sleep_duration': sleep_duration,
        'quality_of_sleep': quality_of_sleep,
        'physical_activity_level': {'Low': 30, 'Moderate': 50, 'High': 75}[physical_activity],
        'stress_level': stress_level,
        'heart_rate': heart_rate,
        'daily_steps': daily_steps,
        'physical_activity': physical_activity,
        'mood_swings': mood_swings,
        'screen_time': screen_time,
        'social_interactions': social_interactions
    }
    
    # Calculate derived features
    sleep_efficiency = sleep_duration * quality_of_sleep / 10
    activity_stress_ratio = user_data['physical_activity_level'] / (stress_level + 1)
    sleep_quality_ratio = quality_of_sleep / sleep_duration
    
    # Prepare features for model
    feature_names = model_data['feature_names']
    features = np.array([
        sleep_duration, quality_of_sleep, user_data['physical_activity_level'],
        stress_level, heart_rate, daily_steps,
        sleep_efficiency, activity_stress_ratio, sleep_quality_ratio
    ]).reshape(1, -1)
    
    # Make prediction
    if st.button("🔮 Predict Mental Health Risk", type="primary"):
        with st.spinner("Analyzing your data..."):
            # Scale features
            features_scaled = model_data['scaler'].transform(features)
            
            # Make prediction
            prediction = model_data['model'].predict(features_scaled)[0]
            prediction_proba = model_data['model'].predict_proba(features_scaled)[0]
            
            # Calculate wellness score
            wellness_score = calculate_wellness_score(user_data)
            
            # Display results
            st.markdown("---")
            st.markdown("## 📈 Assessment Results")
            
            # Risk prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 0:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("### 🟢 Risk Level")
                    st.markdown('<p class="risk-low">LOW RISK</p>', unsafe_allow_html=True)
                    st.markdown(f"Confidence: {prediction_proba[0]:.1%}")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("### 🔴 Risk Level")
                    st.markdown('<p class="risk-high">HIGH RISK</p>', unsafe_allow_html=True)
                    st.markdown(f"Confidence: {prediction_proba[1]:.1%}")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("### 🎯 Wellness Score")
                if wellness_score >= 80:
                    color = "#2ca02c"
                elif wellness_score >= 60:
                    color = "#ff7f0e"
                else:
                    color = "#d62728"
                
                st.markdown(f'<div class="wellness-score" style="background-color: {color}; color: white;">{wellness_score}/100</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Sleep Quality")
                if quality_of_sleep >= 8:
                    st.markdown("🟢 Excellent")
                elif quality_of_sleep >= 6:
                    st.markdown("🟡 Good")
                else:
                    st.markdown("🔴 Poor")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Personalized recommendations
            st.markdown("## 💡 Personalized Recommendations")
            recommendations = get_personalized_recommendations(user_data, prediction)
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
            
            # Feature importance visualization
            st.markdown("## 🔍 What Factors Matter Most")
            feature_importance = model_data['feature_importance']
            
            fig = px.bar(
                feature_importance.head(8),
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance in Mental Health Risk Prediction",
                labels={'importance': 'Importance Score', 'feature': 'Lifestyle Factors'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_phq9_screening():
    """Display the PHQ-9 depression screening questionnaire"""
    
    st.markdown("## 📋 PHQ-9 Depression Screening")
    st.markdown("The Patient Health Questionnaire-9 (PHQ-9) is a validated screening tool for depression.")
    st.markdown("**Instructions**: Over the last 2 weeks, how often have you been bothered by any of the following problems?")
    
    # Initialize session state for PHQ-9 responses
    if 'phq9_responses' not in st.session_state:
        st.session_state.phq9_responses = [0] * 9
    
    # Display questions
    for i, question in enumerate(PHQ9_QUESTIONS):
        st.markdown(f"**{i+1}. {question}**")
        response = st.selectbox(
            f"Response for question {i+1}",
            options=PHQ9_OPTIONS,
            index=st.session_state.phq9_responses[i],
            key=f"phq9_q{i}"
        )
        st.session_state.phq9_responses[i] = PHQ9_OPTIONS.index(response)
    
    # Calculate score
    if st.button("📊 Calculate PHQ-9 Score", type="primary"):
        try:
            result = calculate_phq9_score(st.session_state.phq9_responses)
            
            st.markdown("---")
            st.markdown("## 📈 PHQ-9 Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Score Summary")
                st.metric("Total Score", f"{result['score']}/27")
                st.metric("Severity Level", result['severity'])
                
                # Progress bar for score
                progress = result['score'] / result['max_score']
                st.progress(progress)
                st.markdown(f"Score: {result['score']} out of {result['max_score']}")
            
            with col2:
                st.markdown("### 🎯 Severity Interpretation")
                if result['score'] <= 4:
                    st.success("✅ Minimal depression - Continue maintaining good mental health!")
                elif result['score'] <= 9:
                    st.warning("⚠️ Mild depression - Consider talking to someone you trust")
                elif result['score'] <= 14:
                    st.error("🚨 Moderate depression - Professional help recommended")
                else:
                    st.error("🚨 Severe depression - Immediate professional evaluation needed")
            
            # Recommendations
            st.markdown("## 💡 Recommendations")
            for i, rec in enumerate(result['recommendations'], 1):
                st.markdown(f"{i}. {rec}")
            
            # Disclaimer
            st.markdown("---")
            st.markdown("""
            **⚠️ Important Disclaimer**: 
            This screening tool is for informational purposes only and is not a substitute for professional medical advice, 
            diagnosis, or treatment. If you're experiencing severe symptoms, please contact a mental health professional 
            or crisis hotline immediately.
            """)
            
        except Exception as e:
            st.error(f"Error calculating score: {e}")

def show_model_info(model_data):
    """Display information about the model and dataset"""
    
    st.markdown("## 🤖 About the Model")
    
    st.markdown("### 📊 Dataset Information")
    st.markdown("""
    This model was trained on a comprehensive sleep health and lifestyle dataset containing:
    - **374 individuals** with detailed lifestyle information
    - **Sleep patterns** (duration, quality, disorders)
    - **Physical health metrics** (activity level, heart rate, daily steps)
    - **Stress levels** and mental health indicators
    """)
    
    st.markdown("### 🧠 Model Details")
    st.markdown("""
    - **Algorithm**: Random Forest Classifier
    - **Features**: 9 engineered features from lifestyle data
    - **Target**: Mental health risk (binary classification)
    - **Performance**: Optimized for balanced accuracy
    """)
    
    st.markdown("### 🔍 Feature Engineering")
    st.markdown("The model uses both raw features and engineered features:")
    
    feature_info = {
        "Sleep Duration": "Hours of sleep per night",
        "Quality of Sleep": "Self-reported sleep quality (1-10)",
        "Physical Activity Level": "Activity intensity (Low/Moderate/High)",
        "Stress Level": "Perceived stress (0-10 scale)",
        "Heart Rate": "Resting heart rate (bpm)",
        "Daily Steps": "Average daily step count",
        "Sleep Efficiency": "Sleep duration × quality / 10",
        "Activity-Stress Ratio": "Physical activity / (stress + 1)",
        "Sleep Quality Ratio": "Sleep quality / sleep duration"
    }
    
    for feature, description in feature_info.items():
        st.markdown(f"- **{feature}**: {description}")
    
    st.markdown("### 📈 Model Performance")
    st.markdown("""
    The model has been validated using:
    - **Cross-validation** for robust performance estimation
    - **Balanced accuracy** to handle class imbalance
    - **Feature importance analysis** for interpretability
    """)
    
    # Display feature importance
    st.markdown("### 🎯 Feature Importance")
    feature_importance = model_data['feature_importance']
    
    fig = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        title="Feature Importance in Mental Health Risk Prediction",
        labels={'importance': 'Importance Score', 'feature': 'Features'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 