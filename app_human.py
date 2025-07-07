from flask import Flask, render_template, request
import numpy as np
import pickle
from phq9 import PHQ9_QUESTIONS, PHQ9_OPTIONS, calculate_phq9_score

app = Flask(__name__)

# Load model and scaler at startup
with open('mental_health_model.pkl', 'rb') as f:
    loaded = pickle.load(f)
    model_data = {
        'model': loaded['model'],
        'scaler': loaded['scaler']
    }

def calculate_wellness_score(user_data):
    score = 0
    if user_data['sleep_duration'] >= 7:
        score += 25
    elif user_data['sleep_duration'] >= 6:
        score += 15
    elif user_data['sleep_duration'] >= 5:
        score += 5
    if user_data['physical_activity'] == 'High':
        score += 20
    elif user_data['physical_activity'] == 'Moderate':
        score += 10
    if user_data['stress_level'] <= 3:
        score += 25
    elif user_data['stress_level'] <= 5:
        score += 15
    elif user_data['stress_level'] <= 7:
        score += 5
    if not user_data['mood_swings']:
        score += 15
    if user_data['screen_time'] <= 6:
        score += 15
    elif user_data['screen_time'] <= 8:
        score += 8
    return min(score, 100)

def get_personalized_recommendations(user_data, risk_prediction):
    recommendations = []
    if user_data['sleep_duration'] < 7:
        recommendations.append("Sleep: Aim for 7-9 hours of sleep per night.")
    if user_data['stress_level'] > 5:
        recommendations.append("Stress: Practice meditation, deep breathing, or yoga.")
    if user_data['physical_activity'] != 'High':
        recommendations.append("Exercise: Increase physical activity.")
    if user_data['mood_swings']:
        recommendations.append("Mood: Keep a mood journal.")
    if user_data['screen_time'] > 6:
        recommendations.append("Screen Time: Reduce screen time and take breaks.")
    if user_data['social_interactions'] < 5:
        recommendations.append("Social: Increase social interactions.")
    if risk_prediction == 1:
        recommendations.append("Professional Help: Consider consulting a mental health professional.")
    return recommendations

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def result():
    # Get form data
    sleep_duration = float(request.form['sleep_duration'])
    quality_of_sleep = int(request.form['quality_of_sleep'])
    physical_activity = request.form['physical_activity']
    physical_activity_level = {'Low': 30, 'Moderate': 50, 'High': 75}[physical_activity]
    stress_level = int(request.form['stress_level'])
    heart_rate = int(request.form['heart_rate'])
    daily_steps = int(request.form['daily_steps'])
    mood_swings = 'mood_swings' in request.form
    screen_time = int(request.form['screen_time'])
    social_interactions = int(request.form['social_interactions'])

    user_data = {
        'sleep_duration': sleep_duration,
        'quality_of_sleep': quality_of_sleep,
        'physical_activity_level': physical_activity_level,
        'stress_level': stress_level,
        'heart_rate': heart_rate,
        'daily_steps': daily_steps,
        'physical_activity': physical_activity,
        'mood_swings': mood_swings,
        'screen_time': screen_time,
        'social_interactions': social_interactions
    }
    
    sleep_efficiency = sleep_duration * quality_of_sleep / 10
    activity_stress_ratio = physical_activity_level / (stress_level + 1)
    sleep_quality_ratio = quality_of_sleep / sleep_duration
    
    features = np.array([
        sleep_duration, quality_of_sleep, physical_activity_level,
        stress_level, heart_rate, daily_steps,
        sleep_efficiency, activity_stress_ratio, sleep_quality_ratio
    ]).reshape(1, -1)
    
    features_scaled = model_data['scaler'].transform(features)
    prediction = model_data['model'].predict(features_scaled)[0]
    wellness_score = calculate_wellness_score(user_data)
    recommendations = get_personalized_recommendations(user_data, prediction)
    risk = 'HIGH RISK' if prediction == 1 else 'LOW RISK'
    
    return render_template('result.html', risk=risk, wellness_score=wellness_score, recommendations=recommendations)

@app.route('/phq9', methods=['GET', 'POST'])
def phq9():
    result = None
    if request.method == 'POST':
        responses = [int(request.form.get(f'q{i}', 0)) for i in range(9)]
        result = calculate_phq9_score(responses)
    return render_template('phq9.html', questions=PHQ9_QUESTIONS, options=PHQ9_OPTIONS, result=result)

if __name__ == '__main__':
    app.run(debug=True) 