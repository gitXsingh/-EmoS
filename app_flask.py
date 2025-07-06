from flask import Flask, render_template_string, request, redirect, url_for
import numpy as np
import pickle
from phq9 import PHQ9_QUESTIONS, PHQ9_OPTIONS, calculate_phq9_score

app = Flask(__name__)

# Claude-inspired color palette and chat bubble style
BASE_STYLE = '''
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  html, body {
    background: #f9f7f3;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    color: #232324;
    min-height: 100vh;
    margin: 0;
    font-size: 1.18em;
    line-height: 1.8;
    letter-spacing: -0.01em;
    transition: background 0.5s, color 0.5s;
    padding: 0;
  }
  body {
    padding: 0;
    margin: 0;
  }
  .emos-topbar {
    width: 100vw;
    left: 0;
    top: 0;
    position: fixed;
    z-index: 100;
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(12px);
    box-shadow: 0 2px 16px 0 rgba(0,0,0,0.04);
    border-bottom: 1.5px solid #ffe3b3;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 68px;
    font-size: 1.45em;
    font-weight: 700;
    color: #232324;
    letter-spacing: -0.5px;
    margin: 0;
    padding: 0;
    position: fixed;
  }
  .emos-topbar-inner {
    width: 100%;
    max-width: 1100px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    height: 68px;
    padding: 0 32px;
  }
  .emos-logo {
    color: #ff9900;
    font-weight: 800;
    font-size: 1.05em;
    letter-spacing: 0.5px;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    text-shadow: 0 2px 8px #ffb30022;
    margin: 0;
    position: static;
    text-align: left;
    pointer-events: auto;
  }
  .dark-toggle {
    position: static;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    z-index: 2;
  }
  .toggle-btn {
    background: #fff8e1;
    color: #ff9900;
    border: 1.5px solid #ff9900;
    border-radius: 22px;
    padding: 0 4px;
    cursor: pointer;
    font-size: 0.85em;
    margin: 0;
    transition: background 0.3s, color 0.3s, border 0.3s;
    font-weight: 700;
    box-shadow: 0 1px 4px #ffb30011;
    outline: none;
    letter-spacing: 0.1px;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 20px;
    min-height: 20px;
  }
  .toggle-btn:hover {
    background: #ff9900;
    color: #fff;
    border: 1.5px solid #ff9900;
  }
  /* Dark mode styles */
  body.dark-mode {
    background: #232324;
    color: #f9f6f2;
  }
  body.dark-mode .emos-topbar {
    background: rgba(35,35,36,0.85);
    color: #ffd580;
    border-bottom: 1.5px solid #35363a;
  }
  body.dark-mode .emos-logo {
    color: #ffd580;
    text-shadow: 0 2px 8px #ffb30022;
  }
  body.dark-mode .toggle-btn {
    background: #35363a;
    color: #ffd580;
    border: 1.5px solid #ffd580;
  }
  body.dark-mode .toggle-btn:hover {
    background: #ffd580;
    color: #232324;
    border: 1.5px solid #ffd580;
  }
  body.dark-mode .card {
    background: #232324;
    color: #f9f6f2;
    border: 1.5px solid #35363a;
    box-shadow: 0 4px 32px 0 #23232433;
  }
  body.dark-mode label {
    color: #f9f6f2;
  }
  body.dark-mode input, body.dark-mode select {
    background: #18181a;
    color: #f9f6f2;
    border: 1.5px solid #35363a;
  }
  body.dark-mode input:focus, body.dark-mode select:focus {
    border: 1.5px solid #ffd580;
  }
  body.dark-mode input[type=submit] {
    background: #ffd580;
    color: #232324;
  }
  body.dark-mode input[type=submit]:hover {
    background: #ffb300;
    color: #fff;
  }
  body.dark-mode h2, body.dark-mode .card h2 {
    color: #f9f6f2 !important;
  }
  .container {
    max-width: 540px;
    margin: 0 auto;
    padding: 0 12px;
    margin-top: 68px;
  }
  .card, .bubble {
    background: #fff;
    border-radius: 22px;
    border: 1.5px solid #f3e7d7;
    box-shadow: 0 2px 12px 0 rgba(0,0,0,0.04);
    padding: 36px 28px;
    margin-bottom: 32px;
  }
  .bubble {
    margin-top: 24px;
    margin-bottom: 24px;
    background: #fff8e1;
    border: 1.5px solid #ffb300;
    color: #232324;
    font-size: 1.12em;
    font-weight: 600;
    position: relative;
  }
  .bubble:before {
    content: '';
    position: absolute;
    left: 32px;
    top: -16px;
    width: 24px;
    height: 24px;
    background: #fff8e1;
    border-left: 1.5px solid #ffb300;
    border-top: 1.5px solid #ffb300;
    border-radius: 8px 0 0 0;
    transform: rotate(-45deg);
    z-index: 0;
  }
  h2, h3 {
    color: #232324;
    text-align: left;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 18px;
    margin-top: 0;
    font-size: 1.45em;
    line-height: 1.25;
  }
  label {
    display: block;
    margin: 18px 0 8px;
    font-weight: 600;
    letter-spacing: -0.2px;
  }
  input, select {
    width: 100%;
    padding: 15px;
    border-radius: 14px;
    border: 1.5px solid #f3e7d7;
    margin-bottom: 20px;
    background: #f9f7f3;
    font-size: 1em;
    font-family: inherit;
    color: #232324;
    box-sizing: border-box;
    outline: none;
    transition: border 0.2s;
    font-weight: 500;
  }
  input:focus, select:focus {
    border: 1.5px solid #ff9900;
  }
  input[type=submit] {
    background: #ff9900;
    color: #fff;
    border: none;
    font-weight: 700;
    cursor: pointer;
    transition: background 0.3s;
    box-shadow: 0 2px 8px #ffb30022;
    letter-spacing: 0.2px;
    font-size: 1.13em;
    border-radius: 14px;
    margin-top: 8px;
  }
  input[type=submit]:hover {
    background: #ffb300;
  }
  .risk-high { color: #d7263d; font-weight: bold; }
  .risk-low { color: #2ecc71; font-weight: bold; }
  .wellness-score { font-size: 1.2em; color: #ff9900; font-weight: 700; }
  a { color: #ff9900; text-decoration: none; font-weight: 600; }
  a:hover { text-decoration: underline; color: #ffb300; }
</style>
<script>
function toggleDarkMode() {
  document.body.classList.toggle('dark-mode');
  let btn = document.getElementById('darkBtn');
  if(document.body.classList.contains('dark-mode')) {
    btn.innerHTML = '☀️';
  } else {
    btn.innerHTML = '🌙';
  }
}
window.onload = function() {
  let btn = document.getElementById('darkBtn');
  if(document.body.classList.contains('dark-mode')) {
    btn.innerHTML = '☀️';
  } else {
    btn.innerHTML = '🌙';
  }
}
</script>
'''

HOME_HTML = '''
<!doctype html>
<html><head><title>Mental Health Risk Prediction</title>{BASE_STYLE}</head><body>
<div class="emos-topbar">
  <div class="emos-topbar-inner">
    <span class="emos-logo">EmoS</span>
    <div class="dark-toggle"><button id="darkBtn" class="toggle-btn" onclick="toggleDarkMode()">🌙</button></div>
  </div>
</div>
<div class="container">
  <div class="card">
    <h2>Mental Health Risk Prediction</h2>
    <form method=post action="/result">
      <label>Sleep Duration (hours): <input type=number step=0.1 name=sleep_duration value=7></label>
      <label>Quality of Sleep (1-10): <input type=number name=quality_of_sleep min=1 max=10 value=7></label>
      <label>Physical Activity Level:
        <select name=physical_activity>
          <option value="Low">Low</option>
          <option value="Moderate">Moderate</option>
          <option value="High">High</option>
        </select>
      </label>
      <label>Stress Level (0-10): <input type=number name=stress_level min=0 max=10 value=5></label>
      <label>Resting Heart Rate (bpm): <input type=number name=heart_rate min=50 max=120 value=75></label>
      <label>Daily Steps: <input type=number name=daily_steps min=1000 max=15000 value=6000></label>
      <label>Mood Swings: <input type=checkbox name=mood_swings></label>
      <label>Screen Time (hours/day): <input type=number name=screen_time min=1 max=16 value=6></label>
      <label>Daily Social Interactions <input type=number name=social_interactions min=0 max=20 value=5></label>
      <input type=submit value="Predict">
    </form>
    <br>
    <a href="/phq9">Take PHQ-9 Depression Quiz</a>
  </div>
</div></body></html>
'''.replace('{BASE_STYLE}', BASE_STYLE)

RESULT_HTML = '''
<!doctype html>
<html><head><title>Prediction Result</title>{BASE_STYLE}</head><body>
<div class="emos-topbar">
  <div class="emos-topbar-inner">
    <span class="emos-logo">EmoS</span>
    <div class="dark-toggle"><button id="darkBtn" class="toggle-btn" onclick="toggleDarkMode()">🌙</button></div>
  </div>
</div>
<div class="container">
  <div class="bubble">
    <h3>Prediction Result</h3>
    <p><b>Risk Level:</b> <span class="{{ 'risk-high' if risk == 'HIGH RISK' else 'risk-low' }}">{{ risk }}</span></p>
    <p class="wellness-score"><b>Wellness Score:</b> {{ wellness_score }}/100</p>
    <p><b>Personalized Recommendations:</b></p>
    <ul style="margin-top: 0;">
    {% for rec in recommendations %}<li>{{ rec }}</li>{% endfor %}
    </ul>
    <a href="/">Back to Home</a>
  </div>
</div></body></html>
'''.replace('{BASE_STYLE}', BASE_STYLE)

PHQ9_HTML = '''
<!doctype html>
<html><head><title>PHQ-9 Depression Quiz</title>{BASE_STYLE}</head><body>
<div class="emos-topbar">
  <div class="emos-topbar-inner">
    <span class="emos-logo">EmoS</span>
    <div class="dark-toggle"><button id="darkBtn" class="toggle-btn" onclick="toggleDarkMode()">🌙</button></div>
  </div>
</div>
<div class="container">
  <div class="card">
    <h2>PHQ-9 Depression Screening</h2>
    <form method=post action="/phq9">
    {% for q in questions %}
      <label>{{ loop.index }}. {{ q }}<br>
        <select name="q{{ loop.index0 }}">
          {% for opt in options %}
            <option value="{{ loop.index0 }}">{{ opt }}</option>
          {% endfor %}
        </select>
      </label><br><br>
    {% endfor %}
      <input type=submit value="Get PHQ-9 Score">
    </form>
    {% if result %}
      <div class="bubble">
        <h3>PHQ-9 Score: {{ result['score'] }}/27</h3>
        <p>Severity: {{ result['severity'] }}</p>
        <ul>
        {% for rec in result['recommendations'] %}<li>{{ rec }}</li>{% endfor %}
        </ul>
      </div>
    {% endif %}
    <a href="/">Back to Home</a>
  </div>
</div></body></html>
'''.replace('{BASE_STYLE}', BASE_STYLE)

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
    return render_template_string(HOME_HTML)

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
    return render_template_string(RESULT_HTML, risk=risk, wellness_score=wellness_score, recommendations=recommendations)

@app.route('/phq9', methods=['GET', 'POST'])
def phq9():
    result = None
    if request.method == 'POST':
        responses = [int(request.form.get(f'q{i}', 0)) for i in range(9)]
        result = calculate_phq9_score(responses)
    return render_template_string(PHQ9_HTML, questions=PHQ9_QUESTIONS, options=PHQ9_OPTIONS, result=result)

if __name__ == '__main__':
    app.run(debug=True) 