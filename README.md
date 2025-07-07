# EmoS Project

Welcome to the EmoS repository!

## Description
EmoS is a modern, full-stack machine learning web app for mental health risk prediction and PHQ-9 depression screening. It features a beautiful Claude-inspired UI with glassmorphism, dark mode, and a responsive design. Built with Python and Flask, it uses a trained Random Forest model to provide personalized wellness recommendations.

## Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd EmoS
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure `mental_health_model.pkl` is present in the project root (already included if you cloned the repo).

### Running the App
```bash
python app_flask.py
```
Then open your browser and go to [http://localhost:5000](http://localhost:5000)

## Features
- 🧠 **Mental Health Risk Prediction**: Enter lifestyle and health data to get a risk assessment and wellness score.
- 📋 **PHQ-9 Depression Screening**: Take the PHQ-9 quiz and receive severity and recommendations.
- 💬 **Modern UI**: Claude-inspired, glassy, and responsive interface.
- 🌙 **Dark Mode**: Toggle dark/light mode (persists across pages).
- ⚡ **Personalized Recommendations**: Actionable tips based on your data and model prediction.
- 🔗 **Instant Navigation**: Click the "EmoS" logo to return home from any page.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. 