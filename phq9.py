"""
PHQ-9 Depression Screening Module
Patient Health Questionnaire-9 for depression assessment
"""

PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling or staying asleep, or sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself - or that you are a failure or have let yourself or your family down?",
    "Trouble concentrating on things, such as reading the newspaper or watching television?",
    "Moving or speaking slowly enough that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?",
    "Thoughts that you would be better off dead or of hurting yourself in some way?"
]

PHQ9_OPTIONS = [
    "Not at all (0 points)",
    "Several days (1 point)", 
    "More than half the days (2 points)",
    "Nearly every day (3 points)"
]

PHQ9_SEVERITY_LEVELS = {
    (0, 4): "Minimal depression",
    (5, 9): "Mild depression", 
    (10, 14): "Moderate depression",
    (15, 19): "Moderately severe depression",
    (20, 27): "Severe depression"
}

def get_phq9_severity(score):
    """Get severity level based on PHQ-9 score"""
    for (min_score, max_score), severity in PHQ9_SEVERITY_LEVELS.items():
        if min_score <= score <= max_score:
            return severity
    return "Invalid score"

def get_phq9_recommendations(score):
    """Get recommendations based on PHQ-9 score"""
    if score <= 4:
        return [
            "Continue maintaining good mental health practices",
            "Regular exercise and social activities",
            "Maintain healthy sleep patterns"
        ]
    elif score <= 9:
        return [
            "Consider talking to a trusted friend or family member",
            "Practice stress-reduction techniques (meditation, deep breathing)",
            "Maintain regular sleep schedule",
            "Consider professional counseling if symptoms persist"
        ]
    elif score <= 14:
        return [
            "Strongly consider professional mental health evaluation",
            "Contact a mental health professional",
            "Practice self-care and stress management",
            "Consider medication evaluation with a psychiatrist"
        ]
    else:
        return [
            "Immediate professional mental health evaluation recommended",
            "Contact a mental health professional or crisis hotline",
            "Consider emergency mental health services if needed",
            "Do not hesitate to seek help - you deserve support"
        ]

def calculate_phq9_score(responses):
    """Calculate total PHQ-9 score from responses"""
    if len(responses) != 9:
        raise ValueError("PHQ-9 requires exactly 9 responses")
    
    total_score = sum(responses)
    severity = get_phq9_severity(total_score)
    recommendations = get_phq9_recommendations(total_score)
    
    return {
        'score': total_score,
        'severity': severity,
        'recommendations': recommendations,
        'max_score': 27
    } 