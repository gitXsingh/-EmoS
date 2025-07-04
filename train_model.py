"""
Mental Health Risk Prediction Model Training Script
Trains a Random Forest classifier on sleep and lifestyle data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the sleep health dataset"""
    print("🔄 Loading dataset...")
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    
    # Create mental health risk target based on sleep disorders and stress
    # High risk: Sleep Apnea, Insomnia, or Stress Level >= 7
    # Since everyone has sleep disorders, let's use stress level and sleep quality
    df['mental_health_risk'] = (
        (df['Stress Level'] >= 7) | 
        (df['Quality of Sleep'] <= 5)
    ).astype(int)
    
    # Select relevant features for mental health prediction
    feature_columns = [
        'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
        'Stress Level', 'Heart Rate', 'Daily Steps'
    ]
    
    # Create additional features
    df['sleep_efficiency'] = df['Sleep Duration'] * df['Quality of Sleep'] / 10
    df['activity_stress_ratio'] = df['Physical Activity Level'] / (df['Stress Level'] + 1)
    df['sleep_quality_ratio'] = df['Quality of Sleep'] / df['Sleep Duration']
    
    feature_columns.extend(['sleep_efficiency', 'activity_stress_ratio', 'sleep_quality_ratio'])
    
    X = df[feature_columns]
    y = df['mental_health_risk']
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Target distribution:\n{y.value_counts()}")
    print(f"📈 Features: {feature_columns}")
    
    return X, y, feature_columns

def train_model(X, y):
    """Train the Random Forest model"""
    print("\n🤖 Training Random Forest model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
    
    print(f"✅ Model Accuracy: {accuracy:.3f}")
    print(f"✅ Cross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Check if we have both classes in the test set
    unique_classes = np.unique(y_test)
    if len(unique_classes) > 1:
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    else:
        print(f"\n📋 Note: Test set contains only class {unique_classes[0]} (no classification report available)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 5 Most Important Features:")
    for idx, row in feature_importance.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    return rf_model, scaler, feature_importance, (X_test_scaled, y_test, y_pred)

def save_model(model, scaler, feature_importance, filename='mental_health_model.pkl'):
    """Save the trained model and scaler"""
    print(f"\n💾 Saving model to {filename}...")
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_importance': feature_importance,
        'feature_names': feature_importance['feature'].tolist()
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✅ Model saved successfully!")

def main():
    """Main training pipeline"""
    print("🧠 Mental Health Risk Prediction Model Training")
    print("=" * 50)
    
    # Load and preprocess data
    X, y, feature_columns = load_and_preprocess_data()
    
    # Train model
    model, scaler, feature_importance, test_results = train_model(X, y)
    
    # Save model
    save_model(model, scaler, feature_importance)
    
    print("\n🎉 Training completed successfully!")
    print("📁 Files created:")
    print("   - mental_health_model.pkl (trained model)")
    print("   - Ready to use with Streamlit app!")

if __name__ == "__main__":
    main() 