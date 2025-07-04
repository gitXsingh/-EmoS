import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

print("📊 Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\n🛏️ Sleep Disorder Distribution:")
print(df['Sleep Disorder'].value_counts())

print("\n😰 Stress Level Distribution:")
print(df['Stress Level'].value_counts())

print("\n🎯 Creating Mental Health Risk Target...")
# Create mental health risk target
df['mental_health_risk'] = (
    (df['Stress Level'] >= 7) | 
    (df['Quality of Sleep'] <= 5)
).astype(int)

print("Mental Health Risk Distribution:")
print(df['mental_health_risk'].value_counts())

print(f"\n📈 Risk Percentage: {df['mental_health_risk'].mean():.1%}")

# Check if we need to adjust the threshold
print("\n🔍 Adjusting threshold for better balance...")
for threshold in [5, 6, 7, 8]:
    risk = ((df['Stress Level'] >= threshold) | (df['Quality of Sleep'] <= 5)).astype(int)
    print(f"Stress threshold {threshold}: {risk.mean():.1%} high risk") 