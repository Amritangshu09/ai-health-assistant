import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
df = pd.read_csv('dataset/parkinsons.csv')
print("\n✅ Parkinson's dataset loaded.")
print("📋 Columns in dataset:")
for col in df.columns:
    print(f"- {col}")

# Try to find a valid target column
possible_targets = ['class', 'status', 'Status', 'result', 'Result']
target_column = None
for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    raise ValueError(f"❌ Target column not found. Tried: {', '.join(possible_targets)}")

print(f"\n🎯 Target column detected: '{target_column}'")

# Drop non-feature columns if they exist
non_features = ['name', 'id']  # common non-feature columns
df = df.drop(columns=[col for col in non_features if col in df.columns], errors='ignore')

# Ensure all data is numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Separate features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

print(f"\n📊 Cleaned dataset: {len(df)} samples, {X.shape[1]} features")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained on Parkinson's dataset.")

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/parkinsons_model.pkl')
print("💾 Model saved as 'model/parkinsons_model.pkl'")
