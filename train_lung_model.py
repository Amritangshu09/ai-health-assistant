import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
df = pd.read_csv('dataset/lung.csv')
print("✅ Dataset loaded. Columns:", df.columns.tolist())

# Drop non-numeric/non-feature columns
df = df.drop(['Name', 'Surname'], axis=1)

# Ensure all columns are numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with any missing or invalid values
df = df.dropna()

# Check if dataset has any samples left
if df.empty:
    raise ValueError("❌ Cleaned dataset is empty. Please check for invalid or missing values.")

# Separate features and target
X = df.drop('Result', axis=1)
y = df['Result']

# Debug: Print basic stats
print(f"📊 Total samples: {len(df)} | Features: {X.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# Save the model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/lung_model.pkl')
print("💾 Model saved to 'model/lung_model.pkl'")
