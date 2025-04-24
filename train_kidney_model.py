import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and preprocess dataset
df = pd.read_csv('dataset/kidney.csv')
df = df.replace("?", pd.NA).dropna()
df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})

# Convert columns to numeric safely
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except ValueError:
        pass  # Keep original if conversion fails

X = df.drop('classification', axis=1).select_dtypes(include='number')
y = df['classification']

# Train and save model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, 'model/kidney_model.pkl')
