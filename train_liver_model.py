import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Define column names manually
column_names = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkphos_Alkaline',
                'Sgpt_Alamine', 'Sgot_Aspartate', 'Total_Proteins', 'Albumin',
                'Albumin_and_Globulin_Ratio', 'Result']

# Load the dataset with these column names (fix path and header)
df = pd.read_csv(r'dataset\liver.csv', names=column_names, header=None)

# Optional: View column names
print("Columns loaded:", df.columns.tolist())

# Encode gender
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Drop missing values
df = df.dropna()

# Define X and y correctly using 'Result' column
X = df.drop('Result', axis=1)
y = df['Result'].apply(lambda x: 1 if x == 1 else 0)  # 1 = disease, 0 = healthy

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/liver_model.pkl')

print("Model training and saving successful.")
