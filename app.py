from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load all models
heart_model = joblib.load('model/heart_model.pkl')
diabetes_model = joblib.load('model/diabetes_model.pkl')
liver_model = joblib.load('model/liver_model.pkl')
kidney_model = joblib.load('model/kidney_model.pkl')
lung_model = joblib.load('model/lung_model.pkl')
parkinsons_model = joblib.load('model/parkinsons_model.pkl')

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# -------------------
# Heart Disease
# -------------------
@app.route('/heart', methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_input = [np.array(features)]
        prediction = heart_model.predict(final_input)
        output = "High Risk" if prediction[0] == 1 else "Low Risk"
        return render_template('heart.html', prediction_text=f'Prediction: {output}')
    return render_template('heart.html')

# -------------------
# Diabetes
# -------------------
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        prediction = diabetes_model.predict([features])
        output = "Positive" if prediction[0] == 1 else "Negative"
        return render_template('diabetes.html', prediction_text=f'Diabetes Prediction: {output}')
    return render_template('diabetes.html')

# -------------------
# Liver Disease
# -------------------
@app.route('/liver', methods=['GET', 'POST'])
def liver():
    if request.method == 'POST':
        data = [float(x) for x in request.form.values()]
        prediction = liver_model.predict([data])
        output = "Liver Disease Detected" if prediction[0] == 1 else "No Liver Disease"
        return render_template('liver.html', prediction_text=f'Liver Prediction: {output}')
    return render_template('liver.html')

# -------------------
# Kidney Disease
# -------------------
@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    if request.method == 'POST':
        values = [float(x) for x in request.form.values()]
        prediction = kidney_model.predict([values])
        output = "Chronic Kidney Disease Detected" if prediction[0] == 1 else "No Disease"
        return render_template('kidney.html', prediction_text=output)
    return render_template('kidney.html')

# -------------------
# Lung Cancer
# -------------------
@app.route('/lung', methods=['GET', 'POST'])
def lung():
    if request.method == 'POST':
        values = [float(x) for x in request.form.values()]
        prediction = lung_model.predict([values])[0]
        level = ["Low Risk", "Medium Risk", "High Risk"][prediction]
        return render_template('lung.html', prediction_text=f'Lung Cancer Risk: {level}')
    return render_template('lung.html')

# -------------------
# Parkinson’s Disease
# -------------------
@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    if request.method == 'POST':
        values = [float(x) for x in request.form.values()]
        prediction = parkinsons_model.predict([values])[0]
        output = "Parkinson’s Detected" if prediction == 1 else "Healthy"
        return render_template('parkinsons.html', prediction_text=f'Prediction: {output}')
    return render_template('parkinsons.html')

# -------------------
# Run Flask
# -------------------
if __name__ == '__main__':
    app.run(debug=True)
