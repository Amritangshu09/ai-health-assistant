from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model/heart_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_input = [np.array(features)]
    prediction = model.predict(final_input)
    output = "High Risk" if prediction[0] == 1 else "Low Risk"
    return render_template('index.html', prediction_text=f'Prediction: {output}')

#diabetes
diabetes_model = joblib.load('model/diabetes_model.pkl')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    features = [float(x) for x in request.form.values()]
    prediction = diabetes_model.predict([features])
    output = "Positive" if prediction[0] == 1 else "Negative"
    return render_template('diabetes.html', prediction_text=f'Diabetes Prediction: {output}')

#liver
liver_model = joblib.load('model/liver_model.pkl')

@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/predict_liver', methods=['POST'])
def predict_liver():
    data = [float(x) for x in request.form.values()]
    prediction = liver_model.predict([data])
    output = "Liver Disease Detected" if prediction[0] == 1 else "No Liver Disease"
    return render_template('liver.html', prediction_text=f'Liver Prediction: {output}')

#kidney
kidney_model = joblib.load('model/kidney_model.pkl')

@app.route('/kidney')
def kidney():
    return render_template('kidney.html')

@app.route('/predict_kidney', methods=['POST'])
def predict_kidney():
    values = [float(x) for x in request.form.values()]
    prediction = kidney_model.predict([values])
    output = "Chronic Kidney Disease Detected" if prediction[0] == 1 else "No Disease"
    return render_template('kidney.html', prediction_text=output)

#lung
lung_model = joblib.load('model/lung_model.pkl')

@app.route('/lung')
def lung():
    return render_template('lung.html')

@app.route('/predict_lung', methods=['POST'])
def predict_lung():
    values = [float(x) for x in request.form.values()]
    prediction = lung_model.predict([values])[0]
    level = ["Low Risk", "Medium Risk", "High Risk"][prediction]
    return render_template('lung.html', prediction_text=f'Lung Cancer Risk: {level}')

#parkinsons
parkinsons_model = joblib.load('model/parkinsons_model.pkl')

@app.route('/parkinsons')
def parkinsons():
    return render_template('parkinsons.html')

@app.route('/predict_parkinsons', methods=['POST'])
def predict_parkinsons():
    values = [float(x) for x in request.form.values()]
    prediction = parkinsons_model.predict([values])[0]
    output = "Parkinson’s Detected" if prediction == 1 else "Healthy"
    return render_template('parkinsons.html', prediction_text=f'Prediction: {output}')


if __name__ == '__main__':
    app.run(debug=True)
