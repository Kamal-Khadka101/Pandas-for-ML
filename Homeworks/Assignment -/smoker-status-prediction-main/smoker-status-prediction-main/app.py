from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model (Ensure the model file path is correct)
with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from form and convert to a list of floats
    features = [
        float(request.form['age']),
        float(request.form['height']),
        float(request.form['weight']),
        float(request.form['waist']),
        float(request.form['eyesight_left']),
        float(request.form['eyesight_right']),
        float(request.form['hearing_left']),
        float(request.form['hearing_right']),
        float(request.form['systolic_bp']),
        float(request.form['diastolic_bp']),
        float(request.form['fasting_blood_sugar']),
        float(request.form['cholesterol']),
        float(request.form['triglyceride']),
        float(request.form['hdl']),
        float(request.form['ldl']),
        float(request.form['hemoglobin']),
        float(request.form['serum_creatinine']),
        float(request.form['ast']),
        float(request.form['alt']),
        float(request.form['ggt']),
        float(request.form['dental_cavities']),
        float(request.form['waist_to_height_ratio'])
    ]

    # Convert input data to a numpy array
    input_data = np.array([features])

    # Make a prediction using the model
    prediction = model.predict(input_data)[0]

    # Convert numerical output to human-readable labels
    prediction_label = "Non Smoker" if prediction == 0 else "Smoker"

    # Return the prediction result to the user
    return render_template('index.html', prediction_text=f'Predicted Class: {prediction_label}')

if __name__ == '__main__':
    app.run(debug=True)
