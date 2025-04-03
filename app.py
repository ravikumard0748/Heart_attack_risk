from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load ML Model
model = joblib.load("finalfinalmodel.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/sub", methods=["POST"])
def result():
    if request.method == "POST":
        cols = ['Age', 'Gender', 'Heart rate', 'Systolic blood pressure',
                'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']
        
        try:
            # Extract form values
            age = float(request.form['Age'])
            gender = float(request.form['Gender'])
            heart_rate = float(request.form['Heart_Rate'])
            systolic = float(request.form['Systolic_blood_pressure'])
            diastolic = float(request.form['Diastolic_blood_pressure'])
            blood_sugar = float(request.form['Blood_sugar'])
            ckmb = float(request.form['CK-MB'])
            troponin = float(request.form['Troponin'])

            # Prepare input data
            input_data = np.array([age, gender, heart_rate, systolic, diastolic, blood_sugar, ckmb, troponin]).reshape(1, -1)
            final_input = pd.DataFrame(input_data, columns=cols)

            # Make prediction
            result = model.predict(final_input)

        except Exception as e:
            return render_template("result.html", res=f"Error: {str(e)}")

    return render_template("result.html", res=int(result[0]))  # Show 1 or 0

if __name__ == "__main__":
    app.run(debug=True)
