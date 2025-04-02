from flask import Flask,render_template,request
import joblib
import numpy as np
import pandas as pd
app=Flask(__name__)

model=joblib.load("finalfinalmodel.joblib")

@app.route("/")
def home():
    return render_template("predict.html")

@app.route("/sub",methods=["POST"])
def result():
    result=[[0]]
    if request.method=='POST':
        cols=['Age', 'Gender', 'Heart rate', 'Systolic blood pressure',
       'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']
        age=request.form['Age']
        gender=float(request.form['Gender'])
        Heart_rate=float(request.form['Heart_Rate'])
        systolic=float(request.form['Systolic_blood_pressure'])
        diastolic=float(request.form['Diastolic_blood_pressure'])
        bloodsugar=float(request.form['Blood_sugar'])
        ckmb=float(request.form['CK-MB'])
        troponin=float(request.form['Troponin'])
        input=[age,gender,Heart_rate,systolic,diastolic,bloodsugar,ckmb,troponin]
        inputreshaped=np.array(input).reshape(1,-1)
        finalinput=pd.DataFrame(inputreshaped,columns=cols)
        result=model.predict(finalinput)
        print(result)
        print(finalinput)
    return render_template("result.html",res=result[0])

if __name__ == "__main__":
    app.run(debug=True)
        