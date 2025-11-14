from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model, scaler = pickle.load(open("model/model.pkl", "rb"))

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    data = scaler.transform([data])
    prediction = model.predict(data)[0]

    result = "High Risk" if prediction == 1 else "Low Risk"
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
