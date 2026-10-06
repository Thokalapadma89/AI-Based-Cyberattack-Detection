from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("cyberattack_model.pkl")
data = pd.read_csv("processed/cleaned_dataset.csv")

X = data.drop("Label", axis=1)
X = X.select_dtypes(include=["number"])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    row_number = int(request.form["row_number"])

    sample = X.iloc[[row_number]]

    prediction = model.predict(sample)[0]

    if prediction == 0:
        result = "BENIGN"
    else:
        result = "ATTACK"

    return render_template(
        "index.html",
        prediction=result,
        row_number=row_number
    )

if __name__ == "__main__":
    app.run(debug=True)