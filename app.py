from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("cyberattack_model.pkl")

# Load small demo dataset for online deployment
data = pd.read_csv("demo_dataset.csv")

# Keep original row numbers
original_rows = data["Original_Row"]

# Prepare features
X = data.drop(columns=["Label", "Original_Row"])
X = X.select_dtypes(include=["number"])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    row_number = int(request.form["row_number"])

    # Find the requested original dataset row
    matching = data.index[data["Original_Row"] == row_number].tolist()

    if not matching:
        return render_template(
            "index.html",
            prediction="Row not available in online demo",
            row_number=row_number
        )

    demo_index = matching[0]

    sample = X.iloc[[demo_index]]

    prediction = model.predict(sample)[0]

    if prediction == 0:
        result = "NORMAL TRAFFIC"
    else:
        result = "ATTACK"

    return render_template(
        "index.html",
        prediction=result,
        row_number=row_number
    )


if __name__ == "__main__":
    app.run(debug=True)