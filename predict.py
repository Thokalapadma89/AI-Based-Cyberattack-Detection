import pandas as pd
import joblib

print("=" * 55)
print("      AI-BASED CYBERATTACK DETECTION SYSTEM")
print("=" * 55)

model = joblib.load("cyberattack_model.pkl")

data = pd.read_csv("processed/cleaned_dataset.csv")

X = data.drop("Label", axis=1)
X = X.select_dtypes(include=["number"])

row_number = int(input("\nEnter network traffic row number: "))

if row_number < 0 or row_number >= len(X):
    print("\nInvalid row number!")
else:
    sample = X.iloc[[row_number]]

    prediction = model.predict(sample)[0]

    print("\n" + "-" * 55)
    print("Network Traffic Row:", row_number)

    if prediction == 0:
        print("Result: NORMAL TRAFFIC")
        print("Status: Benign Traffic Detected")
    else:
        print("Result: ATTACK")
        print("Status: Malicious Traffic Detected")

    print("-" * 55)