import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("cyberattack_model.pkl")

data = pd.read_csv("processed/cleaned_dataset.csv")

X = data.drop("Label", axis=1)
X = X.select_dtypes(include=["number"])

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

top_features = importance.head(15)

print("Top 15 Important Features:")
print(top_features)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features for Cyberattack Detection")

plt.tight_layout()
plt.savefig("feature_importance.png")

print("\nGraph saved successfully!")
print("File: feature_importance.png")