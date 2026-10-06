import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("processed/cleaned_dataset.csv")

print("Dataset loaded")
print("Shape:", data.shape)

X = data.drop("Label", axis=1)
y = data["Label"]

X = X.select_dtypes(include=["number"])

print("Features:", X.shape[1])
print("Samples:", X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())