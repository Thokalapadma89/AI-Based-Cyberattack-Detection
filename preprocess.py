import pandas as pd
import numpy as np
import glob
import os

files = glob.glob("dataset/*.csv")

print("CSV files found:", len(files))

dataframes = []

for file in files:
    print("Loading:", os.path.basename(file))
    df = pd.read_csv(file, low_memory=False)
    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

print("\nOriginal shape:", data.shape)

data.columns = data.columns.str.strip()

data["Label"] = data["Label"].str.strip()

data = data.replace([np.inf, -np.inf], np.nan)

data = data.dropna()

print("Shape after cleaning:", data.shape)

data["Label"] = data["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)

print("\nLabel distribution:")
print(data["Label"].value_counts())

os.makedirs("processed", exist_ok=True)

data.to_csv("processed/cleaned_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("File: processed/cleaned_dataset.csv")