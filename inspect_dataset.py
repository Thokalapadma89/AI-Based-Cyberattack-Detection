import pandas as pd

file_path = "dataset/Monday-WorkingHours.pcap_ISCX.csv"

data = pd.read_csv(file_path, low_memory=False)

print("Dataset loaded successfully")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

print("\nColumn Names:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

print("\nAttack Labels:")
print(data[" Label"].value_counts())