import pandas as pd

data = pd.read_csv("../dataset/water_health_data.csv")

print("FIRST 5 ROWS:")
print(data.head())

print("\nDATASET INFORMATION:")
print(data.info())

print("\nMISSING VALUES:")
print(data.isnull().sum())

print("\nRISK COUNTS:")
print(data["Risk"].value_counts())