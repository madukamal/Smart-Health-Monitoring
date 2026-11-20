import pandas as pd

# Load dataset
data = pd.read_csv("../dataset/water_health_data.csv")

# Convert Gender into numbers
data["Gender"] = data["Gender"].map({
    "Male": 0,
    "Female": 1
})

# Convert Risk into numbers
data["Risk"] = data["Risk"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

print("Prepared Dataset:")
print(data)

print("\nData Types:")
print(data.dtypes)