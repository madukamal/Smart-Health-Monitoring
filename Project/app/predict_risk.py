
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("../model/disease_risk_model.pkl")

print("===================================")
print(" SMART HEALTH MONITORING SYSTEM")
print("===================================")

# Get information from the user
age = int(input("Enter Age: "))
gender = input("Enter Gender (Male/Female): ")

ph = float(input("Enter Water pH: "))
turbidity = float(input("Enter Turbidity: "))
temperature = float(input("Enter Water Temperature: "))

fever = int(input("Fever? (1=Yes, 0=No): "))
diarrhea = int(input("Diarrhea? (1=Yes, 0=No): "))
vomiting = int(input("Vomiting? (1=Yes, 0=No): "))
abdominal_pain = int(input("Abdominal Pain? (1=Yes, 0=No): "))

# Convert Gender into a number
gender_value = 0 if gender.lower() == "male" else 1

# Create input data
new_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender_value,
    "pH": ph,
    "Turbidity": turbidity,
    "Temperature": temperature,
    "Fever": fever,
    "Diarrhea": diarrhea,
    "Vomiting": vomiting,
    "Abdominal_Pain": abdominal_pain
}])

# Make prediction
prediction = model.predict(new_data)[0]

# Convert prediction number into a risk label
risk_labels = {
    0: "Low",
    1: "Medium",
    2: "High"
}

risk = risk_labels[prediction]

print("\n===================================")
print(" PREDICTION RESULT")
print("===================================")
print("Predicted Disease Risk:", risk)

if risk == "High":
    print("WARNING: High predicted risk.")
    print("Further investigation is recommended.")
elif risk == "Medium":
    print("CAUTION: Moderate predicted risk.")
else:
    print("STATUS: Low predicted risk.")

print("===================================")