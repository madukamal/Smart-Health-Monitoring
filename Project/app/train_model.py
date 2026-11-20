import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Input features
X = data.drop("Risk", axis=1)

# Target
y = data["Risk"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Training Completed!")
print("-------------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Low", "Medium", "High"],
    zero_division=0
))

# Save model
joblib.dump(model, "../model/disease_risk_model.pkl")

print("\nModel saved successfully!")