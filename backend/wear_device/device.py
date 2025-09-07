# ==============================
# Wearable Dataset Classification + Remedy Recommendation
# ==============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import lightgbm as lgb

# ------------------------------
# 1. Load Dataset
# ------------------------------
df = pd.read_excel(r"C:\Users\IYYAPPAN\Desktop\Health_app\backend\wear_device\wearable_Dataset.xlsx")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Features and Labels
X = df.drop(columns=["label"])
y = df["label"]

# ------------------------------
# 2. Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 3. Train Classifier
# ------------------------------
try:
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, n_jobs=-1, random_state=42)
    model.fit(X_train_scaled, y_train)
except Exception:
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train_scaled, y_train)

# ------------------------------
# 4. Evaluate
# ------------------------------
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# 5. Remedy Recommendation System
# ------------------------------

def generate_remedies(row: pd.Series, disease: str, severity: str) -> list:
    remedies = []

    # ----- Diabetes Related -----
    if disease in ["Diabetes", "Both"]:
        if row.get("glucose_level", np.nan) < 70:
            remedies += [
                "Low glucose detected – consume fast-acting carbs (juice, glucose tablets).",
                "Avoid driving or hazardous activity until glucose stabilizes."
            ]
        elif row.get("glucose_level", np.nan) > 250:
            remedies += [
                "High glucose detected – hydrate with water.",
                "Avoid high-carb food and monitor glucose closely."
            ]

    # ----- Hypertension Related -----
    if disease in ["Hypertension", "Both"]:
        sbp, dbp = row.get("systolic_bp", np.nan), row.get("diastolic_bp", np.nan)
        if sbp >= 160 or dbp >= 100:
            remedies += [
                "High blood pressure detected – avoid caffeine and salty foods.",
                "Practice 5–10 minutes of relaxed breathing."
            ]
        elif sbp < 100 and dbp < 60:
            remedies += [
                "Low BP detected – sit/lie down, hydrate, and recheck readings."
            ]

    # ----- Heart Rate & Stress -----
    if row.get("heart_rate", np.nan) > 120:
        remedies.append("High resting heart rate – rest and recheck in 15 minutes.")
    if row.get("stress_level", np.nan) > 7:  # stress scale seems like 0–10
        remedies.append("High stress detected – try relaxation (deep breathing, walk).")

    # ----- Sleep -----
    if row.get("sleep_hours", np.nan) < 5:
        remedies.append("Insufficient sleep – prioritize rest to improve recovery.")

    # ----- Severity Specific -----
    if severity == "high":
        remedies.append("Contact your clinician promptly. If unwell, seek emergency care.")
    elif severity == "medium":
        remedies.append("Recheck in 15–30 minutes. If values remain abnormal, contact your clinician.")
    else:
        remedies.append("Maintain healthy habits and continue regular monitoring.")

    return list(dict.fromkeys(remedies))  # remove duplicates


def classify_patient(input_row: dict):
    """Classify disease + severity + remedies for a single input row"""
    row_df = pd.DataFrame([input_row])
    row_scaled = scaler.transform(row_df[X.columns])

    pred = model.predict(row_scaled)[0]
    prob = model.predict_proba(row_scaled).max()

    # Severity thresholding
    if prob >= 0.7:
        severity = "high"
    elif prob >= 0.4:
        severity = "medium"
    else:
        severity = "low"

    remedies = generate_remedies(input_row, pred, severity)

    return {
        "Disease": pred,
        "Severity": severity,
        "Remedies": remedies
    }


# ------------------------------
# 6. Example Patients (cover all remedies)
# ------------------------------
patients = {
    "Case 1 - Hypoglycemia": {   # Low glucose
        "glucose_level": 60, "heart_rate": 85, "steps_per_min": 20,
        "sleep_hours": 7, "stress_level": 4, "systolic_bp": 120, "diastolic_bp": 80
    },
    "Case 2 - Hyperglycemia": {  # High glucose
        "glucose_level": 300, "heart_rate": 88, "steps_per_min": 15,
        "sleep_hours": 6, "stress_level": 5, "systolic_bp": 118, "diastolic_bp": 78
    },
    "Case 3 - Hypertension High": {  # High BP
        "glucose_level": 120, "heart_rate": 90, "steps_per_min": 30,
        "sleep_hours": 6, "stress_level": 6, "systolic_bp": 170, "diastolic_bp": 105
    },
    "Case 4 - Hypotension": {  # Low BP
        "glucose_level": 110, "heart_rate": 80, "steps_per_min": 40,
        "sleep_hours": 7, "stress_level": 3, "systolic_bp": 95, "diastolic_bp": 55
    },
    "Case 5 - Tachycardia": {  # High HR
        "glucose_level": 130, "heart_rate": 130, "steps_per_min": 10,
        "sleep_hours": 7, "stress_level": 5, "systolic_bp": 115, "diastolic_bp": 75
    },
    "Case 6 - High Stress": {  # High stress
        "glucose_level": 125, "heart_rate": 85, "steps_per_min": 25,
        "sleep_hours": 7, "stress_level": 9, "systolic_bp": 118, "diastolic_bp": 78
    },
    "Case 7 - Sleep Deprived": {  # Low sleep
        "glucose_level": 115, "heart_rate": 78, "steps_per_min": 20,
        "sleep_hours": 3, "stress_level": 4, "systolic_bp": 110, "diastolic_bp": 70
    },
    "Case 8 - Healthy": {  # Normal values
        "glucose_level": 110, "heart_rate": 75, "steps_per_min": 40,
        "sleep_hours": 7, "stress_level": 3, "systolic_bp": 120, "diastolic_bp": 80
    }
}

# Run classification on all cases
for name, patient in patients.items():
    result = classify_patient(patient)
    print(f"\n{name}:")
    print("Disease:", result["Disease"])
    print("Severity:", result["Severity"])
    print("Remedies:", result["Remedies"])



import joblib
import os
import json

# --- MODIFIED CODE ---

# 1. Define your specific output directory
output_dir = r'C:\Users\IYYAPPAN\Desktop\health\backend\wear_device'

# 2. Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# 3. Save the trained model object to the specified path
joblib.dump(model, os.path.join(output_dir, 'wearable_model.pkl'))

# 4. Save the scaler object to the specified path
joblib.dump(scaler, os.path.join(output_dir, 'wearable_scaler.pkl'))

# 5. Save the column order to the specified path
with open(os.path.join(output_dir, 'wearable_columns.json'), 'w') as f:
    json.dump(X.columns.tolist(), f)

print(f"\n✅ Model, scaler, and columns have been successfully saved to:\n{output_dir}")
