# ==============================
# Wearable Dataset Classification + Advanced Rule Remedies (Improved)
# ==============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import lightgbm as lgb
import joblib
# ------------------------------
# 1. Load dataset
# ------------------------------
df = pd.read_excel(r"D:\React-Projects\cts\Helath_app\backend\wear_device\wearable_Dataset_augmented.xlsx")

FEATURE_COLS = [
    "glucose_level", "heart_rate", "steps_per_min", "sleep_hours",
    "stress_level", "systolic_bp", "diastolic_bp"
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
X = df[FEATURE_COLS].copy()
y = df["label"].copy()

# ------------------------------
# 2. Train-test split + scaling
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 3. Train classifier (LGBM → fallback RF)
# ------------------------------
try:
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, n_jobs=-1, random_state=42
    )
    model.fit(X_train_scaled, y_train)
except Exception as e:
    print("LightGBM failed, falling back to RandomForest:", e)
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train_scaled, y_train)

# ------------------------------
# 4. Evaluate quickly
# ------------------------------
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# ------------------------------
# 5. Advanced Rule-based Remedy Engine (Improved)
# ------------------------------
def generate_rule_based_remedy(meas: dict) -> str:
    g = meas.get("glucose_level", np.nan)
    sbp = meas.get("systolic_bp", np.nan)
    dbp = meas.get("diastolic_bp", np.nan)
    hr = meas.get("heart_rate", np.nan)
    stress = meas.get("stress_level", np.nan)
    sleep = meas.get("sleep_hours", np.nan)

    remedies = []

    # --- Glucose (range-specific) ---
    if not np.isnan(g):
        if g < 54:
            remedies.append(
                "Severe hypoglycemia (<54 mg/dL): take 15–20 g fast-acting carbohydrate, recheck in 15 min. "
                "Seek urgent care if symptoms persist."
            )
        elif g < 70:
            remedies.append(
                "Low glucose (54–69 mg/dL): take 15–20 g fast-acting carbohydrate, recheck in 15 min."
            )
        elif g <= 180:
            pass  # normal range
        elif g <= 250:
            remedies.append(
                "Elevated glucose (181–250 mg/dL): drink water, avoid carbs, recheck in 1 hour."
            )
        elif g <= 300:
            remedies.append(
                "High glucose (251–300 mg/dL): hydrate, recheck in 30–60 min; if persistent, contact clinician."
            )
        else:  # g > 300
            remedies.append(
                "Very high glucose (>300 mg/dL): check ketones (if applicable), contact healthcare provider promptly."
            )

    # --- Blood pressure (range-specific) ---
    if not (np.isnan(sbp) or np.isnan(dbp)):
        if sbp >= 180 or dbp >= 120:
            remedies.append(
                "Hypertensive crisis: if chest pain/severe headache/vision issues — seek urgent care immediately."
            )
        elif sbp >= 160 or dbp >= 100:
            remedies.append(
                "High BP (≥160/100): sit, do deep breathing 5 minutes, recheck in 30 min, consult clinician."
            )
        elif sbp < 90 or dbp < 60:
            remedies.append(
                "Low BP (<90/60): lie down, drink water, rise slowly, seek care if symptoms persist."
            )

    # --- Heart rate ---
    if not np.isnan(hr):
        if hr >= 130:
            remedies.append(
                "Very high HR (≥130): stop activity, rest, recheck; seek urgent care if symptomatic."
            )
        elif hr > 100:
            remedies.append(
                "Elevated HR (101–129): rest, hydrate, recheck in 10–15 min."
            )
        elif hr <= 40:
            remedies.append(
                "Low HR (≤40): if dizzy or fainting, seek urgent medical help."
            )

    # --- Stress & Sleep ---
    if not np.isnan(stress) and stress >= 8:
        remedies.append("High stress: pause, do deep breathing, reduce stimulants.")
    elif not np.isnan(stress) and stress >= 6:
        remedies.append("Moderate stress: brief relaxation and re-evaluate.")

    if not np.isnan(sleep) and sleep < 5:
        remedies.append("Short sleep: rest if possible; aim for 7–8h nightly.")

    # --- Combined-condition logic ---
    if (not np.isnan(g) and g > 180) and (not (np.isnan(sbp) or np.isnan(dbp)) and (sbp >= 160 or dbp >= 100)):
        remedies.append(
            "High glucose + high BP together: avoid strenuous activity, contact clinician within 24h, review meds."
        )

    if (not np.isnan(g) and g > 250) and (not np.isnan(hr) and hr > 100):
        remedies.append(
            "High glucose with high HR: hydrate well, recheck both, contact clinician if persistent."
        )

    if (not np.isnan(g) and g < 70) and (not (np.isnan(sbp) or np.isnan(dbp)) and (sbp < 90 or dbp < 60)):
        remedies.append(
            "Low glucose + low BP: take carbs, lie down, recheck both, seek care if symptoms persist."
        )

    # --- Deduplication ---
    seen, unique_remedies = set(), []
    for r in remedies:
        if r not in seen:
            seen.add(r)
            unique_remedies.append(r)

    if not unique_remedies:
        return "No immediate action — keep monitoring."
    return " | ".join(unique_remedies)

# ------------------------------
# 6. Classify + Remedy
# ------------------------------
def classify_and_remedy(input_row: dict):
    model_input = {col: input_row.get(col, np.nan) for col in FEATURE_COLS}
    row_df = pd.DataFrame([model_input])
    row_scaled = scaler.transform(row_df)

    pred_label = model.predict(row_scaled)[0]
    remedy = generate_rule_based_remedy(model_input)

    return {"Predicted": pred_label, "Remedy": remedy}

# ------------------------------
# 7. Example Patients
# ------------------------------
patients = {
    "Case 1 - Diabetes Mild": {
        "glucose_level": 190, "heart_rate": 85, "steps_per_min": 50,
        "sleep_hours": 6, "stress_level": 4, "systolic_bp": 120, "diastolic_bp": 80
    },
    "Case 2 - Diabetes Severe": {
        "glucose_level": 320, "heart_rate": 88, "steps_per_min": 40,
        "sleep_hours": 5, "stress_level": 6, "systolic_bp": 118, "diastolic_bp": 78
    },
    "Case 3 - Diabetes Hypo": {
    "glucose_level": 65,       # still hypoglycemia (<70)
    "heart_rate": 110,         # elevated (tachycardia, common in hypo)
    "steps_per_min": 10,       # very low activity (resting due to hypo)
    "sleep_hours": 4,          # poor sleep, adds abnormality
    "stress_level": 7,         # higher stress
    "systolic_bp": 100,
    "diastolic_bp": 65
},
    "Case 4 - Hypertension Severe": {
        "glucose_level": 110, "heart_rate": 80, "steps_per_min": 45,
        "sleep_hours": 6, "stress_level": 7, "systolic_bp": 180, "diastolic_bp": 120
    },
    "Case 5 - Hypertension Moderate": {
        "glucose_level": 115, "heart_rate": 75, "steps_per_min": 60,
        "sleep_hours": 7, "stress_level": 5, "systolic_bp": 160, "diastolic_bp": 100
    },
    "Case 6 - Hypotension": {
        "glucose_level": 100, "heart_rate": 72, "steps_per_min": 55,
        "sleep_hours": 8, "stress_level": 3, "systolic_bp": 85, "diastolic_bp": 55
    },
    "Case 7 - Both Diabetes+HTN": {
        "glucose_level": 280, "heart_rate": 95, "steps_per_min": 35,
        "sleep_hours": 6, "stress_level": 8, "systolic_bp": 165, "diastolic_bp": 100
    },
    "Case 8 - Both Moderate": {
        "glucose_level": 220, "heart_rate": 105, "steps_per_min": 25,
        "sleep_hours": 5, "stress_level": 7, "systolic_bp": 160, "diastolic_bp": 95
    },
    "Case 9 - Healthy": {
        "glucose_level": 105, "heart_rate": 75, "steps_per_min": 65,
        "sleep_hours": 7, "stress_level": 3, "systolic_bp": 120, "diastolic_bp": 80
    }
}

# ------------------------------
# 8. Run Predictions
# ------------------------------
for name, p in patients.items():
    out = classify_and_remedy(p)
    print(f"\n{name}:")
    print("Disease:", out["Predicted"])
    print("Remedy:", out["Remedy"])
    

joblib.dump(model, "wearable_model.pkl")
joblib.dump(scaler, "wearable_scaler.pkl")
print("✅ Model and Scaler saved successfully!")