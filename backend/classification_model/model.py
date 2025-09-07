import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.multioutput import MultiOutputClassifier
import warnings
import ast

warnings.filterwarnings('ignore')

# --- 1. Load Data ---
print("--- Loading data from CSV file... ---")
# Use a raw string (r"...") to handle Windows file paths correctly.
file_path = r"C:\Users\IYYAPPAN\Desktop\model\ChronicDisease_ClassificationFinal3.csv"

try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file was not found at the specified path: {file_path}")
    print("Please make sure the file path is correct and the file exists.")
    exit() # Exit the script if the file can't be found

# --- 2. Prepare Data and Engineer Risk Scores ---

# Convert 'Diseases' column from string to list
try:
    df['Diseases'] = df['Diseases'].apply(ast.literal_eval)
except (ValueError, SyntaxError) as e:
    print(f"Warning: Could not convert the 'Diseases' column. Error: {e}")

# Binarize the 'Diseases' column for the classification model
mlb = MultiLabelBinarizer()
y_classification = mlb.fit_transform(df['Diseases'])
disease_names = mlb.classes_

# --- REVISED Risk Score Engineering Function ---
def create_risk_scores(data, diseases):
    """
    Creates a more realistic risk score (0-100) using clinical thresholds and non-linear interactions.
    """
    print("--- Engineering ADVANCED risk scores with clinical thresholds... ---")
    df_risks = pd.DataFrame(index=data.index)

    # --- Define binary flags for critical conditions ---
    is_stage_2_htn = (data['Systolic_BP'] >= 160) | (data['Diastolic_BP'] >= 100)
    is_stage_1_htn = ((data['Systolic_BP'] >= 140) | (data['Diastolic_BP'] >= 90)) & ~is_stage_2_htn
    has_ckd = data['eGFR'] < 60
    has_prior_stroke = data['History_of_Stroke'] == 1
    is_smoker = data['Smoking_Status'] != 'Never'
    has_pre_diabetes = data['HbA1c'] >= 5.7
    has_high_ldl = data['LDL_Cholesterol'] > 130
    
    for disease in diseases:
        score = pd.Series(0, index=data.index)
        
        if disease == 'Hypertension':
            # --- THRESHOLD LOGIC ---
            # Assign high base scores for diagnosed hypertension stages.
            score += is_stage_2_htn * 85  # Very high base for Stage 2
            score += is_stage_1_htn * 70  # High base for Stage 1
            # Add points for age as a contributing factor
            score += (data['Age'] > 60) * 10
            
        elif disease == 'Heart Disease':
            # --- THRESHOLD LOGIC ---
            # A prior stroke or existing CKD automatically confers high risk.
            score += has_prior_stroke * 30
            score += has_ckd * 25
            score += is_stage_1_htn * 15
            score += is_stage_2_htn * 25 # More points for worse HTN
            score += is_smoker * 15
            score += has_high_ldl * 15
            score += has_pre_diabetes * 10

        elif disease == 'Stroke':
            # --- THRESHOLD LOGIC ---
            # A prior stroke is the #1 risk. Severe HTN dramatically amplifies it.
            score += has_prior_stroke * 60
            score += is_stage_2_htn * 40 # Severe HTN is a major stroke risk
            score += is_stage_1_htn * 20
            score += (is_stage_2_htn & has_prior_stroke) * 10 # Interaction bonus

        elif disease == 'CKidney Disease':
            # --- THRESHOLD LOGIC ---
            # Base the score heavily on whether CKD is present or not.
            score += has_ckd * 70
            # Add points for contributing factors that worsen CKD
            score += (has_ckd & is_stage_1_htn) * 10
            score += (has_ckd & is_stage_2_htn) * 20
            score += (has_ckd & has_pre_diabetes) * 15
        
        elif disease == 'Diabetes':
             # This is the corrected line
             score = (data['HbA1c'] >= 5.7) * 40 + (data['BMI'] > 30) * 15 + (data['Glucose_in_Urine'] == 1) * 15

        else: # Fallback for Asthma, etc.
             disease_mask = data['Diseases'].apply(lambda x: disease in x)
             score[disease_mask] = np.random.uniform(60, 90, size=disease_mask.sum())
             score[~disease_mask] = np.random.uniform(5, 40, size=(~disease_mask).sum())
        
        df_risks[f'{disease}_Risk'] = score.clip(0, 100)
    
    return df_risks

# Create the regression targets (y_regression)
y_regression = create_risk_scores(df, disease_names)
X = df.drop('Diseases', axis=1)

# Split data for all models
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_classification, y_regression, test_size=0.2, random_state=42
)
print("--- Data prepared for both Classification and Regression models. ---")

# --- 3. Build and Train Models ---
# Common preprocessor for all models
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=np.number).columns
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# --- MODEL 1: CLASSIFICATION (with tuned hyperparameters) ---
print("\n--- Training Classification Model ---")
classification_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(LGBMClassifier(
        n_estimators=200,      # More trees
        num_leaves=40,         # More complexity
        learning_rate=0.05,
        reg_alpha=0.1,         # L1 regularization
        reg_lambda=0.1,        # L2 regularization
        random_state=42
    )))
])
classification_pipeline.fit(X_train, y_class_train)
print("--- Classification Model Trained. ---\n")

# --- MODEL 2: REGRESSION (with tuned hyperparameters) ---
print("--- Training Regression Models (One per disease) ---")
regression_models = {}
for disease in disease_names:
    print(f"  Training regressor for {disease}...")
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LGBMRegressor(
            n_estimators=200,      # More trees
            num_leaves=40,         # More complexity
            learning_rate=0.05,
            reg_alpha=0.1,         # L1 regularization
            reg_lambda=0.1,        # L2 regularization
            random_state=42
        ))
    ])
    # Target is the specific risk column for this disease
    y_reg_target = y_reg_train[f'{disease}_Risk']
    reg_pipeline.fit(X_train, y_reg_target)
    regression_models[disease] = reg_pipeline
print("--- All Regression Models Trained. ---\n")


# --- 4. User Input and Prediction ---

def get_numeric_input(prompt, value_type=float):
    """Prompts the user for numeric input and validates it."""
    while True:
        try:
            user_input = input(prompt)
            return value_type(user_input)
        except ValueError:
            print(f"Invalid input. Please enter a valid number.")

def get_categorical_input(prompt, options):
    """Prompts the user for categorical input from a list of options."""
    options_lower = [opt.lower() for opt in options]
    prompt_with_options = f"{prompt} ({'/'.join(options)}): "
    while True:
        user_input = input(prompt_with_options).strip().lower()
        if user_input in options_lower:
            # Return the original capitalized version
            return options[options_lower.index(user_input)]
        else:
            print(f"Invalid choice. Please select from: {', '.join(options)}")

def get_patient_data_from_user():
    """Gathers all necessary patient data from the user via prompts."""
    print("\n\n--- Please Enter New Patient Data for Diagnosis ---")
    patient_data = {}
    
    print("\n--- Basic Information ---")
    patient_data['Gender'] = get_categorical_input("Gender", ['Male', 'Female'])
    patient_data['Age'] = get_numeric_input("Age (years): ", int)
    patient_data['BMI'] = get_numeric_input("BMI (e.g., 24.5): ")
    patient_data['Smoking_Status'] = get_categorical_input("Smoking Status", ['Never', 'Former', 'Current'])
    patient_data['History_of_Stroke'] = int(get_categorical_input("History of stroke? (0 for No, 1 for Yes)", ['0', '1']))
    
    print("\n--- Vitals ---")
    patient_data['Systolic_BP'] = get_numeric_input("Systolic Blood Pressure (mmHg): ", int)
    patient_data['Diastolic_BP'] = get_numeric_input("Diastolic Blood Pressure (mmHg): ", int)
    patient_data['Heart_Rate'] = get_numeric_input("Heart Rate (bpm): ", int)
    patient_data['Respiratory_Rate'] = get_numeric_input("Respiratory Rate (breaths/min): ", int)
    
    print("\n--- Lab Results ---")
    patient_data['FBS'] = get_numeric_input("Fasting Blood Sugar (mg/dL): ")
    patient_data['HbA1c'] = get_numeric_input("HbA1c (%): ")
    patient_data['Serum_Creatinine'] = get_numeric_input("Serum Creatinine (mg/dL): ")
    patient_data['eGFR'] = get_numeric_input("eGFR (mL/min/1.73m²): ")
    patient_data['BUN'] = get_numeric_input("Blood Urea Nitrogen (mg/dL): ")
    patient_data['Total_Cholesterol'] = get_numeric_input("Total Cholesterol (mg/dL): ")
    patient_data['LDL_Cholesterol'] = get_numeric_input("LDL Cholesterol (mg/dL): ")
    patient_data['HDL_Cholesterol'] = get_numeric_input("HDL Cholesterol (mg/dL): ")
    patient_data['Triglycerides'] = get_numeric_input("Triglycerides (mg/dL): ")
    patient_data['Hemoglobin'] = get_numeric_input("Hemoglobin (g/dL): ")
    patient_data['Urine_Albumin_ACR'] = get_numeric_input("Urine Albumin-to-Creatinine Ratio (mg/g): ")
    patient_data['Glucose_in_Urine'] = int(get_categorical_input("Glucose present in urine? (0 for No, 1 for Yes)", ['0', '1']))
    patient_data['FEV1_FVC_Ratio'] = get_numeric_input("FEV1/FVC Ratio (e.g., 0.80): ")
    
    return patient_data

patient_input_data = get_patient_data_from_user()
input_df = pd.DataFrame([patient_input_data])

# --- 5. Generate and Display Report ---

# Helper functions for the report
def get_risk_level(score):
    if score < 35: return "Low"
    elif score < 70: return "Medium"
    else: return "High"

def get_disease_explanation(patient_data, disease):
    """
    Generates a detailed, data-driven explanation for a disease prediction
    based on common medical guidelines and risk factors.
    """
    reasons = []
    
    # 1. Chronic Kidney Disease (CKD)
    if disease == 'CKidney Disease':
        if patient_data['eGFR'] < 60:
            reasons.append(f"low eGFR ({patient_data['eGFR']} mL/min/1.73m²), indicating reduced kidney function")
        if patient_data['Urine_Albumin_ACR'] > 30:
            reasons.append(f"high Urine Albumin-to-Creatinine Ratio ({patient_data['Urine_Albumin_ACR']} mg/g), a sign of kidney damage (albuminuria)")
        if patient_data['Serum_Creatinine'] > 1.2:
            reasons.append(f"high Serum Creatinine ({patient_data['Serum_Creatinine']} mg/dL)")
        if patient_data['Systolic_BP'] >= 130 or patient_data['Diastolic_BP'] >= 80:
            reasons.append("co-existing high blood pressure")
        if patient_data['HbA1c'] >= 5.7:
             reasons.append("co-existing high blood sugar")

    # 2. Diabetes
    elif disease == 'Diabetes':
        if patient_data['HbA1c'] >= 6.5:
            reasons.append(f"very high HbA1c ({patient_data['HbA1c']}%), confirming diabetes")
        elif 5.7 <= patient_data['HbA1c'] < 6.5:
            reasons.append(f"elevated HbA1c ({patient_data['HbA1c']}%) in the pre-diabetic range")
        if patient_data['FBS'] >= 126:
            reasons.append(f"high Fasting Blood Sugar ({patient_data['FBS']} mg/dL)")
        if patient_data['Glucose_in_Urine'] == 1:
            reasons.append("presence of glucose in urine")
        if patient_data['BMI'] > 30:
            reasons.append(f"obesity (BMI of {patient_data['BMI']})")

    # 3. Hypertension
    elif disease == 'Hypertension':
        if patient_data['Systolic_BP'] >= 140 or patient_data['Diastolic_BP'] >= 90:
            reasons.append(f"high blood pressure (Stage 2) at {patient_data['Systolic_BP']}/{patient_data['Diastolic_BP']} mmHg")
        elif 130 <= patient_data['Systolic_BP'] < 140 or 80 <= patient_data['Diastolic_BP'] < 90:
            reasons.append(f"elevated blood pressure (Stage 1) at {patient_data['Systolic_BP']}/{patient_data['Diastolic_BP']} mmHg")
        if patient_data['Age'] > 60:
            reasons.append("advanced age")

    # 4. Heart Disease
    elif disease == 'Heart Disease':
        if patient_data['LDL_Cholesterol'] > 130:
            reasons.append(f"high LDL ('bad') cholesterol ({patient_data['LDL_Cholesterol']} mg/dL)")
        if patient_data['HDL_Cholesterol'] < 40:
            reasons.append(f"low HDL ('good') cholesterol ({patient_data['HDL_Cholesterol']} mg/dL)")
        if patient_data['Triglycerides'] > 150:
            reasons.append(f"high triglycerides ({patient_data['Triglycerides']} mg/dL)")
        if patient_data['Systolic_BP'] >= 130:
             reasons.append("presence of high blood pressure")
        if patient_data['Smoking_Status'] != 'Never':
            reasons.append("history of smoking")

    # 5. Stroke
    elif disease == 'Stroke':
        if patient_data['History_of_Stroke'] == 1:
            reasons.append("a prior history of stroke")
        if patient_data['Systolic_BP'] > 180 or patient_data['Diastolic_BP'] > 120:
            reasons.append(f"critically high blood pressure ({patient_data['Systolic_BP']}/{patient_data['Diastolic_BP']} mmHg)")
        if patient_data['Systolic_BP'] >= 140 or patient_data['Diastolic_BP'] >= 90:
             reasons.append("uncontrolled hypertension")
        if patient_data['HbA1c'] >= 6.5:
             reasons.append("uncontrolled diabetes")

    # 6. Asthma
    elif disease == 'Asthma':
        if patient_data['FEV1_FVC_Ratio'] < 0.7:
            reasons.append(f"low FEV1/FVC ratio ({patient_data['FEV1_FVC_Ratio']}), indicating airway obstruction")
        if patient_data['Respiratory_Rate'] > 20:
            reasons.append(f"high respiratory rate ({patient_data['Respiratory_Rate']} breaths/min)")
            
    # Fallback message if no specific rules are met
    return ", ".join(reasons) if reasons else "a combination of multiple sub-clinical factors."


# --- STEP 1: RUN CLASSIFICATION MODEL ---
predicted_diseases_matrix = classification_pipeline.predict(input_df)
predicted_diseases_list = mlb.inverse_transform(predicted_diseases_matrix)[0]

print("\n\n==============================================")
print("          FINAL DIAGNOSTIC REPORT          ")
print("==============================================")

# --- Display Classification Results ---
print("\n--- Part 1: Predicted Conditions ---")
if not predicted_diseases_list:
    print("✅ The classification model predicts the patient is Healthy.")
else:
    print("The model predicts the patient may have the following conditions:")
    for disease in predicted_diseases_list:
        reason = get_disease_explanation(patient_input_data, disease)
        print(f"  - *{disease}*: Detected based on {reason}.")

# --- STEP 2: RUN REGRESSION MODELS (ONLY FOR PREDICTED DISEASES) ---
print("\n--- Part 2: Detailed Risk Assessment ---")
if not predicted_diseases_list:
    print("No risk assessment needed as no diseases were predicted.")
else:
    for disease in predicted_diseases_list:
        # Select the correct regression model
        reg_model = regression_models[disease]
        
        # Predict the risk score
        risk_score = reg_model.predict(input_df)[0]
        risk_score = max(0, min(100, risk_score)) # Ensure score is 0-100
        
        # Get level and explanation
        risk_level = get_risk_level(risk_score)
        explanation = get_disease_explanation(patient_input_data, disease) # Re-using explanation for consistency
        
        print(f"\n  ------------------------------------")
        print(f"  Risk Analysis for: *{disease}*")
        print(f"  ------------------------------------")
        print(f"  - *Risk Score*: {risk_score:.1f} / 100")
        print(f"  - *Risk Level*: {risk_level}")
        print(f"  - *Primary Drivers*: This risk level is driven by {explanation}.")

print("\n==============================================")
print("Disclaimer: This is an AI-generated prediction and not a substitute for professional medical advice.")
print("==============================================\n")



# --- ADD THIS CODE TO THE END OF YOUR TRAINING SCRIPT ---
import pickle

print("\n--- 5. Saving models to pickle files... ---")

# Save the main classification pipeline
with open('classification_model.pkl', 'wb') as f:
    pickle.dump(classification_pipeline, f)
print("Saved: classification_model.pkl")

# Save the dictionary of regression models
with open('regression_models.pkl', 'wb') as f:
    pickle.dump(regression_models, f)
print("Saved: regression_models.pkl")

# Save the MultiLabelBinarizer which knows the disease names
with open('mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)
print("Saved: mlb.pkl")

print("\n--- All models and the binarizer have been saved successfully. ---")