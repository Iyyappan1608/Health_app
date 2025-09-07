import pickle
import os
import sys
import pandas as pd
import numpy as np
import random
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
import os
import traceback
from datetime import datetime, timedelta, timezone
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import text
from care_plan_generator import generate_care_plan_from_report, split_days
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# --- Import the AI Care Plan Generator ---
try:
    from care_plan_generator import generate_care_plan_from_report, split_days
    print("--- Care Plan Generator loaded successfully! ---")
except ImportError as e:
    print(f"!!! Could not import care_plan_generator.py: {e}. Care plan endpoint will not work. !!!")
    def generate_care_plan_from_report(report_text):
        return "Error: Care plan generator module not found."

# --- 1. SETUP ---
app = Flask(__name__)
bcrypt = Bcrypt(app)
CORS(app, resources={r"/*": {"origins": ["http://localhost:8081", "http://127.0.0.1:8081"]}})
# Make sure your password special characters are URL-encoded (e.g., '@' becomes '%40')
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:1234@localhost:3307/health_app_db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- 2. DATABASE MODELS ---
class Patient(db.Model):
    __tablename__ = 'patients'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

class UserPrediction(db.Model):
    __tablename__ = 'user_predictions'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    prediction_type = db.Column(db.Enum('chronic_disease', 'diabetes_subtype', 'hypertension', 'vitals', 'general_health', 'wearable'), nullable=False)
    page_source = db.Column(db.String(100))
    input_data = db.Column(db.JSON)
    output_data = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patient = db.relationship('Patient', backref='predictions')

class UserSession(db.Model):
    __tablename__ = 'user_sessions'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    session_token = db.Column(db.String(255), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    patient = db.relationship('Patient', backref='sessions')

class DiabetesCheck(db.Model):
    __tablename__ = 'diabetes_checks'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    is_pregnant = db.Column(db.Boolean)
    age_at_diagnosis = db.Column(db.Integer)
    bmi_at_diagnosis = db.Column(db.Float)
    family_history = db.Column(db.String(50))
    hba1c = db.Column(db.Float)
    c_peptide_level = db.Column(db.Float)  # Note: Your table has 'c_perbide_level' (typo)
    autoantibodies_status = db.Column(db.String(50))
    genetic_test_result = db.Column(db.String(50))
    report_json = db.Column(db.Text)  # Note: Your table has 'report.json' with a dot
    created_at = db.Column(db.TIMESTAMP, default=datetime.utcnow)
    patient = db.relationship('Patient', backref='diabetes_checks')

class HypertensionCheck(db.Model):
    __tablename__ = 'hypertension_checks'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    age = db.Column(db.Integer)
    sex = db.Column(db.Integer)
    bmi = db.Column(db.Float)
    family_history = db.Column(db.Boolean)
    creatinine = db.Column(db.Float)
    systolic_bp = db.Column(db.Integer)
    diastolic_bp = db.Column(db.Integer)
    report_json = db.Column(db.Text)
    created_at = db.Column(db.TIMESTAMP, default=datetime.utcnow)
    patient = db.relationship('Patient', backref='hypertension_checks')
    
class UserActivityLog(db.Model):
    __tablename__ = 'user_activity_log'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    activity_type = db.Column(db.String(100))
    page_visited = db.Column(db.String(100))
    details = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patient = db.relationship('Patient', backref='activities')

class CarePlan(db.Model):
    __tablename__ = 'care_plans'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    source_data_summary = db.Column(db.Text, nullable=False)
    generated_plan = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patient = db.relationship('Patient', backref='care_plans')
    
class UserCarePlanInteraction(db.Model):
    __tablename__ = 'user_care_plan_interactions'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    day_number = db.Column(db.Integer, nullable=False)
    question = db.Column(db.String(100), nullable=False)
    response = db.Column(db.String(500), nullable=False)
    care_plan_text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patient = db.relationship('Patient', backref='care_plan_interactions')

class CarePlanProgress(db.Model):
    __tablename__ = 'care_plan_progress'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    current_day = db.Column(db.Integer, default=1)
    last_completed_day = db.Column(db.Integer, default=0)
    last_accessed = db.Column(db.DateTime, default=datetime.utcnow)
    is_completed = db.Column(db.Boolean, default=False)
    patient = db.relationship('Patient', backref='care_plan_progress')

with app.app_context():
    db.create_all()
    print("Database tables initialized successfully!")

# --- HypertensionPipeline class ---
class HypertensionPipeline(BaseEstimator):
    def __init__(self):
        self.pipeline = self._create_pipeline()
    def _create_pipeline(self):
        numerical_features = ['age', 'bmi', 'creatinine', 'systolic_bp', 'diastolic_bp']
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features)])
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        return pipeline
    def fit(self, X, y): return self.pipeline.fit(X, y)
    def predict(self, X): return self.pipeline.predict(X)
    def predict_proba(self, X): return self.pipeline.predict_proba(X)
    def get_params(self, deep=True): return self.pipeline.get_params(deep=deep)
    def set_params(self, **params): return self.pipeline.set_params(**params)

class UserChatInteraction(db.Model):
    __tablename__ = 'user_chat_interactions'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    bot_response = db.Column(db.Text, nullable=False)
    health_context = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patient = db.relationship('Patient', backref='chat_interactions')

# --- 3. MODEL LOADING ---
# --- 3. MODEL LOADING ---
models = {}

# Base directory of backend folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    # Load Chronic Disease Models
    with open(os.path.join(BASE_DIR, 'pkl', 'classification_model.pkl'), 'rb') as f:
        models['chronic_classifier'] = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'pkl', 'regression_models.pkl'), 'rb') as f:
        models['chronic_regressors'] = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'pkl', 'mlb.pkl'), 'rb') as f:
        models['chronic_mlb'] = pickle.load(f)
    print("--- Chronic disease models loaded successfully! ---")

    # Load Diabetes Subtype Models
    models['diabetes_subtype_classifier'] = joblib.load(os.path.join(BASE_DIR, 'diabetes_class', 'lgbm_classifier.pkl'))
    models['diabetes_risk_model'] = joblib.load(os.path.join(BASE_DIR, 'diabetes_class', 'kmeans_model.pkl'))
    models['diabetes_preprocessor'] = joblib.load(os.path.join(BASE_DIR, 'diabetes_class', 'preprocessor.pkl'))
    models['diabetes_cluster_map'] = joblib.load(os.path.join(BASE_DIR, 'diabetes_class', 'cluster_risk_map.pkl'))
    print("--- Diabetes subtype models loaded successfully! ---")

    # Load Hypertension Model
    models['hypertension_pipeline'] = joblib.load(os.path.join(BASE_DIR, 'hypertension_class', 'complete_pipeline.pkl'))
    print("--- Hypertension main pipeline loaded successfully! ---")

except Exception as e:
    print(f"!!! WARNING: Error loading one or more models: {e}. Some endpoints may not work. !!!")

# Load Wearable Models
wearable_model_path = os.path.join(BASE_DIR, 'wear_device')

try:
    models['wearable_model'] = joblib.load(os.path.join(wearable_model_path, 'wearable_model.pkl'))
    models['wearable_scaler'] = joblib.load(os.path.join(wearable_model_path, 'wearable_scaler.pkl'))
    with open(os.path.join(wearable_model_path, 'wearable_columns.json'), 'r', encoding='utf-8') as f:
        models['wearable_columns'] = json.load(f)
    print("--- Wearable models loaded successfully! ---")
except Exception as e:
    print(f"!!! WARNING: Error loading wearable models: {e} !!!")
    

# --- CHATBOT SETUP ---
# --- CHATBOT SETUP ---
def setup_chatbot():
    """Initialize the chatbot with Groq API"""
    api_key = "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"  # Fixed: removed repeated characters
    if not api_key:
        print("❌ Missing Groq API key - chatbot will not work")
        return None
    
    try:
        # Initialize LLM
        llm_engine = ChatGroq(
            groq_api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_retries=2,
            timeout=30,
        )
        
        # Enhanced system prompt with medical knowledge
        system_prompt = """You are an experienced and empathetic medical assistant named HealthAI. You have access to the patient's health data and can provide personalized advice.

GUIDELINES:
1. Always be supportive, compassionate, and professional
2. Use the patient's health context to provide personalized responses
3. Explain medical concepts in simple, understandable terms
4. If unsure about something, admit it and suggest consulting a healthcare provider
5. For emergencies, always recommend seeking immediate medical attention
6. Keep responses concise but thorough (3-5 sentences typically)
7. Focus on prevention, management, and healthy lifestyle recommendations

SPECIALTIES:
- Diabetes management and prevention
- Hypertension and heart health
- Chronic disease management
- Nutrition and exercise advice
- Medication understanding
- Symptom interpretation

Remember: You are an AI assistant, not a replacement for professional medical care."""
        
        # Build prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{user_input}")
        ])
        
        # Create chain
        chain = prompt | llm_engine | StrOutputParser()
        print("✅ Chatbot initialized successfully")
        return chain
        
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        return None

# Initialize chatbot
chatbot_chain = setup_chatbot()


# --- 4. ML HELPER FUNCTIONS ---
def get_risk_level(score):
    if score < 35: return "Low"
    elif score < 70: return "Medium"
    else: return "High"

def get_disease_explanation(patient_data, disease):
    reasons = []
    if disease == 'CKidney Disease':
        if patient_data.get('eGFR', 100) < 60: reasons.append(f"low eGFR ({patient_data.get('eGFR')})")
    elif disease == 'Diabetes':
        if patient_data.get('HbA1c', 0) >= 6.5: reasons.append(f"very high HbA1c ({patient_data.get('HbA1c')}%)")
    elif disease == 'Hypertension':
        if patient_data.get('Systolic_BP', 0) >= 140 or patient_data.get('Diastolic_BP', 0) >= 90: reasons.append(f"high blood pressure (Stage 2)")
    elif disease == 'Heart Disease':
        if patient_data.get('LDL_Cholesterol', 0) > 130: reasons.append(f"high LDL cholesterol")
    elif disease == 'Stroke':
        if patient_data.get('History_of_Stroke') == 1: reasons.append("a prior history of stroke")
    elif disease == 'Asthma':
        if patient_data.get('FEV1_FVC_Ratio', 1) < 0.7: reasons.append(f"low FEV1/FVC ratio")
    return ", ".join(reasons) if reasons else "a combination of sub-clinical factors."

def generate_hypertension_explanation(prediction, patient_data, probability, stage, subtype):
    explanation_parts = []
    if prediction:
        explanation_parts.append(f"The model predicts {subtype} hypertension ({stage}) with {probability:.1%} confidence.")
        explanation_parts.append("Primary contributing factors:")
        if patient_data.get('systolic_bp', 0) >= 140: explanation_parts.append(f"• Elevated systolic BP ({patient_data.get('systolic_bp')} mmHg)")
        if patient_data.get('diastolic_bp', 0) >= 90: explanation_parts.append(f"• Elevated diastolic BP ({patient_data.get('diastolic_bp')} mmHg)")
        if patient_data.get('age', 0) > 50: explanation_parts.append(f"• Age ({patient_data.get('age')} years)")
        if patient_data.get('bmi', 0) >= 25: explanation_parts.append(f"• BMI ({patient_data.get('bmi')})")
        if patient_data.get('family_history', 0) == 1: explanation_parts.append("• Family history")
        if patient_data.get('creatinine', 0) > 1.2: explanation_parts.append(f"• Creatinine level ({patient_data.get('creatinine')} mg/dL)")
    else:
        explanation_parts.append("No hypertension detected. Blood pressure levels appear normal.")
        if patient_data.get('systolic_bp', 0) < 120 and patient_data.get('diastolic_bp', 0) < 80:
            explanation_parts.append("• Optimal blood pressure range")
    return "\n".join(explanation_parts)

def calculate_hypertension_risks(systolic_bp, diastolic_bp, age, creatinine):
    bp_risk_factor = max(0, (systolic_bp - 120) / 40 + (diastolic_bp - 80) / 20)
    age_factor = max(0, (age - 40) / 30)
    creatinine_factor = max(0, (creatinine - 0.8) / 0.4)
    kidney_risk = min(95, 30 + bp_risk_factor * 25 + age_factor * 15 + creatinine_factor * 20)
    stroke_risk = min(95, 25 + bp_risk_factor * 30 + age_factor * 20)
    heart_risk = min(95, 20 + bp_risk_factor * 28 + age_factor * 18)
    return kidney_risk, stroke_risk, heart_risk

def determine_hypertension_stage(systolic_bp, diastolic_bp):
    if systolic_bp >= 180 or diastolic_bp >= 120: return "Stage_3", "Hypertensive_Crisis"
    elif systolic_bp >= 160 or diastolic_bp >= 100: return "Stage_2", "Established_Hypertension"
    elif systolic_bp >= 140 or diastolic_bp >= 90: return "Stage_1", "Primary_Hypertension"
    elif systolic_bp >= 130 or diastolic_bp >= 80: return "Elevated", "Prehypertension"
    else: return "Normal", "Optimal"

def determine_hypertension_subtype(patient_data):
    if patient_data.get('family_history', 0) == 1: return "Familial"
    elif patient_data.get('creatinine', 1.0) > 1.3: return "Secondary_Renal"
    elif patient_data.get('bmi', 25) > 30: return "Obesity_Related"
    else: return "Primary_Essential"

def generate_remedies(row: pd.DataFrame, disease: str, severity: str) -> list:
    # ... (this function is unchanged)
    remedies = []
    vitals = row.iloc[0]

    if disease in ["Diabetes", "Both"]:
        if vitals.get("glucose_level", np.nan) < 70: remedies += ["Low glucose detected – consume fast-acting carbs (juice, glucose tablets).", "Avoid driving or hazardous activity until glucose stabilizes."]
        elif vitals.get("glucose_level", np.nan) > 250: remedies += ["High glucose detected – hydrate with water.", "Avoid high-carb food and monitor glucose closely."]
    if disease in ["Hypertension", "Both"]:
        sbp, dbp = vitals.get("systolic_bp", np.nan), vitals.get("diastolic_bp", np.nan)
        if sbp >= 160 or dbp >= 100: remedies += ["High blood pressure detected – avoid caffeine and salty foods.", "Practice 5–10 minutes of relaxed breathing."]
        elif sbp < 100 and dbp < 60: remedies += ["Low BP detected – sit/lie down, hydrate, and recheck readings."]
    if vitals.get("heart_rate", np.nan) > 120: remedies.append("High resting heart rate – rest and recheck in 15 minutes.")
    if vitals.get("stress_level", np.nan) > 7: remedies.append("High stress detected – try relaxation (deep breathing, walk).")
    if vitals.get("sleep_hours", np.nan) < 5: remedies.append("Insufficient sleep – prioritize rest to improve recovery.")
    if severity == "high": remedies.append("Contact your clinician promptly. If unwell, seek emergency care.")
    elif severity == "medium": remedies.append("Recheck in 15–30 minutes. If values remain abnormal, contact your clinician.")
    else:
        if not remedies: remedies.append("Maintain healthy habits and continue regular monitoring.")
    return list(dict.fromkeys(remedies))

def generate_rule_based_remedy(meas: dict) -> str:
    g = meas.get("glucose_level", np.nan)
    sbp = meas.get("systolic_bp", np.nan)
    dbp = meas.get("diastolic_bp", np.nan)
    hr = meas.get("heart_rate", np.nan)
    stress = meas.get("stress_level", np.nan)
    sleep = meas.get("sleep_hours", np.nan)

    remedies = []

    if not np.isnan(g):
        if g < 54:
            remedies.append("Severe hypoglycemia (<54 mg/dL): take 15–20 g fast-acting carbohydrate, recheck in 15 min. Seek urgent care if symptoms persist.")
        elif g < 70:
            remedies.append("Low glucose (54–69 mg/dL): take 15–20 g fast-acting carbohydrate, recheck in 15 min.")
        elif g <= 180:
            pass
        elif g <= 250:
            remedies.append("Elevated glucose (181–250 mg/dL): drink water, avoid carbs, recheck in 1 hour.")
        elif g <= 300:
            remedies.append("High glucose (251–300 mg/dL): hydrate, recheck in 30–60 min; if persistent, contact clinician.")
        else:
            remedies.append("Very high glucose (>300 mg/dL): check ketones, contact healthcare provider promptly.")

    if not (np.isnan(sbp) or np.isnan(dbp)):
        if sbp >= 180 or dbp >= 120:
            remedies.append("Hypertensive crisis: if chest pain/severe headache/vision issues — seek urgent care immediately.")
        elif sbp >= 160 or dbp >= 100:
            remedies.append("High BP (≥160/100): sit, do deep breathing, recheck in 30 min, consult clinician.")
        elif sbp < 90 or dbp < 60:
            remedies.append("Low BP (<90/60): lie down, drink water, rise slowly, seek care if symptoms persist.")

    if not np.isnan(hr):
        if hr >= 130:
            remedies.append("Very high HR (≥130): stop activity, rest, recheck; urgent care if symptomatic.")
        elif hr > 100:
            remedies.append("Elevated HR (101–129): rest, hydrate, recheck in 10–15 min.")
        elif hr <= 40:
            remedies.append("Low HR (≤40): if dizzy or fainting, seek urgent medical help.")

    if not np.isnan(stress) and stress >= 8:
        remedies.append("High stress: pause, do deep breathing, reduce stimulants.")
    elif not np.isnan(stress) and stress >= 6:
        remedies.append("Moderate stress: brief relaxation and re-evaluate.")

    if not np.isnan(sleep) and sleep < 5:
        remedies.append("Short sleep: rest if possible; aim for 7–8h nightly.")

    if (not np.isnan(g) and g > 180) and (not (np.isnan(sbp) or np.isnan(dbp)) and (sbp >= 160 or dbp >= 100)):
        remedies.append("High glucose + high BP: avoid strenuous activity, contact clinician within 24h, review meds.")

    if (not np.isnan(g) and g > 250) and (not np.isnan(hr) and hr > 100):
        remedies.append("High glucose with high HR: hydrate well, recheck both, contact clinician if persistent.")

    if (not np.isnan(g) and g < 70) and (not (np.isnan(sbp) or np.isnan(dbp)) and (sbp < 90 or dbp < 60)):
        remedies.append("Low glucose + low BP: take carbs, lie down, recheck both, seek care if symptoms persist.")

    seen, unique_remedies = set(), []
    for r in remedies:
        if r not in seen:
            seen.add(r)
            unique_remedies.append(r)

    return " | ".join(unique_remedies) if unique_remedies else "No immediate action — keep monitoring."


# ------------------------------
# Normalize vitals into model format
# ------------------------------
def normalize_vitals(raw: dict) -> dict:
    """
    Convert live-monitor raw vitals (HR, BP, GLUCOSE, etc.) 
    into model input (glucose_level, heart_rate, systolic_bp, diastolic_bp, etc.)
    """
    model_input = {}
    if "HR" in raw: model_input["heart_rate"] = raw["HR"]
    if "GLUCOSE" in raw: model_input["glucose_level"] = raw["GLUCOSE"]
    if "BP" in raw:
        try:
            sbp, dbp = map(int, raw["BP"].split('/'))
            model_input["systolic_bp"] = sbp
            model_input["diastolic_bp"] = dbp
        except:
            model_input["systolic_bp"], model_input["diastolic_bp"] = np.nan, np.nan
    # For missing features, set NaN
    for col in models.get("wearable_columns", []):
        if col not in model_input:
            model_input[col] = np.nan
    return model_input

# --- 5. DATABASE HELPER FUNCTIONS ---
def get_current_patient_id_from_token():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '): return None
        session_token = auth_header.split(' ')[1]
        session = UserSession.query.filter_by(session_token=session_token).filter(UserSession.expires_at > datetime.now(timezone.utc)
).first()
        return session.patient_id if session else None
    except Exception: return None

def save_user_activity(patient_id, activity_type, page_visited, details=None):
    try:
        activity = UserActivityLog(patient_id=patient_id, activity_type=activity_type, page_visited=page_visited, details=details)
        db.session.add(activity)
        db.session.commit()
    except Exception as e:
        print(f"DB Activity Save Error: {e}")
        db.session.rollback()

def save_to_diabetes_checks(patient_id, input_data, output_data):
    try:
        print(f"Attempting to save diabetes check for patient_id: {patient_id}")
        
        # Use the correct column name (user_id instead of patient_id)
        sql = """
        INSERT INTO diabetes_checks 
        (user_id, is_pregnant, age_at_diagnosis, bmi_at_diagnosis, family_history, 
         hba1c, c_peptide_level, autoantibodies_status, genetic_test_result, report_json, created_at)
        VALUES (:user_id, :is_pregnant, :age_at_diagnosis, :bmi_at_diagnosis, :family_history, 
                :hba1c, :c_peptide_level, :autoantibodies_status, :genetic_test_result, :report_json, :created_at)
        """
        
        # Use named parameters and pass as a dictionary
        params = {
            'user_id': patient_id,  # Changed from patient_id to user_id
            'is_pregnant': bool(input_data.get('Is_Pregnant')),
            'age_at_diagnosis': input_data.get('Age_at_Diagnosis'),
            'bmi_at_diagnosis': input_data.get('BMI_at_Diagnosis'),
            'family_history': input_data.get('Family_History'),
            'hba1c': input_data.get('HbA1c'),
            'c_peptide_level': input_data.get('C_Peptide_Level'),
            'autoantibodies_status': input_data.get('Autoantibodies_Status'),
            'genetic_test_result': input_data.get('Genetic_Test_Result'),
            'report_json': json.dumps(output_data),
            'created_at': datetime.now(timezone.utc)

        }
        
        # Execute with named parameters
        result = db.session.execute(text(sql), params)
        db.session.commit()
        
        print(f"Save successful, affected rows: {result.rowcount}")
        return True
        
    except Exception as e:
        print(f"Error saving to diabetes_checks: {e}")
        import traceback
        traceback.print_exc()
        db.session.rollback()
        return False

def save_to_hypertension_checks(patient_id, input_data, output_data):
    try:
        print(f"Attempting to save hypertension check for patient_id: {patient_id}")
        
        # Use the correct column name (user_id instead of patient_id)
        sql = """
        INSERT INTO hypertension_checks 
        (user_id, age, sex, bmi, family_history, creatinine, systolic_bp, diastolic_bp, report_json, created_at)
        VALUES (:user_id, :age, :sex, :bmi, :family_history, :creatinine, :systolic_bp, :diastolic_bp, :report_json, :created_at)
        """
        
        params = {
            'user_id': patient_id,  # Changed from patient_id to user_id
            'age': input_data.get('age'),
            'sex': input_data.get('sex'),
            'bmi': input_data.get('bmi'),
            'family_history': bool(input_data.get('family_history')),
            'creatinine': input_data.get('creatinine'),
            'systolic_bp': input_data.get('systolic_bp'),
            'diastolic_bp': input_data.get('diastolic_bp'),
            'report_json': json.dumps(output_data),
            'created_at': datetime.now(timezone.utc)

        }
        
        result = db.session.execute(text(sql), params)
        db.session.commit()
        
        print(f"Save successful, affected rows: {result.rowcount}")
        return True
        
    except Exception as e:
        print(f"Error saving to hypertension_checks: {e}")
        import traceback
        traceback.print_exc()
        db.session.rollback()
        return False
def save_prediction_to_db(patient_id, prediction_type, page_source, input_data, output_data):
    try:
        prediction = UserPrediction(patient_id=patient_id, prediction_type=prediction_type, page_source=page_source, input_data=input_data, output_data=output_data)
        db.session.add(prediction)
        db.session.commit()
    except Exception as e:
        print(f"DB Prediction Save Error: {e}")
        db.session.rollback()
        
def get_case_name(label):
    """Convert label to descriptive case name"""
    case_names = {
        "Diabetes_Mild": "Case 1 - Diabetes Mild",
        "Diabetes_Severe": "Case 2 - Diabetes Severe",
        "Diabetes_Hypo": "Case 3 - Diabetes Hypo",
        "Hypertension_Severe": "Case 4 - Hypertension Severe",
        "Hypertension_Moderate": "Case 5 - Hypertension Moderate",
        "Hypotension": "Case 6 - Hypotension",
        "Both_Diabetes_HTN": "Case 7 - Both Diabetes+HTN",
        "Both_Moderate": "Case 8 - Both Moderate",
        "Healthy": "Case 9 - Healthy"
    }
    return case_names.get(label, "Unknown Case")

def get_current_user_id_from_token():
    try:
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
            
        session_token = auth_header.split(' ')[1]
        
        session = UserSession.query.filter_by(session_token=session_token).filter(
            UserSession.expires_at > datetime.now(timezone.utc)
).first()
        
        if session:
            # Return patient_id but note that the database expects user_id
            # This will work if patient_id and user_id are the same values
            return session.patient_id
        else:
            return None
            
    except Exception as e:
        print(f"Error in get_current_user_id_from_token: {e}")
        return None
# --- 6. API ENDPOINTS ---
# --- 6. API ENDPOINTS ---

from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        patient_id = get_current_patient_id_from_token()
        if not patient_id:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function
def get_patient_health_context(patient_id):
    """Compile comprehensive health context for a patient"""
    context_parts = []
    
    # Get patient basic info
    patient = Patient.query.get(patient_id)
    if patient:
        context_parts.append(f"Patient: {patient.name}")
    
    # Get latest chronic disease prediction
    chronic_prediction = UserPrediction.query.filter_by(
        patient_id=patient_id, 
        prediction_type='chronic_disease'
    ).order_by(UserPrediction.created_at.desc()).first()
    
    if chronic_prediction and chronic_prediction.output_data:
        try:
            data = chronic_prediction.output_data
            if 'predicted_conditions' in data and data['predicted_conditions']:
                conditions = [cond['disease'] for cond in data['predicted_conditions']]
                context_parts.append(f"Chronic conditions: {', '.join(conditions)}")
            
            if 'risk_assessment' in data and data['risk_assessment']:
                for risk in data['risk_assessment']:
                    context_parts.append(f"{risk['disease']} risk: {risk['risk_level']} ({risk['risk_score']}%)")
        except Exception as e:
            print(f"Error parsing chronic prediction: {e}")
    
    # Get latest diabetes prediction
    diabetes_check = DiabetesCheck.query.filter_by(
        user_id=patient_id
    ).order_by(DiabetesCheck.created_at.desc()).first()
    
    if diabetes_check and diabetes_check.report_json:
        try:
            report = json.loads(diabetes_check.report_json)
            context_parts.append(f"Diabetes type: {report.get('predicted_type', 'Unknown')}")
            context_parts.append(f"Diabetes risk: {report.get('risk_level', 'Unknown')}")
        except Exception as e:
            print(f"Error parsing diabetes report: {e}")
    
    # Get latest hypertension prediction
    hypertension_check = HypertensionCheck.query.filter_by(
        user_id=patient_id
    ).order_by(HypertensionCheck.created_at.desc()).first()
    
    if hypertension_check and hypertension_check.report_json:
        try:
            report = json.loads(hypertension_check.report_json)
            if report.get('hypertension_risk', False):
                context_parts.append(f"Hypertension: {report.get('stage', 'Unknown')} stage")
                context_parts.append(f"Hypertension subtype: {report.get('subtype', 'Unknown')}")
                context_parts.append(f"Hypertension risk level: {report.get('risk_level', 'Unknown')}")
        except Exception as e:
            print(f"Error parsing hypertension report: {e}")
    
    # Get input data from latest prediction for detailed health metrics
    if chronic_prediction and chronic_prediction.input_data:
        try:
            input_data = chronic_prediction.input_data
            # Add relevant health metrics
            metrics = []
            if 'Age' in input_data: metrics.append(f"Age: {input_data['Age']}")
            if 'BMI' in input_data: metrics.append(f"BMI: {input_data['BMI']}")
            if 'HbA1c' in input_data: metrics.append(f"HbA1c: {input_data['HbA1c']}%")
            if 'Systolic_BP' in input_data: metrics.append(f"BP: {input_data.get('Systolic_BP', 'N/A')}/{input_data.get('Diastolic_BP', 'N/A')}")
            if 'LDL_Cholesterol' in input_data: metrics.append(f"LDL: {input_data['LDL_Cholesterol']} mg/dL")
            
            if metrics:
                context_parts.append("Health metrics: " + ", ".join(metrics))
        except Exception as e:
            print(f"Error parsing input data: {e}")
    
    # If no health data found, provide a generic context
    if not context_parts:
        context_parts.append("No detailed health history available. Patient may be new or hasn't completed health assessments yet.")
    
    return ". ".join(context_parts)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Python server is running!"})

# Updated signup route to match frontend
@app.route('/signup', methods=['POST'])
def patient_signup():
    data = request.get_json()
    name, email, password = data.get('name'), data.get('email'), data.get('password')
    if not all([name, email, password]): 
        return jsonify({"message": "Missing required fields"}), 400
    try:
        if Patient.query.filter_by(email=email).first():
            return jsonify({"message": "An account with this email already exists."}), 409
        
        pw_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        patient = Patient(name=name, email=email, password_hash=pw_hash)
        db.session.add(patient)
        db.session.flush()
        
        session_token = bcrypt.generate_password_hash(f"{email}{datetime.now(timezone.utc)
}").decode('utf-8').replace('/', '').replace('.', '')
        session = UserSession(patient_id=patient.id, session_token=session_token, 
                             expires_at=datetime.now(timezone.utc)
 + timedelta(days=7))
        db.session.add(session)
        db.session.commit()
        
        save_user_activity(patient.id, "signup", "signup", {"email": email})
        return jsonify({
            "message": "Patient account created successfully", 
            "patient_id": patient.id, 
            "session_token": session_token, 
            "name": name
        }), 201
        
    except Exception as err:
        db.session.rollback()
        print(f"Signup error: {err}")  # Add detailed error logging
        return jsonify({"message": f"Database error: {err}"}), 500

# Updated login route to match frontend
@app.route('/login', methods=['POST'])
def patient_login():
    data = request.get_json()
    email, password = data.get('email'), data.get('password')
    if not email or not password: 
        return jsonify({"message": "Missing email or password"}), 400
    
    try:
        patient = Patient.query.filter_by(email=email).first()
        if patient and bcrypt.check_password_hash(patient.password_hash, password):
            session_token = bcrypt.generate_password_hash(f"{email}{datetime.now(timezone.utc)
}").decode('utf-8').replace('/', '').replace('.', '')
            session = UserSession(patient_id=patient.id, session_token=session_token, 
                                 expires_at=datetime.now(timezone.utc)
 + timedelta(days=7))
            db.session.add(session)
            db.session.commit()
            
            save_user_activity(patient.id, "login", "login", {"email": email})
            return jsonify({
                "message": "Login successful", 
                "patient_id": patient.id, 
                "session_token": session_token, 
                "name": patient.name
            }), 200
        else:
            return jsonify({"message": "Invalid email or password"}), 401
            
    except Exception as err:
        print(f"Login error: {err}")  # Add detailed error logging
        db.session.rollback()
        return jsonify({"message": f"Database error: {err}"}), 500  

@app.route('/debug/db', methods=['GET'])
def debug_db():
    try:
        # Test database connection
        result = db.session.execute(db.text('SELECT 1')).scalar()
        patient_count = Patient.query.count()
        return jsonify({
            "db_connection": "success",
            "test_query": result,
            "patient_count": patient_count
        })
    except Exception as e:
        return jsonify({"db_connection": "failed", "error": str(e)}), 500  
    

@app.route('/generate_report', methods=['POST'])
@login_required
def generate_report():
    if 'chronic_classifier' not in models:
        return jsonify({"error": "Chronic disease models are not loaded."}), 500
    
    patient_input_data = request.get_json()
    if not patient_input_data:
        return jsonify({"error": "No input data provided."}), 400
    
    try:
        patient_id = get_current_patient_id_from_token()
        input_df = pd.DataFrame([patient_input_data])
        final_report = {"predicted_conditions": [], "risk_assessment": []}
        
        # Your existing prediction logic
        predicted_list = models['chronic_mlb'].inverse_transform(models['chronic_classifier'].predict(input_df))[0]
        if not predicted_list:
            final_report["predicted_conditions"].append({"disease": "Healthy", "explanation": "The model predicts the patient is Healthy."})
        else:
            for disease in predicted_list:
                explanation = get_disease_explanation(patient_input_data, disease)
                final_report["predicted_conditions"].append({"disease": disease, "explanation": f"Detected based on {explanation}."})
                if disease in models['chronic_regressors']:
                    reg_model = models['chronic_regressors'][disease]
                    risk_score = max(0, min(100, reg_model.predict(input_df)[0]))
                    final_report["risk_assessment"].append({
                        "disease": disease, 
                        "risk_score": round(risk_score, 1), 
                        "risk_level": get_risk_level(risk_score), 
                        "primary_drivers": f"This risk level is driven by {explanation}."
                    })
        
        # Save to user_predictions table using your existing function
        if patient_id:
            save_prediction_to_db(patient_id, 'chronic_disease', 'generate-report', patient_input_data, final_report)
            save_user_activity(patient_id, 'prediction', 'generate-report', {
                'conditions': list(predicted_list) if predicted_list else ['Healthy']
            })
        
        return jsonify(final_report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/predict_diabetes_subtype', methods=['POST'])
def predict_diabetes_subtype():
    if 'diabetes_subtype_classifier' not in models:
        return jsonify({"error": "Diabetes subtype models are not loaded."}), 500
    try:
        data = request.get_json()
        print(f"Received diabetes prediction request: {data}")
        
        user_id = get_current_user_id_from_token()
        print(f"Extracted user_id: {user_id}")
        
        # Your existing prediction logic
        input_data = pd.DataFrame([data])
        prediction = models['diabetes_subtype_classifier'].predict(input_data)[0]
        prediction_proba = models['diabetes_subtype_classifier'].predict_proba(input_data)[0]
        processed_data = models['diabetes_preprocessor'].transform(input_data)
        risk_cluster = models['diabetes_risk_model'].predict(processed_data)[0]
        risk_level = models['diabetes_cluster_map'].get(risk_cluster, 'Unknown')
        
        explanation = f"The model predicts {prediction} with {max(prediction_proba)*100:.1f}% confidence. Risk assessment: {risk_level} risk level."
        response = {
            'predicted_type': str(prediction), 
            'confidence_score': float(max(prediction_proba)) * 100, 
            'risk_level': risk_level, 
            'explanation': explanation
        }
        
        print(f"Prediction result: {response}")
        
        # Save to diabetes_checks table
        if user_id:
            print("Attempting to save to database...")
            success = save_to_diabetes_checks(user_id, data, response)
            if success:
                print("Database save successful!")
                save_user_activity(user_id, 'prediction', 'diabetes-check', {
                    'type': 'diabetes', 
                    'predicted_type': str(prediction)
                })
            else:
                print("Database save failed!")
        else:
            print("No user_id, skipping database save")
        
        return jsonify(response)
    except Exception as e:
        print(f"Error in predict_diabetes_subtype: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500        
@app.route('/predict_hypertension', methods=['POST'])
def predict_hypertension():
    try:
        data = request.get_json()
        patient_id = get_current_patient_id_from_token()
        
        # Extract all required fields with defaults
        age = data.get('age', 50)
        sex = data.get('sex', 1)  # Default to Male
        bmi = data.get('bmi', 25.0)
        family_history = data.get('family_history', False)
        creatinine = data.get('creatinine', 1.0)
        systolic_bp = data.get('systolic_bp', 120)
        diastolic_bp = data.get('diastolic_bp', 80)
        
        # Your existing prediction logic
        stage, stage_description = determine_hypertension_stage(systolic_bp, diastolic_bp)
        subtype = determine_hypertension_subtype(data)
        kidney_risk, stroke_risk, heart_risk = calculate_hypertension_risks(systolic_bp, diastolic_bp, age, creatinine)
        hypertension_risk = stage not in ["Normal", "Elevated"]
        probability = min(0.95, (kidney_risk + stroke_risk + heart_risk) / 300)
        risk_level = 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
        explanation = generate_hypertension_explanation(hypertension_risk, data, probability, stage_description, subtype)
        
        response = {
            'hypertension_risk': hypertension_risk, 
            'probability': probability, 
            'risk_level': risk_level, 
            'stage': stage, 
            'subtype': subtype, 
            'kidney_risk_1yr': round(kidney_risk, 2), 
            'stroke_risk_1yr': round(stroke_risk, 2), 
            'heart_risk_1yr': round(heart_risk, 2), 
            'explanation': explanation
        }
        
        # Save to hypertension_checks table
        if patient_id:
            save_to_hypertension_checks(patient_id, data, response)
            save_user_activity(patient_id, 'prediction', 'hypertension-check', {
                'type': 'hypertension', 
                'risk_level': risk_level, 
                'stage': stage
            })
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/generate_care_plan', methods=['POST'])
def generate_care_plan_endpoint():
    patient_id = get_current_patient_id_from_token()
    if not patient_id:
        return jsonify({"error": "Not authenticated"}), 401
    try:
        latest_report = UserPrediction.query.filter_by(patient_id=patient_id).order_by(UserPrediction.created_at.desc()).first()
        if not latest_report:
            return jsonify({'error': 'No reports found for this user. Please submit a health entry first.'}), 404
        
        summary_text = f"PATIENT HEALTH SUMMARY:\n\n{json.dumps(latest_report.output_data, indent=2)}"
        care_plan_text = generate_care_plan_from_report(summary_text)

        # ✅ split into structured JSON
        days_dict = split_days(care_plan_text)

        new_plan = CarePlan(
            patient_id=patient_id,
            source_data_summary=summary_text,
            generated_plan=care_plan_text
        )
        db.session.add(new_plan)
        db.session.commit()
        
        save_user_activity(patient_id, 'care_plan_generated', 'care-plan')

        return jsonify({
            "care_plan": care_plan_text,
            "structured_plan": days_dict   # <-- added structured plan for frontend
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to generate care plan: {str(e)}"}), 500    
@app.route('/care_plan/interaction', methods=['POST'])
@login_required
def record_care_plan_interaction():
    """Record user interactions with the care plan"""
    try:
        patient_id = get_current_patient_id_from_token()
        data = request.get_json()
        
        interaction = UserCarePlanInteraction(
            patient_id=patient_id,
            day_number=data.get('day_number'),
            question=data.get('question'),
            response=data.get('response'),
            care_plan_text=data.get('care_plan_text')
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        save_user_activity(patient_id, 'care_plan_interaction', 'care-plan', {
            'day': data.get('day_number'),
            'question': data.get('question')
        })
        
        return jsonify({"message": "Interaction recorded successfully"}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/care_plan/progress', methods=['GET', 'POST'])
@login_required
def care_plan_progress():
    """Get or update care plan progress"""
    try:
        patient_id = get_current_patient_id_from_token()
        
        if request.method == 'GET':
            # Get current progress
            progress = CarePlanProgress.query.filter_by(patient_id=patient_id).first()
            
            if not progress:
                return jsonify({
                    "current_day": 1,
                    "last_completed_day": 0,
                    "is_completed": False
                }), 200
            
            return jsonify({
                "current_day": progress.current_day,
                "last_completed_day": progress.last_completed_day,
                "is_completed": progress.is_completed,
                "last_accessed": progress.last_accessed.isoformat()
            }), 200
            
        else:
            # Update progress
            data = request.get_json()
            progress = CarePlanProgress.query.filter_by(patient_id=patient_id).first()
            
            if not progress:
                progress = CarePlanProgress(patient_id=patient_id)
                db.session.add(progress)
            
            if 'current_day' in data:
                progress.current_day = data['current_day']
            if 'last_completed_day' in data:
                progress.last_completed_day = data['last_completed_day']
            if 'is_completed' in data:
                progress.is_completed = data['is_completed']
            
            progress.last_accessed = datetime.now(timezone.utc)

            db.session.commit()
            
            return jsonify({"message": "Progress updated successfully"}), 200
            
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/care_plan/can_access_day', methods=['POST'])
@login_required
def can_access_care_plan_day():
    """Check if user can access the requested day"""
    try:
        patient_id = get_current_patient_id_from_token()
        data = request.get_json()
        requested_day = data.get('day', 1)
        
        progress = CarePlanProgress.query.filter_by(patient_id=patient_id).first()
        
        # If no progress record, allow day 1
        if not progress:
            return jsonify({"can_access": requested_day == 1, "current_day": 1}), 200
        
        # Check if requested day is valid
        can_access = (
            requested_day <= progress.current_day and 
            requested_day <= progress.last_completed_day + 1
        )
        
        return jsonify({
            "can_access": can_access,
            "current_day": progress.current_day,
            "last_completed_day": progress.last_completed_day
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/care_plan/notify', methods=['POST'])
@login_required
def send_care_plan_notification():
    """Send notification for current day's care plan"""
    try:
        patient_id = get_current_patient_id_from_token()
        data = request.get_json()
        day_number = data.get('day_number', 1)
        care_plan_text = data.get('care_plan_text', '')
        
        # Here you would integrate with your notification service
        # For now, we'll just log it
        print(f"Notification sent for day {day_number} to patient {patient_id}")
        
        save_user_activity(patient_id, 'care_plan_notification', 'care-plan', {
            'day': day_number,
            'notification_sent': True
        })
        
        return jsonify({"message": "Notification processed"}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/care_plan/get_day', methods=['POST'])
@login_required
def get_care_plan_day():
    """Get a specific day's care plan content"""
    try:
        patient_id = get_current_patient_id_from_token()
        data = request.get_json()
        day_number = data.get('day_number', 1)
        
        # Get the latest care plan for this patient
        care_plan = CarePlan.query.filter_by(patient_id=patient_id).order_by(CarePlan.created_at.desc()).first()
        
        if not care_plan:
            return jsonify({"error": "No care plan found for this patient"}), 404
        
        # Split the care plan into days
        days_dict = split_days(care_plan.generated_plan)
        
        # Get the requested day
        day_key = f"Day {day_number}"
        if day_key not in days_dict:
            return jsonify({"error": f"Day {day_number} not found in care plan"}), 404
        
        # Check if user can access this day
        progress = CarePlanProgress.query.filter_by(patient_id=patient_id).first()
        if not progress:
            # First time accessing, create progress record
            progress = CarePlanProgress(patient_id=patient_id)
            db.session.add(progress)
            db.session.commit()
        
        # Update current day if accessing a new day
        if day_number > progress.current_day:
            progress.current_day = day_number
            progress.last_accessed = datetime.now(timezone.utc)
            db.session.commit()
        
        return jsonify({
            "day_number": day_number,
            "content": days_dict[day_key],
            "current_day": progress.current_day,
            "last_completed_day": progress.last_completed_day,
            "is_completed": progress.is_completed
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/care_plan/complete_day', methods=['POST'])
@login_required
def complete_care_plan_day():
    """Mark a day as completed"""
    try:
        patient_id = get_current_patient_id_from_token()
        data = request.get_json()
        day_number = data.get('day_number', 1)
        
        progress = CarePlanProgress.query.filter_by(patient_id=patient_id).first()
        
        if not progress:
            progress = CarePlanProgress(patient_id=patient_id)
            db.session.add(progress)
        
        # Update last completed day
        if day_number > progress.last_completed_day:
            progress.last_completed_day = day_number
        
        # Check if all days are completed
        if progress.last_completed_day >= 7:
            progress.is_completed = True
        
        progress.last_accessed = datetime.now(timezone.utc)
        db.session.commit()
        
        # Record the interaction
        care_plan = CarePlan.query.filter_by(patient_id=patient_id).order_by(CarePlan.created_at.desc()).first()
        day_content = ""
        if care_plan:
            days_dict = split_days(care_plan.generated_plan)
            day_key = f"Day {day_number}"
            day_content = days_dict.get(day_key, "")
        
        interaction = UserCarePlanInteraction(
            patient_id=patient_id,
            day_number=day_number,
            question="completed_day",
            response="yes",
            care_plan_text=day_content
        )
        db.session.add(interaction)
        db.session.commit()
        
        save_user_activity(patient_id, 'care_plan_day_completed', 'care-plan', {
            'day': day_number,
            'completed': True
        })
        
        return jsonify({
            "message": f"Day {day_number} marked as completed",
            "last_completed_day": progress.last_completed_day,
            "is_completed": progress.is_completed
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/live_monitor', methods=['POST'])
def live_monitor():
    if 'wearable_model' not in models:
        return jsonify({"error": "Wearable model is not loaded on the server."}), 500
    
    sample_data = [
        {'HR': 79, 'TEMPERATURE': 98.6, 'SpO2': 95, 'GLUCOSE': 101, 'RESPIRATION': 21, 'BP': '120/80'},
        {'HR': 62, 'TEMPERATURE': 98.6, 'SpO2': 98, 'GLUCOSE': 95, 'RESPIRATION': 17, 'BP': '110/70'},
        {'HR': 95, 'TEMPERATURE': 99.5, 'SpO2': 93, 'GLUCOSE': 140, 'RESPIRATION': 24, 'BP': '130/85'},
        {'HR': 110, 'TEMPERATURE': 100.2, 'SpO2': 91, 'GLUCOSE': 160, 'RESPIRATION': 28, 'BP': '140/90'},
        {'HR': 85, 'TEMPERATURE': 98.2, 'SpO2': 96, 'GLUCOSE': 110, 'RESPIRATION': 22, 'BP': '125/82'},
        {'HR': 70, 'TEMPERATURE': 98.8, 'SpO2': 97, 'GLUCOSE': 105, 'RESPIRATION': 19, 'BP': '118/78'},
    ]
    try:
        selected_vitals = random.choice(sample_data)
        model_input = normalize_vitals(selected_vitals)

        df = pd.DataFrame([model_input], columns=models['wearable_columns'])
        scaled_data = models['wearable_scaler'].transform(df)
        prediction = models['wearable_model'].predict(scaled_data)[0]
        prob = models['wearable_model'].predict_proba(scaled_data).max()

        if prob >= 0.7: severity = "high"
        elif prob >= 0.4: severity = "medium"
        else: severity = "low"

        remedies = generate_rule_based_remedy(model_input)

        return jsonify({
            "input_vitals": selected_vitals,
            "normalized_input": model_input,
            "prediction": str(prediction),
            "severity": severity,
            "remedies": remedies
        })
    except Exception as e:
        print(f"!!! ERROR in /live_monitor: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Prediction error: {e}"}), 500    
@app.route('/wearable/connect', methods=['GET'])
def wearable_connect():
    try:
        # Updated test cases with proper labels and realistic values
        test_cases = [
            # Case 1 - Diabetes Mild
            {"glucose_level": 190, "heart_rate": 85, "steps_per_min": 50, "sleep_hours": 6, "stress_level": 4, "systolic_bp": 120, "diastolic_bp": 80, "label": "Diabetes_Mild"},
            # Case 2 - Diabetes Severe
            {"glucose_level": 320, "heart_rate": 88, "steps_per_min": 40, "sleep_hours": 5, "stress_level": 6, "systolic_bp": 118, "diastolic_bp": 78, "label": "Diabetes_Severe"},
            # Case 3 - Diabetes Hypo
            {"glucose_level": 65, "heart_rate": 110, "steps_per_min": 10, "sleep_hours": 4, "stress_level": 7, "systolic_bp": 100, "diastolic_bp": 65, "label": "Diabetes_Hypo"},
            # Case 4 - Hypertension Severe
            {"glucose_level": 110, "heart_rate": 80, "steps_per_min": 45, "sleep_hours": 6, "stress_level": 7, "systolic_bp": 180, "diastolic_bp": 120, "label": "Hypertension_Severe"},
            # Case 5 - Hypertension Moderate
            {"glucose_level": 115, "heart_rate": 75, "steps_per_min": 60, "sleep_hours": 7, "stress_level": 5, "systolic_bp": 160, "diastolic_bp": 100, "label": "Hypertension_Moderate"},
            # Case 6 - Hypotension
            {"glucose_level": 100, "heart_rate": 72, "steps_per_min": 55, "sleep_hours": 8, "stress_level": 3, "systolic_bp": 85, "diastolic_bp": 55, "label": "Hypotension"},
            # Case 7 - Both Diabetes+HTN
            {"glucose_level": 280, "heart_rate": 95, "steps_per_min": 35, "sleep_hours": 6, "stress_level": 8, "systolic_bp": 165, "diastolic_bp": 100, "label": "Both_Diabetes_HTN"},
            # Case 8 - Both Moderate
            {"glucose_level": 220, "heart_rate": 105, "steps_per_min": 25, "sleep_hours": 5, "stress_level": 7, "systolic_bp": 160, "diastolic_bp": 95, "label": "Both_Moderate"},
            # Case 9 - Healthy
            {"glucose_level": 105, "heart_rate": 75, "steps_per_min": 65, "sleep_hours": 7, "stress_level": 3, "systolic_bp": 120, "diastolic_bp": 80, "label": "Healthy"}
        ]

        random_case = random.choice(test_cases)
        
        # Create DataFrame with all columns including label
        row_df = pd.DataFrame([random_case], columns=models['wearable_columns'])
        
        # For prediction, we need to exclude the label column as it's the target
        input_columns = [col for col in models['wearable_columns'] if col != 'label']
        input_data = {col: random_case[col] for col in input_columns}
        
        # Transform and predict using only input features (excluding label)
        input_df = pd.DataFrame([input_data], columns=input_columns)
        row_scaled = models['wearable_scaler'].transform(input_df)

        prediction = models['wearable_model'].predict(row_scaled)[0]
        prob = models['wearable_model'].predict_proba(row_scaled).max()

        if prob >= 0.7: severity = "high"
        elif prob >= 0.4: severity = "medium"
        else: severity = "low"

        remedies = generate_rule_based_remedy(input_data)

        return jsonify({
            "input_data": random_case,  # Include label in the response
            "prediction": str(prediction),
            "severity": severity,
            "remedies": remedies,
            "actual_label": random_case['label'],  # Also show the actual label for reference
            "case_name": get_case_name(random_case['label'])  # Add descriptive case name
        })
    except Exception as e:
        print(f"!!! ERROR in /wearable/connect: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
# --- COMPLETELY REWRITTEN ENDPOINT ---
@app.route('/get_live_update', methods=['GET'])
def get_live_update():
    if 'wearable_model' not in models:
        return jsonify({"error": "Wearable model is not loaded."}), 500
    
    patients = {
        "Case 1 - Hypoglycemia": {"glucose_level": 60, "heart_rate": 85, "steps_per_min": 20, "sleep_hours": 7, "stress_level": 4, "systolic_bp": 120, "diastolic_bp": 80},
        "Case 2 - Hyperglycemia": {"glucose_level": 300, "heart_rate": 88, "steps_per_min": 15, "sleep_hours": 6, "stress_level": 5, "systolic_bp": 118, "diastolic_bp": 78},
        "Case 3 - Hypertension High": {"glucose_level": 120, "heart_rate": 90, "steps_per_min": 30, "sleep_hours": 6, "stress_level": 6, "systolic_bp": 170, "diastolic_bp": 105},
        "Case 4 - Hypotension": {"glucose_level": 110, "heart_rate": 80, "steps_per_min": 40, "sleep_hours": 7, "stress_level": 3, "systolic_bp": 95, "diastolic_bp": 55},
        "Case 5 - Tachycardia": {"glucose_level": 130, "heart_rate": 130, "steps_per_min": 10, "sleep_hours": 7, "stress_level": 5, "systolic_bp": 115, "diastolic_bp": 75},
        "Case 6 - High Stress": {"glucose_level": 125, "heart_rate": 85, "steps_per_min": 25, "sleep_hours": 7, "stress_level": 9, "systolic_bp": 118, "diastolic_bp": 78},
        "Case 7 - Sleep Deprived": {"glucose_level": 115, "heart_rate": 78, "steps_per_min": 20, "sleep_hours": 3, "stress_level": 4, "systolic_bp": 110, "diastolic_bp": 70},
        "Case 8 - Healthy": {"glucose_level": 110, "heart_rate": 75, "steps_per_min": 40, "sleep_hours": 7, "stress_level": 3, "systolic_bp": 120, "diastolic_bp": 80}
    }

    try:
        case_name, vitals_dict = random.choice(list(patients.items()))
        row_df = pd.DataFrame([vitals_dict])
        row_df = row_df[models['wearable_columns']]
        row_scaled = models['wearable_scaler'].transform(row_df)
        pred = models['wearable_model'].predict(row_scaled)[0]
        prob = models['wearable_model'].predict_proba(row_scaled).max()

        if prob >= 0.7: severity = "high"
        elif prob >= 0.4: severity = "medium"
        else: severity = "low"
        
        remedies = generate_remedies(row_df, pred, severity)

        response_data = {
            "caseName": case_name, "vitals": vitals_dict,
            "analysis": { "Disease": pred, "Severity": severity, "Remedies": remedies }
        }
        return jsonify(response_data)
    except Exception as e:
        # This new print statement will give us a very clear error in the terminal
        print(f"!!! CRITICAL ERROR in /get_live_update: {str(e)} !!!")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500
    
@app.route('/chatbot', methods=['POST'])
@login_required
def chatbot_endpoint():
    """Chat with the medical AI assistant"""
    try:
        patient_id = get_current_patient_id_from_token()
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "Message is required"}), 400
        
        user_message = data['message']
        
        # Get comprehensive patient health data for context
        health_context = get_patient_health_context(patient_id)
        
        # Add health context to the user message
        enhanced_message = f"PATIENT HEALTH CONTEXT: {health_context}. USER QUESTION: {user_message}"
        
        # Get response from chatbot
        if not chatbot_chain:
            return jsonify({"error": "Chatbot is not available at the moment"}), 503
            
        try:
            bot_response = chatbot_chain.invoke({"user_input": enhanced_message})
        except Exception as e:
            print(f"Chatbot API error: {e}")
            return jsonify({"error": "Chatbot service is temporarily unavailable"}), 503
        
        # Try to save the interaction to database
        try:
            interaction = UserChatInteraction(
                patient_id=patient_id,
                user_message=user_message,
                bot_response=bot_response,
                health_context=health_context
            )
            db.session.add(interaction)
            db.session.commit()
        except Exception as e:
            print(f"Note: Chat interaction not saved to database: {e}")
            # Don't fail the request if saving fails
        
        save_user_activity(patient_id, 'chatbot_interaction', 'chatbot', {
            'message_length': len(user_message)
        })
        
        return jsonify({"response": bot_response})
        
    except Exception as e:
        print(f"Chatbot endpoint error: {e}")
        return jsonify({"error": "Internal server error"}), 500
        
# --- (The rest of your file is unchanged) ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)