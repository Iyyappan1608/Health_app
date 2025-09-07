import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os
from datetime import datetime, timedelta
import time
import xgboost as xgb
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)

# Define hypertension stages and subtypes
HYPERTENSION_STAGES = {
    'Normal': {'systolic_range': (90, 120), 'diastolic_range': (60, 80)},
    'Elevated': {'systolic_range': (120, 129), 'diastolic_range': (60, 80)},
    'Stage_1': {'systolic_range': (130, 139), 'diastolic_range': (80, 89)},
    'Stage_2': {'systolic_range': (140, 179), 'diastolic_range': (90, 119)},
    'Crisis': {'systolic_range': (180, 300), 'diastolic_range': (120, 300)}
}

HYPERTENSION_SUBTYPES = [
    'Primary',
    'Secondary_Endocrine', 
    'Secondary_Renal',
    'Secondary_Renovascular',
    'Secondary_Sleep_Apnea',
    'White_Coat',
    'Masked'
]

# Define risk metadata
RISK_METADATA = {
    'kidney_risk_1yr': {
        'condition': 'kidney function decline',
        'timeframe': '1 year',
        'condition_category': 'renal',
        'severity_thresholds': {'low': 5, 'moderate': 15, 'high': 30, 'critical': 50}
    },
    'kidney_risk_2yr': {
        'condition': 'kidney function decline',
        'timeframe': '2 years',
        'condition_category': 'renal',
        'severity_thresholds': {'low': 8, 'moderate': 20, 'high': 35, 'critical': 55}
    },
    'stroke_risk_1yr': {
        'condition': 'stroke',
        'timeframe': '1 year',
        'condition_category': 'neurological',
        'severity_thresholds': {'low': 3, 'moderate': 10, 'high': 20, 'critical': 40}
    },
    'heart_risk_1yr': {
        'condition': 'heart attack',
        'timeframe': '1 year',
        'condition_category': 'cardiovascular',
        'severity_thresholds': {'low': 4, 'moderate': 12, 'high': 25, 'critical': 45}
    },
    'vision_risk_2yr': {
        'condition': 'vision loss',
        'timeframe': '2 years',
        'condition_category': 'ophthalmic',
        'severity_thresholds': {'low': 2, 'moderate': 8, 'high': 18, 'critical': 35}
    }
}

# Define statement templates
STATEMENT_TEMPLATES = {
    'default': "{risk_value}% risk of {condition} in {timeframe} if BP not controlled",
    'personalized': "Based on your profile, you have a {severity_level} risk ({risk_value}%) of {condition} within {timeframe}",
    'clinical': "Patient exhibits {risk_value}% probability of {condition} development in {timeframe} without intervention",
    'urgent': "⚠ URGENT: {risk_value}% risk of {condition} in {timeframe} - immediate action required!"
}

def get_severity_level(risk_value, risk_type):
    """Determine severity level based on risk value and type"""
    thresholds = RISK_METADATA[risk_type]['severity_thresholds']
    
    if risk_value >= thresholds['critical']:
        return 'critical'
    elif risk_value >= thresholds['high']:
        return 'high'
    elif risk_value >= thresholds['moderate']:
        return 'moderate'
    elif risk_value >= thresholds['low']:
        return 'low'
    else:
        return 'minimal'

def generate_risk_statements(risk_predictions, template_type='default'):
    """Generate natural language risk statements"""
    statements = []
    
    for risk_type, risk_value in risk_predictions.items():
        if risk_type in RISK_METADATA and risk_value > RISK_METADATA[risk_type]['severity_thresholds']['low']:
            metadata = RISK_METADATA[risk_type]
            severity_level = get_severity_level(risk_value, risk_type)
            
            if severity_level == 'critical' and template_type != 'clinical':
                template = STATEMENT_TEMPLATES['urgent']
            elif template_type == 'clinical':
                template = STATEMENT_TEMPLATES['clinical']
            elif template_type == 'personalized':
                template = STATEMENT_TEMPLATES['personalized']
            else:
                template = STATEMENT_TEMPLATES['default']
            
            statement = template.format(
        risk_value=f"{risk_value:.2f}",
        condition=metadata['condition'],
        timeframe=metadata['timeframe'],
        severity_level=severity_level
        )
            statements.append(statement)
    
    return statements

class HypertensionPipeline:
    """Complete hypertension staging and risk prediction pipeline"""
    
    def _init_(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.is_trained = False
    
    def train_pipeline(self, df):
        """Train the complete pipeline"""
        print("🚀 Training Hypertension Prediction Pipeline...")
        
        # 1. Hypertension Stage Classifier
        print("\n1. Training Hypertension Stage Classifier...")
        X_stage = df[['systolic_bp', 'diastolic_bp', 'age', 'bmi', 'creatinine']]
        y_stage = df['hypertension_stage']
        
        le_stage = LabelEncoder()
        y_stage_encoded = le_stage.fit_transform(y_stage)
        
        X_stage_train, X_stage_test, y_stage_train, y_stage_test = train_test_split(
            X_stage, y_stage_encoded, test_size=0.2, random_state=42
        )
        
        scaler_stage = StandardScaler()
        X_stage_train_scaled = scaler_stage.fit_transform(X_stage_train)
        X_stage_test_scaled = scaler_stage.transform(X_stage_test)
        
        stage_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        stage_model.fit(X_stage_train_scaled, y_stage_train)
        
        stage_accuracy = accuracy_score(y_stage_test, stage_model.predict(X_stage_test_scaled))
        print(f"   Stage Classification Accuracy: {stage_accuracy:.4f}")
        
        # 2. Hypertension Subtype Classifier
        print("\n2. Training Hypertension Subtype Classifier...")
        X_subtype = df[['systolic_bp', 'diastolic_bp', 'age', 'bmi', 'creatinine', 
                        'family_history', 'heart_damage', 'kidney_damage']]
        y_subtype = df['hypertension_subtype']
        
        le_subtype = LabelEncoder()
        y_subtype_encoded = le_subtype.fit_transform(y_subtype)
        
        X_subtype_train, X_subtype_test, y_subtype_train, y_subtype_test = train_test_split(
            X_subtype, y_subtype_encoded, test_size=0.2, random_state=42
        )
        
        scaler_subtype = StandardScaler()
        X_subtype_train_scaled = scaler_subtype.fit_transform(X_subtype_train)
        X_subtype_test_scaled = scaler_subtype.transform(X_subtype_test)
        
        subtype_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        subtype_model.fit(X_subtype_train_scaled, y_subtype_train)
        
        subtype_accuracy = accuracy_score(y_subtype_test, subtype_model.predict(X_subtype_test_scaled))
        print(f"   Subtype Classification Accuracy: {subtype_accuracy:.4f}")
        
        # 3. Risk Prediction Models (one for each timeframe and condition)
        print("\n3. Training Risk Prediction Models...")
        risk_features = ['age', 'sex', 'bmi', 'family_history', 'creatinine', 
                         'systolic_bp', 'diastolic_bp', 'education_level', 'income_level',
                         'heart_damage', 'kidney_damage', 'brain_damage', 'eye_damage', 'vessel_damage']
        
        # Generate synthetic risk data based on organ damage and other factors
        # This simulates what the ML model will learn from the data
        risk_models = {}
        risk_scalers = {}
        
        for risk_type in ['kidney_risk_1yr', 'kidney_risk_2yr', 'stroke_risk_1yr', 'heart_risk_1yr', 'vision_risk_2yr']:
            print(f"   Training {risk_type} model...")
            
            # Create synthetic target based on organ damage and other factors
            if 'kidney' in risk_type:
                base_risk = df['kidney_damage'] * 100
                timeframe_factor = 1.2 if '2yr' in risk_type else 1.0
                y_risk = np.clip(base_risk * timeframe_factor + 
                                 df['age'] * 0.1 + 
                                 (df['systolic_bp'] - 120) * 0.05 +
                                 (df['diastolic_bp'] - 80) * 0.03, 0, 100)
            
            elif 'stroke' in risk_type:
                base_risk = df['brain_damage'] * 80
                y_risk = np.clip(base_risk + 
                                 df['age'] * 0.15 + 
                                 (df['systolic_bp'] - 120) * 0.08 +
                                 (df['diastolic_bp'] - 80) * 0.05, 0, 100)
            
            elif 'heart' in risk_type:
                base_risk = df['heart_damage'] * 90
                y_risk = np.clip(base_risk + 
                                 df['age'] * 0.12 + 
                                 (df['systolic_bp'] - 120) * 0.07 +
                                 (df['diastolic_bp'] - 80) * 0.04, 0, 100)
            
            elif 'vision' in risk_type:
                base_risk = df['eye_damage'] * 70
                timeframe_factor = 1.3 if '2yr' in risk_type else 1.0
                y_risk = np.clip(base_risk * timeframe_factor + 
                                 df['age'] * 0.08 + 
                                 (df['systolic_bp'] - 120) * 0.04 +
                                 (df['diastolic_bp'] - 80) * 0.02, 0, 100)
            
            # Add some noise to make it more realistic
            y_risk = y_risk * np.random.normal(1, 0.1, len(y_risk))
            y_risk = np.clip(y_risk, 0, 100)
            
            X_risk = df[risk_features]
            
            X_risk_train, X_risk_test, y_risk_train, y_risk_test = train_test_split(
                X_risk, y_risk, test_size=0.2, random_state=42
            )
            
            scaler_risk = StandardScaler()
            X_risk_train_scaled = scaler_risk.fit_transform(X_risk_train)
            X_risk_test_scaled = scaler_risk.transform(X_risk_test)
            
            risk_model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
            risk_model.fit(X_risk_train_scaled, y_risk_train)
            
            y_risk_pred = risk_model.predict(X_risk_test_scaled)
            risk_mae = mean_absolute_error(y_risk_test, y_risk_pred)
            risk_r2 = r2_score(y_risk_test, y_risk_pred)
            
            print(f"     {risk_type} - MAE: {risk_mae:.2f}%, R²: {risk_r2:.4f}")
            
            risk_models[risk_type] = risk_model
            risk_scalers[risk_type] = scaler_risk
        
        # Store models and components
        self.models = {
            'stage_classifier': stage_model,
            'subtype_classifier': subtype_model,
            'risk_predictors': risk_models
        }
        
        self.scalers = {
            'stage': scaler_stage,
            'subtype': scaler_subtype,
            'risk': risk_scalers
        }
        
        self.encoders = {
            'stage': le_stage,
            'subtype': le_subtype
        }
        
        self.features = {
            'stage': list(X_stage.columns),
            'subtype': list(X_subtype.columns),
            'risk': risk_features
        }
        
        self.is_trained = True
        print("✅ Pipeline training complete!")
    
    def predict_patient(self, patient_data):
        """Complete prediction for a single patient"""
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train_pipeline() first.")
        
        # 1. Predict Hypertension Stage
        stage_features = [patient_data[feature] for feature in self.features['stage']]
        stage_scaled = self.scalers['stage'].transform([stage_features])
        stage_encoded = self.models['stage_classifier'].predict(stage_scaled)[0]
        hypertension_stage = self.encoders['stage'].inverse_transform([stage_encoded])[0]
        
        # Get prediction probabilities for confidence score
        stage_probs = self.models['stage_classifier'].predict_proba(stage_scaled)[0]
        stage_confidence = max(stage_probs) * 100
        
        # 2. Predict Hypertension Subtype
        subtype_features = [patient_data[feature] for feature in self.features['subtype']]
        subtype_scaled = self.scalers['subtype'].transform([subtype_features])
        subtype_encoded = self.models['subtype_classifier'].predict(subtype_scaled)[0]
        hypertension_subtype = self.encoders['subtype'].inverse_transform([subtype_encoded])[0]
        
        subtype_probs = self.models['subtype_classifier'].predict_proba(subtype_scaled)[0]
        subtype_confidence = max(subtype_probs) * 100
        
        # 3. Predict Risks
        risk_features = [patient_data[feature] for feature in self.features['risk']]
        risk_predictions = {}
        
        for risk_type, risk_model in self.models['risk_predictors'].items():
            risk_scaled = self.scalers['risk'][risk_type].transform([risk_features])
            risk_prediction = risk_model.predict(risk_scaled)[0]
            risk_predictions[risk_type] = max(0, min(100, risk_prediction))
        
        # Generate risk statements
        risk_statements = generate_risk_statements(risk_predictions)
        
        return {
            'hypertension_stage': {
                'prediction': hypertension_stage,
                'confidence': round(stage_confidence, 1),
                'explanation': self._generate_stage_explanation(hypertension_stage, patient_data)
            },
            'hypertension_subtype': {
                'prediction': hypertension_subtype,
                'confidence': round(subtype_confidence, 1),
                'explanation': self._generate_subtype_explanation(hypertension_subtype)
            },
            'risk_predictions': risk_predictions,
            'risk_statements': risk_statements
        }
    
    def _generate_stage_explanation(self, stage, patient_data):
        """Generate explanation for hypertension stage"""
        explanations = {
            'Normal': "Blood pressure within healthy range. No immediate intervention needed.",
            'Elevated': "Borderline elevated blood pressure. Lifestyle modifications recommended.",
            'Stage_1': "Stage 1 hypertension. Medication and lifestyle changes advised.",
            'Stage_2': "Stage 2 hypertension. Requires aggressive treatment and monitoring.",
            'Crisis': "Hypertensive crisis. Immediate medical attention required.",
            'Unknown': "Unable to determine stage from provided readings."
        }
        return explanations.get(stage, explanations['Unknown'])
    
    def _generate_subtype_explanation(self, subtype):
        """Generate explanation for hypertension subtype"""
        explanations = {
            'Primary': "Essential hypertension - most common type, often related to lifestyle and genetics.",
            'Secondary_Endocrine': "Secondary to endocrine disorders like thyroid or adrenal problems.",
            'Secondary_Renal': "Kidney-related hypertension requiring renal function assessment.",
            'Secondary_Renovascular': "Related to renal artery stenosis or vascular kidney issues.",
            'Secondary_Sleep_Apnea': "Associated with obstructive sleep apnea and nocturnal hypoxia.",
            'White_Coat': "Elevated BP in clinical settings only. Home monitoring recommended.",
            'Masked': "Normal clinic BP but elevated home readings. Requires ambulatory monitoring.",
            'No_Hypertension': "No evidence of hypertension."
        }
        return explanations.get(subtype, "Unknown hypertension subtype.")

# Test the pipeline
def test_pipeline(pipeline):
    """Test the complete hypertension pipeline"""
    print("\n🧪 TESTING COMPLETE HYPERTENSION PIPELINE")
    print("=" * 60)
    
    test_patients = [
        {
            'name': 'Stage 2 Primary Hypertension',
            'data': {
                'age': 65, 'sex': 1, 'bmi': 32, 'family_history': 1,
                'creatinine': 1.2, 'systolic_bp': 160, 'diastolic_bp': 100,
                'education_level': 2, 'income_level': 2,
                'heart_damage': 0.25, 'kidney_damage': 0.18, 
                'brain_damage': 0.15, 'eye_damage': 0.10, 'vessel_damage': 0.30
            }
        },
        {
            'name': 'White Coat Hypertension',
            'data': {
                'age': 45, 'sex': 0, 'bmi': 26, 'family_history': 0,
                'creatinine': 0.9, 'systolic_bp': 145, 'diastolic_bp': 90,
                'education_level': 4, 'income_level': 3,
                'heart_damage': 0.08, 'kidney_damage': 0.06, 
                'brain_damage': 0.05, 'eye_damage': 0.03, 'vessel_damage': 0.10
            }
        }
    ]
    
    for patient in test_patients:
        print(f"\n📋 Patient: {patient['name']}")
        print("-" * 50)
        
        print("Patient Profile:")
        print(f"  Age: {patient['data']['age']}, BP: {patient['data']['systolic_bp']}/{patient['data']['diastolic_bp']}")
        print(f"  BMI: {patient['data']['bmi']}, Creatinine: {patient['data']['creatinine']}")
        
        # Get predictions
        results = pipeline.predict_patient(patient['data'])
        
        print(f"\n🎯 Hypertension Stage: {results['hypertension_stage']['prediction']}")
        print(f"   Confidence: {results['hypertension_stage']['confidence']}%")
        print(f"   Explanation: {results['hypertension_stage']['explanation']}")
        
        print(f"\n🔍 Hypertension Subtype: {results['hypertension_subtype']['prediction']}")
        print(f"   Confidence: {results['hypertension_subtype']['confidence']}%")
        print(f"   Explanation: {results['hypertension_subtype']['explanation']}")
        
        print(f"\n⚠  Complication Risks:")
        # In test_pipeline(), when printing risk lines:
        for risk_type, risk_value in results['risk_predictions'].items():
            condition = risk_type.split('_')[0].title()
            yrs_token = risk_type.split('_')[-1]  # like '1yr' or '2yr'
            years = int(yrs_token.replace('yr', ''))
            timeframe = f"{years} year" if years == 1 else f"{years} years"
            print(f"   {condition} risk in {timeframe}: {float(risk_value):.1f}%")

        print(f"\n💬 Risk Statements:")
        for statement in results['risk_statements']:
            print(f"   • {statement}")
        
        print("-" * 50)

if __name__ == "__main__":
    # Specify the dataset path here (with 'r' for raw string)
    dataset_path = r"C:\Users\IYYAPPAN\Desktop\Health_app\backend\hypertension_class\hypertension_data_20250901_230818.csv"
    
    # Check if the file exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        print("Please specify the correct path to your hypertension dataset CSV file.")
        print("You can either:")
        print("1. Update the 'dataset_path' variable above to point to your CSV file")
        print("2. Place your CSV file in the same directory and update the filename")
        print("3. Run generate_data.py first to generate a synthetic dataset")
        exit(1)
    
    print(f"📁 Loading data from: {dataset_path}")
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    print(f"📊 Dataset shape: {df.shape}")
    
    # Train the complete pipeline
    pipeline = HypertensionPipeline()
    pipeline.train_pipeline(df)
    
    # Save the complete pipeline
    model_dir = r"C:\Users\IYYAPPAN\Desktop\Health_app\backend\hypertension_class"
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(pipeline, os.path.join(model_dir, "complete_pipeline.pkl"))
    joblib.dump(HYPERTENSION_STAGES, os.path.join(model_dir, "hypertension_stages.pkl"))
    joblib.dump(HYPERTENSION_SUBTYPES, os.path.join(model_dir, "hypertension_subtypes.pkl"))
    joblib.dump(RISK_METADATA, os.path.join(model_dir, "risk_metadata.pkl"))
    
    print("✅ Complete pipeline saved successfully!")
    
    # Test the pipeline
    test_pipeline(pipeline)