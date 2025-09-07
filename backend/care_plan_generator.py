from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from groq import Groq
import re
import mysql.connector
import json

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Lohith@123',
    'database': 'health_app_db'
}

# ----------------- Direct API Key Declaration -----------------
api_key = "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"

if not api_key or api_key.strip() == "":
    raise ValueError("❌ No API key found! Please add your Groq API key to the api_key variable.")

# ----------------- Database Functions -----------------
def get_patient_data(patient_id):
    """Retrieve all prediction data for a patient from the database"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # First check what columns exist in the patients table
        cursor.execute("SHOW COLUMNS FROM patients")
        columns = [col[0] for col in cursor.fetchall()]
        
        # Build query based on available columns
        select_columns = []
        if 'name' in columns:
            select_columns.append('name')
        if 'email' in columns:
            select_columns.append('email')
        if 'age' in columns:
            select_columns.append('age')
        if 'gender' in columns:
            select_columns.append('gender')
        
        if select_columns:
            column_list = ', '.join(select_columns)
            cursor.execute(f"SELECT {column_list} FROM patients WHERE id = %s", (patient_id,))
            patient_info = cursor.fetchone()
        else:
            patient_info = {}
        
        if not patient_info:
            print(f"No patient found with ID {patient_id}")
            return None
        
        # Get all predictions for the patient
        query = """
            SELECT prediction_type, input_data, output_data, created_at 
            FROM user_predictions 
            WHERE patient_id = %s 
            ORDER BY created_at DESC
        """
        
        cursor.execute(query, (patient_id,))
        predictions = cursor.fetchall()
        
        return {
            'patient_info': patient_info,
            'predictions': predictions
        }
        
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return None
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def format_patient_report(patient_data):
    """Format patient data into a comprehensive report for the LLM"""
    if not patient_data:
        return "No patient data available"
    
    report_parts = []
    patient_info = patient_data.get('patient_info', {})
    predictions = patient_data.get('predictions', [])
    
    # Add patient info
    report_parts.append("--- PATIENT INFORMATION ---")
    report_parts.append(f"Name: {patient_info.get('name', 'Not specified')}")
    
    if 'age' in patient_info and patient_info.get('age') is not None:
        report_parts.append(f"Age: {patient_info.get('age')}")
    if 'gender' in patient_info and patient_info.get('gender') is not None:
        report_parts.append(f"Gender: {patient_info.get('gender')}")
    
    report_parts.append("")
    
    # Process each prediction
    for prediction in predictions:
        pred_type = prediction['prediction_type']
        input_data = json.loads(prediction['input_data']) if prediction['input_data'] else {}
        output_data = json.loads(prediction['output_data']) if prediction['output_data'] else {}
        
        if pred_type == 'chronic_disease':
            report_parts.append("--- CHRONIC DISEASE ASSESSMENT ---")
            if output_data.get('predicted_conditions'):
                for condition in output_data['predicted_conditions']:
                    report_parts.append(f"- Disease: {condition.get('disease', 'Unknown')}")
                    report_parts.append(f"  Explanation: {condition.get('explanation', 'No explanation provided')}")
            
            if output_data.get('risk_assessment'):
                for risk in output_data['risk_assessment']:
                    report_parts.append(f"  Risk Score: {risk.get('risk_score', 'N/A')}/100 ({risk.get('risk_level', 'Unknown')})")
                    report_parts.append(f"  Reasoning: {risk.get('primary_drivers', 'No reasoning provided')}")
            report_parts.append("")
        
        elif pred_type == 'hypertension':
            report_parts.append("--- HYPERTENSION ASSESSMENT ---")
            report_parts.append(f"Hypertension Risk: {'Yes' if output_data.get('hypertension_risk') else 'No'}")
            report_parts.append(f"Probability: {output_data.get('probability', 'N/A')}")
            report_parts.append(f"Risk Level: {output_data.get('risk_level', 'Unknown')}")
            report_parts.append(f"Stage: {output_data.get('stage', 'Unknown')}")
            report_parts.append(f"Subtype: {output_data.get('subtype', 'Unknown')}")
            if 'kidney_risk_1yr' in output_data:
                report_parts.append(f"Kidney Risk (1yr): {output_data.get('kidney_risk_1yr', 'N/A')}%")
            if 'stroke_risk_1yr' in output_data:
                report_parts.append(f"Stroke Risk (1yr): {output_data.get('stroke_risk_1yr', 'N/A')}%")
            if 'heart_risk_1yr' in output_data:
                report_parts.append(f"Heart Risk (1yr): {output_data.get('heart_risk_1yr', 'N/A')}%")
            report_parts.append(f"Explanation: {output_data.get('explanation', 'No explanation provided')}")
            report_parts.append("")
        
        elif pred_type == 'vitals':
            report_parts.append("--- RECENT VITALS ---")
            for key, value in input_data.items():
                if value is not None:
                    report_parts.append(f"{key}: {value}")
            report_parts.append("")
    
    return "\n".join(report_parts)

# ----------------- Initialize LLM -----------------
def initialize_llm():
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            max_retries=2,
            timeout=30
        )
        # Test the connection
        llm.invoke("test")
        return llm
    except Exception as e:
        print(f"⚠ Failed to init LLM: {e}")
        return None

llm_engine = initialize_llm()

# ----------------- PROMPTS -----------------
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a healthcare professional generating a personalized 7-day care plan. "
    "Recommendations must be based ONLY on the provided patient report text (not on generic assumptions). "
    "Create varied and personalized daily recommendations while maintaining medical appropriateness. "
    "Ensure all recommendations are practical and directly tailored to the conditions explicitly identified in the report."
)

def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def _extract_plan_only(text: str) -> str:
    header_idx = text.find("Enhanced 7-Day Care Plan")
    if header_idx == -1:
        header_idx = text.find("7-Day Care Plan")
    if header_idx == -1:
        header_idx = text.find("Day 1")
    if header_idx == -1:
        return text.strip()
    return text[header_idx:].strip()

def _format_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r'(Day \d+)', r'\n\n\1\n', text)
    text = re.sub(r'(🏃[^🧘🥗💧❌✅⚠]+)', r'\1\n', text)
    text = re.sub(r'(🧘[^🥗💧❌✅⚠]+)', r'\1\n', text)
    text = re.sub(r'(🥗[^💧❌✅⚠]+)', r'\1\n', text)
    text = re.sub(r'(💧[^❌✅⚠]+)', r'\1\n', text)
    text = re.sub(r'(❌[^✅⚠]+)', r'\1\n', text)
    text = re.sub(r'(✅[^⚠]+)', r'\1\n', text)
    text = re.sub(r'(⚠.*?)(?=\nDay|\Z)', r'\1\n', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def generate_care_plan_from_report(report_text: str) -> str:
    if llm_engine is None:
        raise Exception("LLM engine not available.")
    
    prompt_template = ChatPromptTemplate.from_messages([
        system_prompt,
        ("human", """As a healthcare professional, generate a personalized 7-day care plan for this patient.

PATIENT REPORT:
{report_text}

GUIDELINES:
- Only use the information provided in the patient report (no assumptions, no generic conditions).
- Address ALL identified conditions with targeted care recommendations.
- Vary activities, meals, and suggestions each day.
- Keep this structure for each day:

Day [X]  
🏃 Physical Activity: [Specific activity] → [How it helps conditions]  
🧘 Mental Wellness: [Specific activity] → [How it helps conditions]  
🥗 Meals: [Specific meals] → [How it helps conditions]  
💧 Hydration: [Plan] → [How it helps conditions]  
❌ Avoid: [Items/behaviors] → [Risks if taken]  
✅ Today's risk reduction: [How today's plan reduces risks]  
⚠ Consequences if skipped: [Realistic risks like worsening health, poor sugar control,or hypertension etc.]""")
    ])
    
    chain = prompt_template | llm_engine | StrOutputParser()
    response = chain.invoke({"report_text": report_text})
    text = _strip_think_tags(response)
    text = _extract_plan_only(text)
    text = _format_output(text)
    return text

# ----------------- Main Execution -----------------
if __name__ == "__main__":
    # Get patient ID from user input
    try:
        patient_id = int(input("Enter patient ID: "))
    except ValueError:
        print("Please enter a valid patient ID (number)")
        exit()
    
    # Retrieve patient data from database
    print("Retrieving patient data from database...")
    patient_data = get_patient_data(patient_id)
    
    if not patient_data:
        print("No patient data found. Please check the patient ID.")
        exit()
    
    # Format the data into a report
    report_input = format_patient_report(patient_data)
    
    # Print the report for debugging
    print("\n" + "="*50)
    print("PATIENT REPORT FOR LLM")
    print("="*50)
    print(report_input)
    print("="*50)
    
    # Generate care plan if LLM is available
    if llm_engine:
        try:
            print("Analyzing patient report and generating care plan...")
            care_plan_output = generate_care_plan_from_report(report_input)
            print("✅ Care plan generated successfully!")
            
            # Print the care plan
            print("\n" + "="*50)
            print("GENERATED CARE PLAN")
            print("="*50)
            print(care_plan_output)
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
    else:
        print("⚠ Please check your Groq API connection.")