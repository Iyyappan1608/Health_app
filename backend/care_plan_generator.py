from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from groq import Groq
import re
import mysql.connector
from datetime import datetime
import json

# ----------------- Direct API Key Declaration -----------------
api_key = "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"

if not api_key or api_key.strip() == "":
    raise ValueError("❌ No API key found! Please add your Groq API key to the api_key variable.")

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'root',
    'password': '1234',
    'database': 'health_app_db'
}

# ----------------- Server Check -----------------
def check_groq_server(api_key, model="llama-3.1-8b-instant"):
    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model=model,
            max_tokens=1,
            temperature=0.1
        )
        if response.choices[0].message.content:
            return True
    except Exception:
        return False
    return False

# ----------------- Initialize LLM -----------------
server_status = check_groq_server(api_key)
llm_engine = None
if server_status:
    try:
        llm_engine = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            max_retries=2,
            timeout=30
        )
    except Exception as e:
        print(f"⚠ Failed to init LLM: {e}")
        server_status = False

# ----------------- PROMPTS & HELPERS -----------------
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a senior healthcare professional generating scientifically-grounded 7-day care plans. "
    "You must perform comprehensive analysis of the patient's medical data and create a practical, "
    "safe, and evidence-based care plan. EVERY recommendation must be medically appropriate for "
    "the patient's specific age, conditions, and physical capabilities."
)

def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def _extract_plan_only(text: str) -> str:
    patterns = [
        "Enhanced 7-Day Care Plan",
        "7-Day Care Plan",
        "Day 1",
        "Personalized Care Plan",
        "Daily Care Plan"
    ]
    for pattern in patterns:
        idx = text.find(pattern)
        if idx != -1:
            return text[idx:].strip()
    return text.strip()

def _format_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r'(Day \d+)', r'\n\n\1\n', text)
    text = re.sub(r'(🏃|🧘|🥗|💧|❌|✅|⚠)', r'\n\1', text)
    text = re.sub(r'→', ' → ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _validate_plan(text: str) -> None:
    required = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
    for d in required:
        if d not in text:
            raise ValueError("Plan missing days")

def _remove_unwanted_sections(text: str) -> str:
    sections_to_remove = [
        "Monitoring Instructions:",
        "Medication Interactions:",
        "Fall Risk Precautions:",
        "Conclusion:",
        "Regular monitoring and follow-up",
        "Medication adherence:"
    ]
    for section in sections_to_remove:
        idx = text.find(section)
        if idx != -1:
            next_section = None
            for next_sec in sections_to_remove:
                if next_sec != section:
                    next_idx = text.find(next_sec, idx + len(section))
                    if next_idx != -1 and (next_section is None or next_idx < next_section):
                        next_section = next_idx
            if next_section is not None:
                text = text[:idx] + text[next_section:]
            else:
                text = text[:idx]
    return text.strip()

# ----------------- DATABASE HELPERS -----------------
def store_user_interaction(patient_id, day_number, question, response, care_plan_text=None):
    """Store user interactions in the database"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        sql = """
        INSERT INTO user_care_plan_interactions 
        (patient_id, day_number, question, response, care_plan_text, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (patient_id, day_number, question, response, care_plan_text, datetime.now()))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error storing user interaction: {e}")
        return False

def get_patient_health_data(patient_id):
    """Fetch all available health data for a patient from the database"""
    health_data = {
        "patient_info": {},
        "chronic_disease": {},
        "diabetes": {},
        "hypertension": {},
        "wearable": {}
    }
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # Get basic patient info
        cursor.execute("SELECT name, email FROM patients WHERE id = %s", (patient_id,))
        patient_info = cursor.fetchone()
        if patient_info:
            health_data["patient_info"] = patient_info
            print(f"Found patient info: {patient_info}")
        
        # Get latest chronic disease prediction
        cursor.execute("""
            SELECT input_data, output_data 
            FROM user_predictions 
            WHERE patient_id = %s AND prediction_type = 'chronic_disease'
            ORDER BY created_at DESC LIMIT 1
        """, (patient_id,))
        chronic_data = cursor.fetchone()
        if chronic_data:
            try:
                health_data["chronic_disease"] = {
                    "input": json.loads(chronic_data["input_data"]) if chronic_data["input_data"] else {},
                    "output": json.loads(chronic_data["output_data"]) if chronic_data["output_data"] else {}
                }
                print("Found chronic disease data")
            except json.JSONDecodeError:
                print("Error parsing chronic disease JSON data")
        
        # Get latest diabetes check
        cursor.execute("""
            SELECT is_pregnant, age_at_diagnosis, bmi_at_diagnosis, family_history, 
                   hba1c, c_peptide_level, autoantibodies_status, genetic_test_result, report_json 
            FROM diabetes_checks 
            WHERE user_id = %s
            ORDER BY created_at DESC LIMIT 1
        """, (patient_id,))
        diabetes_data = cursor.fetchone()
        if diabetes_data:
            health_data["diabetes"] = {
                "input": {
                    "is_pregnant": diabetes_data["is_pregnant"],
                    "age_at_diagnosis": diabetes_data["age_at_diagnosis"],
                    "bmi_at_diagnosis": diabetes_data["bmi_at_diagnosis"],
                    "family_history": diabetes_data["family_history"],
                    "hba1c": diabetes_data["hba1c"],
                    "c_peptide_level": diabetes_data["c_peptide_level"],
                    "autoantibodies_status": diabetes_data["autoantibodies_status"],
                    "genetic_test_result": diabetes_data["genetic_test_result"]
                }
            }
            try:
                health_data["diabetes"]["output"] = json.loads(diabetes_data["report_json"]) if diabetes_data["report_json"] else {}
                print("Found diabetes data")
            except (json.JSONDecodeError, TypeError):
                health_data["diabetes"]["output"] = {}
        
        # Get latest hypertension check
        cursor.execute("""
            SELECT age, sex, bmi, family_history, creatinine, systolic_bp, diastolic_bp, report_json 
            FROM hypertension_checks 
            WHERE user_id = %s
            ORDER BY created_at DESC LIMIT 1
        """, (patient_id,))
        hypertension_data = cursor.fetchone()
        if hypertension_data:
            health_data["hypertension"] = {
                "input": {
                    "age": hypertension_data["age"],
                    "sex": hypertension_data["sex"],
                    "bmi": hypertension_data["bmi"],
                    "family_history": hypertension_data["family_history"],
                    "creatinine": hypertension_data["creatinine"],
                    "systolic_bp": hypertension_data["systolic_bp"],
                    "diastolic_bp": hypertension_data["diastolic_bp"]
                }
            }
            try:
                health_data["hypertension"]["output"] = json.loads(hypertension_data["report_json"]) if hypertension_data["report_json"] else {}
                print("Found hypertension data")
            except (json.JSONDecodeError, TypeError):
                health_data["hypertension"]["output"] = {}
        
        # Get latest wearable data
        cursor.execute("""
            SELECT input_data, output_data 
            FROM user_predictions 
            WHERE patient_id = %s AND prediction_type = 'wearable'
            ORDER BY created_at DESC LIMIT 1
        """, (patient_id,))
        wearable_data = cursor.fetchone()
        if wearable_data:
            try:
                health_data["wearable"] = {
                    "input": json.loads(wearable_data["input_data"]) if wearable_data["input_data"] else {},
                    "output": json.loads(wearable_data["output_data"]) if wearable_data["output_data"] else {}
                }
                print("Found wearable data")
            except json.JSONDecodeError:
                print("Error parsing wearable JSON data")
        
        cursor.close()
        conn.close()
        
        # Check if we have any meaningful data
        has_data = any([
            health_data["chronic_disease"],
            health_data["diabetes"].get("input"),
            health_data["hypertension"].get("input"),
            health_data["wearable"]
        ])
        
        if not has_data:
            print(f"Warning: No health data found for patient {patient_id}")
        
        return health_data
        
    except Exception as e:
        print(f"Error fetching patient health data: {e}")
        import traceback
        traceback.print_exc()
        return health_data
# ----------------- SPLIT & PRINT HELPERS -----------------
def split_days(plan_text: str):
    """Split care plan into dictionary with Day 1..Day 7"""
    days = {}
    matches = re.split(r"\n(?=Day \d+)", plan_text.strip())
    for m in matches:
        if m.strip():
            day_label = m.split("\n")[0].strip()
            days[day_label] = m.strip()
    return days

def print_day(day_text: str, day_number: int, patient_id: int = None):
    """Print day content and ask for user input, storing responses"""
    print(f"\n{'='*60}")
    print(f"DAY {day_number} PLAN")
    print(f"{'='*60}")

    if not day_text or not day_text.strip():
        print("No plan available for this day.")
    else:
        # nicer formatting for display
        formatted_text = re.sub(r'(🏃|🧘|🥗|💧|❌|✅|⚠)', r'\n\n\1', day_text).strip()
        formatted_text = re.sub(r'→', ' → ', formatted_text)
        formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)
        print(formatted_text)

    # Ask user for choice
    user_choice = input("\nDid you follow today's care plan? (yes/no): ").strip().lower()

    # Store the interaction
    if patient_id:
        store_user_interaction(patient_id, day_number, "followed_plan", user_choice, day_text)

    # Show reduction or consequences based on response
    reduction_idx = day_text.find("✅ Today's risk reduction")
    consequences_idx = day_text.find("⚠ Consequences if skipped")

    if user_choice == "yes" and reduction_idx != -1:
        next_idx = consequences_idx if consequences_idx != -1 else len(day_text)
        reduction_text = day_text[reduction_idx:next_idx].strip()
        reduction_text = re.sub(r'→', ' → ', reduction_text)
        print(f"\n{reduction_text}")
    elif user_choice == "no" and consequences_idx != -1:
        consequences_text = day_text[consequences_idx:].strip()
        consequences_text = re.sub(r'→', ' → ', consequences_text)
        print(f"\n{consequences_text}")

    return user_choice

# ----------------- CARE PLAN GENERATION -----------------
def generate_care_plan_from_report(report_text: str) -> str:
    if llm_engine is None:
        raise Exception("LLM engine not available.")

    # Enhanced prompt with strict instructions for personalization (from cp.py)
    prompt_template = ChatPromptTemplate.from_messages([
        system_prompt,
        HumanMessagePromptTemplate.from_template(""" 

CRITICAL HEALTHCARE ANALYSIS TASK:

PATIENT REPORT TO ANALYZE:
{report_text}

IMPORTANT: ONLY generate the 7-day care plan. DO NOT include any monitoring instructions, medication interactions, fall risk precautions, or conclusion sections.

REQUIRED ANALYSIS PHASE:
1. Extract ALL medical data: demographics, vitals, test results, diagnosed conditions
2. Identify primary health issues and their severity levels
3. Assess patient capabilities based on age, mobility, and existing conditions
4. Determine contraindications and safety limitations

SCIENTIFIC CARE PLAN CREATION RULES:
- Base EVERY recommendation strictly on the patient's actual test results and medical data
- Prioritize SAFETY and medical appropriateness above all else
- Create VARIED but PRACTICAL activities and meals
- Ensure ALL activities are AGE-APPROPRIATE and CONDITION-SAFE
- Include specific, measurable instructions: portion sizes, durations, frequencies
- Address ALL identified medical conditions in each day's plan
- Consider medication interactions, mobility limitations, and fall risks
- Balance variety with practicality

DAILY STRUCTURE (ONLY include these sections):
Day [X]  
🏃 Physical Activity: [Medically safe activity for patient's age/condition] → [Physiological benefit specific to their health issues]  
🧘 Mental Wellness: [Appropriate technique] → [Mental health benefit addressing their specific needs]  
🥗 Meals: [Practical meals with specific ingredients/portions] → [Nutritional benefits for their conditions]  
💧 Hydration: [Appropriate fluid types/amounts] → [Health benefits based on their kidney function, medications, etc.]  
❌ Avoid: [Specific contraindicated items/behaviors] → [Exact risks based on their test results]  
✅ Today's risk reduction: [How today's plan addresses their specific risk percentages]  
⚠ Consequences if skipped: [Realistic worsening of their actual medical conditions]

MANDATORY REQUIREMENTS:
- Reference SPECIFIC patient data in every recommendation
- Ensure all activities are medically safe for their age and conditions
- Maintain practicality - the plan must be executable in real life
- Balance variety with consistency where medically beneficial
- Prioritize evidence-based recommendations over novelty

ABSOLUTELY PROHIBITED:
- Generic advice not tied to specific patient data
- Medically unsafe recommendations
- Activities beyond patient's physical capabilities
- Nutritionally inappropriate suggestions
- Recommendations that contradict their medical conditions
- Monitoring instructions, medication interactions, fall risk precautions, or conclusion sections

The plan must be clinically sound, practical, and tailored to this individual's capabilities and limitations.
""")
    ])

    chain = prompt_template | llm_engine | StrOutputParser()
    response = chain.invoke({"report_text": report_text})
    text = _strip_think_tags(response)
    text = _extract_plan_only(text)
    text = _format_output(text)
    text = _remove_unwanted_sections(text)  # Remove unwanted sections
    
    # ✅ Safer validation
    required = [f"Day {i}" for i in range(1, 8)]
    missing = [d for d in required if d not in text]
    if missing:
        print(f"⚠ Warning: Care plan missing {', '.join(missing)}.")
        # Do not raise, return partial plan

    return text
def generate_care_plan_from_health_data(patient_id: int) -> str:
    """Generate a care plan from patient health data in the database"""
    if llm_engine is None:
        raise Exception("LLM engine not available.")
    
    # Get health data from database
    health_data = get_patient_health_data(patient_id)
    
    # Format the health data for the LLM
    health_summary = f"COMPREHENSIVE PATIENT HEALTH DATA:\n\n{json.dumps(health_data, indent=2)}"
    
    # Use the enhanced prompt format from cp.py
    prompt_template = ChatPromptTemplate.from_messages([
        system_prompt,
        HumanMessagePromptTemplate.from_template(""" 

CRITICAL HEALTHCARE ANALYSIS TASK:

PATIENT HEALTH DATA TO ANALYZE:
{health_summary}

IMPORTANT: ONLY generate the 7-day care plan. DO NOT include any monitoring instructions, medication interactions, fall risk precautions, or conclusion sections.

REQUIRED ANALYSIS PHASE:
1. Extract ALL medical data: demographics, vitals, test results, diagnosed conditions
2. Identify primary health issues and their severity levels
3. Assess patient capabilities based on age, mobility, and existing conditions
4. Determine contraindications and safety limitations

SCIENTIFIC CARE PLAN CREATION RULES:
- Base EVERY recommendation strictly on the patient's actual test results and medical data
- Prioritize SAFETY and medical appropriateness above all else
- Create VARIED but PRACTICAL activities and meals
- Ensure ALL activities are AGE-APPROPRIATE and CONDITION-SAFE
- Include specific, measurable instructions: portion sizes, durations, frequencies
- Address ALL identified medical conditions in each day's plan
- Consider medication interactions, mobility limitations, and fall risks
- Balance variety with practicality

DAILY STRUCTURE (ONLY include these sections):
Day [X]  
🏃 Physical Activity: [Medically safe activity for patient's age/condition] → [Physiological benefit specific to their health issues]  
🧘 Mental Wellness: [Appropriate technique] → [Mental health benefit addressing their specific needs]  
🥗 Meals: [Practical meals with specific ingredients/portions] → [Nutritional benefits for their conditions]  
💧 Hydration: [Appropriate fluid types/amounts] → [Health benefits based on their kidney function, medications, etc.]  
❌ Avoid: [Specific contraindicated items/behaviors] → [Exact risks based on their test results]  
✅ Today's risk reduction: [How today's plan addresses their specific risk percentages]  
⚠ Consequences if skipped: [Realistic worsening of their actual medical conditions]

MANDATORY REQUIREMENTS:
- Reference SPECIFIC patient data in every recommendation
- Ensure all activities are medically safe for their age and conditions
- Maintain practicality - the plan must be executable in real life
- Balance variety with consistency where medically beneficial
- Prioritize evidence-based recommendations over novelty

ABSOLUTELY PROHIBITED:
- Generic advice not tied to specific patient data
- Medically unsafe recommendations
- Activities beyond patient's physical capabilities
- Nutritionally inappropriate suggestions
- Recommendations that contradict their medical conditions
- Monitoring instructions, medication interactions, fall risk precautions, or conclusion sections

The plan must be clinically sound, practical, and tailored to this individual's capabilities and limitations.
""")
    ])

    chain = prompt_template | llm_engine | StrOutputParser()
    response = chain.invoke({"health_summary": health_summary})
    text = _strip_think_tags(response)
    text = _extract_plan_only(text)
    text = _format_output(text)
    text = _remove_unwanted_sections(text)

    # ✅ Safer validation
    required = [f"Day {i}" for i in range(1, 8)]
    missing = [d for d in required if d not in text]
    if missing:
        print(f"⚠ Warning: Care plan missing {', '.join(missing)}.")
        # Do not raise, return partial plan

    return text
# ----------------- MAIN FUNCTION FOR EXTERNAL USE -----------------
def generate_care_plan_for_patient(patient_id: int) -> dict:
    """
    Main function to generate a care plan for a specific patient.
    Returns a dictionary with the care plan text and structured days.
    """
    try:
        if not server_status or not llm_engine:
            return {"error": "LLM service not available"}
        
        # Get patient health data first
        health_data = get_patient_health_data(patient_id)
        
        # Check if we have any health data
        if not health_data or not any(health_data.values()):
            return {"error": "No health data found for patient. Please complete some health assessments first."}
        
        print(f"Generating care plan for patient {patient_id} with data: {health_data}")
        
        # Generate care plan from patient data
        care_plan_text = generate_care_plan_from_health_data(patient_id)
        
        # Validate care plan was generated
        if not care_plan_text or care_plan_text.strip() == "":
            return {"error": "Failed to generate care plan - empty response from LLM"}
        
        print(f"Generated care plan text length: {len(care_plan_text)}")
        
        # Split into structured format
        days_dict = split_days(care_plan_text)
        
        # Validate we have at least some days
        if not days_dict:
            return {"error": "Failed to parse care plan into daily structure"}
        
        print(f"Parsed into {len(days_dict)} days: {list(days_dict.keys())}")
        
        return {
            "success": True,
            "care_plan": care_plan_text,
            "structured_plan": days_dict,
            "patient_data_summary": health_data  # Include for debugging
        }
        
    except Exception as e:
        print(f"Error in generate_care_plan_for_patient: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to generate care plan: {str(e)}"}