import AsyncStorage from '@react-native-async-storage/async-storage';

const API_BASE_URL = "http://172.20.10.10:5000/";

const getAuthHeaders = async () => {
  const token = await AsyncStorage.getItem('session_token');
  return {
    'Content-Type': 'application/json',
    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
  };
};

const get = async (endpoint: string) => {
  try {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_BASE_URL}${endpoint}`, { method: 'GET', headers });
    const data = await response.json();
    return { ok: response.ok, data };
  } catch (error) {
    console.error(`API Error on GET ${endpoint}:`, error);
    return { ok: false, data: { message: 'Network error or server is down.' } };
  }
};

// In ApiService.ts, update the post function to include better error handling:
const post = async (endpoint: string, body: object, isAuthenticated = true) => {
  try {
    const headers = isAuthenticated ? await getAuthHeaders() : { 'Content-Type': 'application/json' };
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
    
    // First, check if the response is OK (status 200-299)
    if (!response.ok) {
      // Try to get error message from JSON response
      let errorData;
      try {
        errorData = await response.json();
      } catch (e) {
        // If response is not JSON, use status text
        errorData = { message: `Server error: ${response.status} ${response.statusText}` };
      }
      return { ok: false, data: errorData, status: response.status };
    }
    
    // If response is OK, parse the JSON
    const data = await response.json();
    return { ok: true, data, status: response.status };
    
  } catch (error) {
    console.error(`API Error on POST ${endpoint}:`, error);
    return { 
      ok: false, 
      data: { 
        message: 'Network error or server is down.',
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      status: 0
    };
  }
};

const ApiService = {
  // Updated to match backend routes
  signup: (name: string, email: string, password: string) => post('signup', { name, email, password }, false),
  login: (email: string, password: string) => post('login', { email, password }, false),

  generateReport: (patientData: object) => post('generate_report', patientData),
  predictDiabetes: (diabetesData: object) => post('predict_diabetes_subtype', diabetesData),
  predictHypertension: (hypertensionData: object) => post('predict_hypertension', hypertensionData),
  generateCarePlan: () => post('generate_care_plan', {}),
  getLiveUpdate: () => get('get_live_update'),
  
  // Add new endpoints if needed
  wearableConnect: () => get('wearable/connect'),
  liveMonitor: () => post('live_monitor', {}),
};

export default ApiService;