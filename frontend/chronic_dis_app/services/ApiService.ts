import AsyncStorage from '@react-native-async-storage/async-storage';

const API_BASE_URL = "http://192.168.1.4:5000/";

const getAuthHeaders = async () => {
  const token = await AsyncStorage.getItem('session_token');
  return {
    'Content-Type': 'application/json',
    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
  };
};

// Separate function for unauthenticated GET requests
const getUnauthenticated = async (endpoint: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, { 
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    const data = await response.json();
    return { ok: response.ok, data };
  } catch (error) {
    console.error(`API Error on GET ${endpoint}:`, error);
    return { ok: false, data: { message: 'Network error or server is down.' } };
  }
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

const post = async (endpoint: string, body: object, isAuthenticated = true) => {
  try {
    const headers = isAuthenticated ? await getAuthHeaders() : { 'Content-Type': 'application/json' };
    
    console.log(`Making POST request to: ${endpoint}`);
    
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
    
    let responseData;
    try {
      responseData = await response.json();
    } catch (e) {
      responseData = { message: `Invalid JSON response: ${response.status} ${response.statusText}` };
    }
    
    console.log(`Response from ${endpoint}:`, { status: response.status, data: responseData });
    
    return { 
      ok: response.ok, 
      data: responseData, 
      status: response.status 
    };
    
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
  getLiveUpdate: () => getUnauthenticated('get_live_update'),

  recordCarePlanInteraction: (interactionData: object) => post('care_plan/interaction', interactionData),
  getCarePlanProgress: () => get('care_plan/progress'),
  updateCarePlanProgress: (progressData: object) => post('care_plan/progress', progressData),
  canAccessCarePlanDay: (dayData: object) => post('care_plan/can_access_day', dayData),
  sendCarePlanNotification: (notificationData: object) => post('care_plan/notify', notificationData),
  
  // Use unauthenticated GET for wearable endpoints
  wearableConnect: () => getUnauthenticated('wearable/connect'),
  liveMonitor: () => post('live_monitor', {}),
   chatWithBot: (message: string) => post('chatbot', { message }),
};

export default ApiService;