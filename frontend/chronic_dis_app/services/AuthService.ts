import AsyncStorage from '@react-native-async-storage/async-storage';

const TOKEN_KEY = 'session_token';
const USER_NAME_KEY = 'user_name';
const PATIENT_ID_KEY = 'patient_id';

const AuthService = {
  storeAuthData: async (token: string, name: string, patientId: any): Promise<void> => {
    try {
      await AsyncStorage.setItem(TOKEN_KEY, token);
      await AsyncStorage.setItem(USER_NAME_KEY, name);
      await AsyncStorage.setItem(PATIENT_ID_KEY, String(patientId));
    } catch (e) {
      console.error('Failed to store auth data.', e);
    }
  },

  // --- NEW FUNCTION TO GET DATA ---
  getAuthData: async (): Promise<{ token: string; name: string; patientId: string } | null> => {
    try {
      const token = await AsyncStorage.getItem(TOKEN_KEY);
      const name = await AsyncStorage.getItem(USER_NAME_KEY);
      const patientId = await AsyncStorage.getItem(PATIENT_ID_KEY);

      if (token && name && patientId) {
        return { token, name, patientId };
      }
      return null;
    } catch (e) {
      console.error('Failed to retrieve auth data.', e);
      return null;
    }
  },

  clearAuthData: async (): Promise<void> => {
    try {
      await AsyncStorage.removeItem(TOKEN_KEY);
      await AsyncStorage.removeItem(USER_NAME_KEY);
      await AsyncStorage.removeItem(PATIENT_ID_KEY);
    } catch (e) {
      console.error('Failed to clear auth data.', e);
    }
  },
};

export default AuthService;