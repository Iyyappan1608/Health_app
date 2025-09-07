import React, { createContext, useState, useContext, ReactNode, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

export type VitalReading = { date: string; glucose: string | null; bloodPressure: string | null; heartRate: string | null; notes: string | null; };
export type CarePlanItem = { id: string; task: string; completed: boolean; icon: 'walk-outline' | 'medkit-outline' | 'restaurant-outline'; };

// --- This is the corrected, simplified UserData type ---
type UserData = {
  name: string;
  email: string;
  predictedConditions: string[]; // This is the ONLY property we need for predictions
  vitalHistory: VitalReading[];
  carePlan: CarePlanItem[];
};

type DataContextType = {
  userData: UserData | null;
  isLoading: boolean;
  addVitals: (newReading: Omit<VitalReading, 'date'>) => void;
  toggleCarePlanItem: (id: string) => void;
  updatePredictions: (predictions: string[]) => void; 
  updateUserName: (name: string) => void; // Added function to update user name
};

const DataContext = createContext<DataContextType | undefined>(undefined);
const STORAGE_KEY = '@HealthApp:userData';

export const DataProvider = ({ children }: { children: ReactNode }) => {
  const [userData, setUserData] = useState<UserData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        const jsonValue = await AsyncStorage.getItem(STORAGE_KEY);
        if (jsonValue !== null) {
          setUserData(JSON.parse(jsonValue));
        } else {
          // Default initial data
          const initialData: UserData = {
            name: 'User', // Default name
            email: 'user@example.com',
            predictedConditions: [], // Start with empty predictions
            vitalHistory: [],
            carePlan: [
              { id: '1', task: '30-minute morning walk', completed: false, icon: 'walk-outline' },
              { id: '2', task: 'Take Metformin (500mg)', completed: false, icon: 'medkit-outline' },
              { id: '3', task: 'Log your lunch meal', completed: false, icon: 'restaurant-outline' },
            ],
          };
          setUserData(initialData);
        }
      } catch (e) {
        console.error("Failed to load data from storage", e);
      } finally {
        setIsLoading(false);
      }
    };
    loadData();
  }, []);

  // Save data to AsyncStorage whenever it changes
  useEffect(() => {
    const saveData = async () => {
      if (userData && !isLoading) {
        try {
          const jsonValue = JSON.stringify(userData);
          await AsyncStorage.setItem(STORAGE_KEY, jsonValue);
        } catch (e) {
          console.error("Failed to save data to storage", e);
        }
      }
    };
    saveData();
  }, [userData, isLoading]);

  const addVitals = (newReading: Omit<VitalReading, 'date'>) => {
    if (userData) {
      const readingWithDate: VitalReading = {
        ...newReading,
        date: new Date().toISOString(),
      };
      
      setUserData(prevData => ({
        ...prevData!,
        vitalHistory: [readingWithDate, ...prevData!.vitalHistory],
      }));
    }
  };

  const toggleCarePlanItem = (id: string) => {
    if (userData) {
      setUserData(prevData => ({
        ...prevData!,
        carePlan: prevData!.carePlan.map(item =>
          item.id === id ? { ...item, completed: !item.completed } : item
        ),
      }));
    }
  };

  const updatePredictions = (predictions: string[]) => {
    if (userData) {
      setUserData(prevData => ({
        ...prevData!,
        predictedConditions: predictions,
      }));
    }
  };

  const updateUserName = (name: string) => {
    if (userData) {
      setUserData(prevData => ({
        ...prevData!,
        name: name,
      }));
    }
  };

  return (
    <DataContext.Provider value={{ 
      userData, 
      isLoading, 
      addVitals, 
      toggleCarePlanItem, 
      updatePredictions,
      updateUserName 
    }}>
      {children}
    </DataContext.Provider>
  );
};

export const useData = () => {
    const context = useContext(DataContext);
    if (context === undefined) {
        throw new Error('useData must be used within a DataProvider');
    }
    return context;
};