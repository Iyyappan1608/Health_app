import { Stack } from 'expo-router';
import { DataProvider } from '../src/context/DataContext';
import { Colors } from '../constants/Colors';

export default function RootLayout() {
  return (
    <DataProvider>
      <Stack
        screenOptions={{
          headerStyle: {
            backgroundColor: Colors.surface,
          },
          headerTintColor: Colors.text,
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        }}
      >
        {/* Main App Screens */}
        <Stack.Screen name="index" options={{ headerShown: false }} />
        <Stack.Screen name="welcome" options={{ headerShown: false }} />
        <Stack.Screen name="login" options={{ headerShown: false }} />
        <Stack.Screen name="signup" options={{ headerShown: false }} />
        <Stack.Screen name="doctor-login" options={{ headerShown: false }} />
        <Stack.Screen name="doctor-signup" options={{ headerShown: false }} />

        {/* Layout Groups */}
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen name="(doctor)" options={{ headerShown: false }} />
        
        {/* Modal Screens */}
        <Stack.Screen name="add-entry" options={{ presentation: 'modal', title: 'New Health Entry' }} />
        
        {/* THIS IS THE MISSING LINE */}
        <Stack.Screen name="report" options={{ presentation: 'modal', title: 'Analysis Report' }} />

        <Stack.Screen name="diabetes-check" options={{ presentation: 'modal', title: 'Diabetes Analysis' }} />
        <Stack.Screen name="hypertension-check" options={{ presentation: 'modal', title: 'Hypertension Analysis' }} />
        <Stack.Screen name="hypertension-report" options={{ presentation: 'modal', title: 'Hypertension Report' }} />
        
        {/* Standard Screen */}
        <Stack.Screen name="live-monitor" options={{ title: 'Live Monitor' }} />
        <Stack.Screen name="care-plan" options={{ presentation: 'modal', title: 'My Care Plan' }} />
        
      </Stack>
    </DataProvider>
  );
}