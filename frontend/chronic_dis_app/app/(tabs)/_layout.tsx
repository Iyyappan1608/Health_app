import { Ionicons } from '@expo/vector-icons';
import { Tabs } from 'expo-router';
import React from 'react';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors } from '../../constants/Colors';

export default function TabLayout() {
  const { bottom } = useSafeAreaInsets();

  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: Colors.primary,
        tabBarInactiveTintColor: Colors.inactive,
        tabBarStyle: {
          height: 60 + bottom, // Using a more standard height
          paddingBottom: 5 + bottom,
          paddingTop: 5,
          backgroundColor: Colors.surface,
          borderTopWidth: 1,
          borderTopColor: '#E0E0E0'
        },
      }}>
      <Tabs.Screen
        name="index"
        options={{
          title: 'Dashboard',
          headerShown: true, // <-- THIS IS THE KEY CHANGE
          tabBarIcon: ({ color }) => <Ionicons size={28} name="home-outline" color={color} />,
        }}
      />
      <Tabs.Screen
        name="care-plan"
        options={{
          title: 'Care Plan',
          headerShown: false,
          tabBarIcon: ({ color }) => <Ionicons size={28} name="list-outline" color={color} />,
        }}
      />
      <Tabs.Screen
        name="live-monitor"
        options={{
          title: 'Live Monitor',
          headerShown: false,
          tabBarIcon: ({ color }) => <Ionicons size={28} name="watch-outline" color={color} />,
        }}
      />
      <Tabs.Screen
        name="profile"
        options={{
          title: 'Profile',
          headerShown: false,
          tabBarIcon: ({ color }) => <Ionicons size={28} name="person-outline" color={color} />,
        }}
      />
       {/* Hiding these screens from the tab bar */}
      <Tabs.Screen name="simulation" options={{ href: null }} />
      <Tabs.Screen name="forecast" options={{ href: null }} />
    </Tabs>
  );
}