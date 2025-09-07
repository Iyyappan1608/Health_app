import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

// This defines what "props" (properties) our component can accept.
type DashboardCardProps = {
  title: string;
  // This clever TypeScript type ensures we can only pass valid Ionicons icon names.
  icon: keyof typeof Ionicons.glyphMap; 
  iconColor?: string; // The '?' makes this property optional.
  children: React.ReactNode; // 'children' allows us to place other components inside this one.
};

const DashboardCard = ({ title, icon, iconColor = "#4A90E2", children }: DashboardCardProps) => {
  return (
    <View style={styles.cardContainer}>
      <View style={styles.cardHeader}>
        <Ionicons name={icon} size={24} color={iconColor} />
        <Text style={styles.cardTitle}>{title}</Text>
      </View>
      <View style={styles.cardContent}>
        {children}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  cardContainer: {
    backgroundColor: '#FFFFFF',
    borderRadius: 15,
    padding: 20,
    marginBottom: 20,
    shadowColor: "#000",
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#F0F4F8',
    paddingBottom: 10,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginLeft: 10,
  },
  cardContent: {
    // The content inside the card is styled by the parent component (your dashboard screen)
  },
});

export default DashboardCard;