import { Ionicons } from '@expo/vector-icons';
import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import AnimatedCard from '../../components/AnimatedCard';
import DashboardCard from '../../components/DashboardCard';
import VitalsChart from '../../components/VitalsChart';
import { useData } from '../../src/context/DataContext';

export default function ForecastScreen() {
  const { userData } = useData();

  // --- MOCK FORECAST DATA ---
  const mockForecast = {
    riskReductionPercentage: 15,
    improvementTimeline: '3 Months',
    nextCheckupDate: 'December 15, 2025',
  };

  // Create a projected trend by taking the user's current data 
  // and adding some fake future data points that show improvement.
  const currentGlucoseData = userData?.vitalHistory
    .map(reading => reading.glucose ? parseInt(reading.glucose, 10) : null)
    .filter((value): value is number => value !== null && !isNaN(value))
    .reverse() ?? [];
  
  const lastReading = currentGlucoseData.length > 0 ? currentGlucoseData[currentGlucoseData.length - 1] : 140;

  const projectedDataPoints = [
      ...currentGlucoseData,
      Math.round(lastReading * 0.95), // Simulate 5% improvement
      Math.round(lastReading * 0.90), // Simulate 10% improvement
      Math.round(lastReading * 0.88), // Simulate 12% improvement
  ];
  
  const chartLabels = projectedDataPoints.map((_, index) => `${index + 1}`);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <Text style={styles.headerTitle}>Health Forecast</Text>
      <Text style={styles.headerSubtitle}>Based on your current data and adherence.</Text>

      <AnimatedCard index={0}>
        <DashboardCard icon="trending-down-outline" title="Expected Risk Reduction">
          <View style={styles.metricContainer}>
            <Text style={styles.metricValue}>{mockForecast.riskReductionPercentage}%</Text>
            <Text style={styles.metricLabel}>in the next {mockForecast.improvementTimeline}</Text>
          </View>
        </DashboardCard>
      </AnimatedCard>

      <AnimatedCard index={1}>
        <DashboardCard icon="analytics-outline" title="Projected Glucose Trend">
           <VitalsChart labels={chartLabels} data={projectedDataPoints} />
           <Text style={styles.chartNote}>Chart includes your current readings and a future projection.</Text>
        </DashboardCard>
      </AnimatedCard>

      <AnimatedCard index={2}>
        <DashboardCard icon="calendar-outline" title="Next Suggested Check-in">
           <View style={styles.metricContainer}>
            <Ionicons name="medkit" size={32} color="#4A90E2" />
            <Text style={styles.checkupText}>{mockForecast.nextCheckupDate}</Text>
          </View>
        </DashboardCard>
      </AnimatedCard>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F0F4F8' },
  contentContainer: { padding: 20 },
  headerTitle: { fontSize: 28, fontWeight: 'bold', color: '#333', marginTop: 20 },
  headerSubtitle: { fontSize: 16, color: '#666', marginBottom: 20 },
  metricContainer: {
    alignItems: 'center',
    padding: 10,
  },
  metricValue: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#2ECC71',
  },
  metricLabel: {
    fontSize: 16,
    color: '#666',
    marginTop: 5,
  },
  chartNote: {
      fontSize: 12,
      color: '#888',
      textAlign: 'center',
      marginTop: 10,
      fontStyle: 'italic',
  },
  checkupText: {
      fontSize: 22,
      fontWeight: '600',
      color: '#333',
      marginLeft: 15,
  }
});