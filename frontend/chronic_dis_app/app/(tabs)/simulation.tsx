import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import Slider from '@react-native-community/slider';
import { Ionicons } from '@expo/vector-icons';
import DashboardCard from '../../components/DashboardCard';
import AnimatedCard from '../../components/AnimatedCard';

export default function SimulationScreen() {
  // State to hold the user's choices
  const [medicationAdherence, setMedicationAdherence] = useState(90);
  const [exerciseLevel, setExerciseLevel] = useState(2); // 1=Low, 2=Medium, 3=High
  const [dietQuality, setDietQuality] = useState(2); // 1=Poor, 2=Good, 3=Excellent

  // State for the fake predicted outcome
  const [predictedRisk, setPredictedRisk] = useState('');
  const [riskColor, setRiskColor] = useState('#2ECC71');

  // This effect runs whenever a slider changes to update the prediction
  useEffect(() => {
    const score = medicationAdherence + (exerciseLevel * 10) + (dietQuality * 10);
    if (score > 120) {
      setPredictedRisk('Low Risk');
      setRiskColor('#2ECC71'); // Green
    } else if (score > 100) {
      setPredictedRisk('Medium Risk');
      setRiskColor('#F1C40F'); // Yellow
    } else {
      setPredictedRisk('High Risk');
      setRiskColor('#E74C3C'); // Red
    }
  }, [medicationAdherence, exerciseLevel, dietQuality]);

  const exerciseLabels = ['Low', 'Medium', 'High'];
  const dietLabels = ['Poor', 'Good', 'Excellent'];

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <Text style={styles.headerTitle}>Digital Twin Simulation</Text>
      <Text style={styles.headerSubtitle}>See how your choices could impact your future health.</Text>

      <AnimatedCard index={0}>
        <DashboardCard icon="medkit-outline" title="Medication Adherence">
          <Text style={styles.sliderValue}>{medicationAdherence.toFixed(0)}%</Text>
          <Slider
            style={{ width: '100%', height: 40 }}
            minimumValue={0}
            maximumValue={100}
            step={5}
            value={medicationAdherence}
            onValueChange={setMedicationAdherence}
            minimumTrackTintColor="#4A90E2"
            maximumTrackTintColor="#d3d3d3"
            thumbTintColor="#4A90E2"
          />
        </DashboardCard>
      </AnimatedCard>

      <AnimatedCard index={1}>
        <DashboardCard icon="walk-outline" title="Weekly Exercise Level">
          <Text style={styles.sliderValue}>{exerciseLabels[exerciseLevel - 1]}</Text>
          <Slider
            style={{ width: '100%', height: 40 }}
            minimumValue={1}
            maximumValue={3}
            step={1}
            value={exerciseLevel}
            onValueChange={setExerciseLevel}
            minimumTrackTintColor="#2ECC71"
            maximumTrackTintColor="#d3d3d3"
            thumbTintColor="#2ECC71"
          />
        </DashboardCard>
      </AnimatedCard>
      
      <AnimatedCard index={2}>
        <DashboardCard icon="restaurant-outline" title="Overall Diet Quality">
           <Text style={styles.sliderValue}>{dietLabels[dietQuality - 1]}</Text>
          <Slider
            style={{ width: '100%', height: 40 }}
            minimumValue={1}
            maximumValue={3}
            step={1}
            value={dietQuality}
            onValueChange={setDietQuality}
            minimumTrackTintColor="#F39C12"
            maximumTrackTintColor="#d3d3d3"
            thumbTintColor="#F39C12"
          />
        </DashboardCard>
      </AnimatedCard>

      <AnimatedCard index={3}>
        <DashboardCard icon="analytics-outline" title="Predicted 3-Month Complication Risk">
            <View style={styles.predictionContainer}>
                <Ionicons name="shield-checkmark" size={40} color={riskColor} />
                <Text style={[styles.predictionText, { color: riskColor }]}>{predictedRisk}</Text>
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
  sliderValue: {
    fontSize: 18,
    fontWeight: '600',
    textAlign: 'center',
    marginBottom: 10,
    color: '#333'
  },
  predictionContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  predictionText: {
      fontSize: 24,
      fontWeight: 'bold',
      marginLeft: 15,
  }
});