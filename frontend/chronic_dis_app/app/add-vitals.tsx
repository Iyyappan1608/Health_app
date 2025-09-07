import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import React, { useState } from 'react';
import {
    Alert,
    Platform,
    ScrollView,
    StatusBar,
    StyleSheet,
    Text,
    TextInput,
    TouchableOpacity,
    View,
} from 'react-native';
// CORRECT PATH
import { useData } from '../src/context/DataContext';

export default function AddVitalsScreen() {
  // Get the addVitals function from our global context
  const { addVitals } = useData(); 
  
  // State variables to hold the input from the user
  const [glucose, setGlucose] = useState('');
  const [systolic, setSystolic] = useState('');
  const [diastolic, setDiastolic] = useState('');
  const [heartRate, setHeartRate] = useState('');
  const [notes, setNotes] = useState('');

  const handleSave = () => {
    // Basic validation to ensure at least one field is filled
    if (!glucose && !systolic && !diastolic && !heartRate) {
      Alert.alert('Missing Data', 'Please enter at least one vital sign reading.');
      return;
    }
    
    // Call the function from our context to add the new data to our global store
    addVitals({
      glucose: glucose || null,
      bloodPressure: (systolic && diastolic) ? `${systolic}/${diastolic}` : null,
      heartRate: heartRate || null,
      notes: notes || null,
    });

    // Go back to the dashboard screen
    router.back();
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <StatusBar barStyle="dark-content" />

      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="close-circle" size={40} color="#AAA" />
        </TouchableOpacity>
        <Text style={styles.title}>Add New Reading</Text>
      </View>
      
      <Text style={styles.sectionTitle}>Vitals</Text>

      {/* Blood Glucose Input */}
      <Text style={styles.label}>Blood Glucose (mg/dL)</Text>
      <View style={styles.inputContainer}>
        <Ionicons name="water-outline" size={24} color="#9E9E9E" style={styles.inputIcon} />
        <TextInput
          style={styles.input}
          placeholder="e.g., 120"
          keyboardType="numeric"
          value={glucose}
          onChangeText={setGlucose}
        />
      </View>

      {/* Blood Pressure Input */}
      <Text style={styles.label}>Blood Pressure (mmHg)</Text>
      <View style={styles.bpContainer}>
        <View style={[styles.inputContainer, { flex: 1, marginRight: 10 }]}>
          <Ionicons name="pulse-outline" size={24} color="#9E9E9E" style={styles.inputIcon} />
          <TextInput
            style={styles.input}
            placeholder="Systolic (e.g., 120)"
            keyboardType="numeric"
            value={systolic}
            onChangeText={setSystolic}
          />
        </View>
        <View style={[styles.inputContainer, { flex: 1 }]}>
          <TextInput
            style={styles.input}
            placeholder="Diastolic (e.g., 80)"
            keyboardType="numeric"
            value={diastolic}
            onChangeText={setDiastolic}
          />
        </View>
      </View>
      
      {/* Heart Rate Input */}
      <Text style={styles.label}>Heart Rate (bpm)</Text>
      <View style={styles.inputContainer}>
        <Ionicons name="heart-outline" size={24} color="#9E9E9E" style={styles.inputIcon} />
        <TextInput
          style={styles.input}
          placeholder="e.g., 75"
          keyboardType="numeric"
          value={heartRate}
          onChangeText={setHeartRate}
        />
      </View>
      
      <Text style={styles.sectionTitle}>Lifestyle</Text>

      {/* Notes Input */}
      <Text style={styles.label}>Notes (Optional)</Text>
      <TextInput
        style={[styles.input, styles.notesInput]}
        placeholder="e.g., Feeling tired, post-lunch reading..."
        multiline
        value={notes}
        onChangeText={setNotes}
      />

      {/* Action Buttons */}
      <TouchableOpacity style={styles.saveButton} onPress={handleSave}>
        <Text style={styles.saveButtonText}>Save Reading</Text>
      </TouchableOpacity>

    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F0F4F8' },
  contentContainer: { padding: 20 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
    marginTop: Platform.OS === 'ios' ? 20 : 0,
  },
  backButton: {
    marginRight: 15,
  },
  title: { fontSize: 28, fontWeight: 'bold', color: '#333' },
  sectionTitle: { fontSize: 22, fontWeight: '600', color: '#333', marginTop: 20, marginBottom: 10, borderBottomWidth: 1, borderBottomColor: '#DDD', paddingBottom: 5 },
  label: { fontSize: 16, color: '#666', marginBottom: 8, marginLeft: 5 },
  inputContainer: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#FFFFFF', borderRadius: 10, height: 50, paddingHorizontal: 15, marginBottom: 15, borderWidth: 1, borderColor: '#E0E0E0' },
  inputIcon: { marginRight: 10 },
  input: { flex: 1, fontSize: 16, color: '#333' },
  bpContainer: { flexDirection: 'row', justifyContent: 'space-between' },
  notesInput: { height: 100, textAlignVertical: 'top', paddingVertical: 10, backgroundColor: '#FFFFFF', borderRadius: 10, borderWidth: 1, borderColor: '#E0E0E0', paddingHorizontal: 15 },
  saveButton: { backgroundColor: '#2ECC71', padding: 18, borderRadius: 15, alignItems: 'center', marginTop: 30 },
  saveButtonText: { color: '#FFFFFF', fontSize: 18, fontWeight: 'bold' },
});