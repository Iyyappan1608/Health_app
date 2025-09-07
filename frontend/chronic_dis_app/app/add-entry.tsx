import React, { useState, useMemo } from 'react';
import { View, Text, StyleSheet, ScrollView, TextInput, TouchableOpacity, Alert, Switch, ActivityIndicator, SafeAreaView } from 'react-native';
import { Stack, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '../constants/Colors';
import DashboardCard from '../components/DashboardCard';
import { useData } from '../src/context/DataContext';
import ApiService from '../services/ApiService';

// Helper component for choice buttons
type ChoiceButtonProps = { label: string; value: string; selectedValue: string; onSelect: (value: string) => void; };
const ChoiceButton = ({ label, value, selectedValue, onSelect }: ChoiceButtonProps) => (
    <TouchableOpacity
      style={[styles.choiceButton, selectedValue === value && styles.choiceButtonSelected]}
      onPress={() => onSelect(value)}
    >
      <Text style={[styles.choiceButtonText, selectedValue === value && styles.choiceButtonTextSelected]}>{label}</Text>
    </TouchableOpacity>
);

type PredictedCondition = {
    disease: string;
    explanation: string;
}

export default function AddEntryScreen() {
    const router = useRouter();
    const { updatePredictions } = useData();
    const [isSubmitting, setIsSubmitting] = useState(false);

    // State for all form fields
    const [gender, setGender] = useState('Male');
    const [age, setAge] = useState('');
    const [bmi, setBmi] = useState('');
    const [smokingStatus, setSmokingStatus] = useState('Never');
    const [historyOfStroke, setHistoryOfStroke] = useState(false);
    const [systolicBP, setSystolicBP] = useState('');
    const [diastolicBP, setDiastolicBP] = useState('');
    const [heartRate, setHeartRate] = useState('');
    const [respiratoryRate, setRespiratoryRate] = useState('');
    const [fbs, setFbs] = useState('');
    const [hba1c, setHba1c] = useState('');
    const [serumCreatinine, setSerumCreatinine] = useState('');
    const [egfr, setEgfr] = useState('');
    const [bun, setBun] = useState('');
    const [totalCholesterol, setTotalCholesterol] = useState('');
    const [ldlCholesterol, setLdlCholesterol] = useState('');
    const [hdlCholesterol, setHdlCholesterol] = useState('');
    const [triglycerides, setTriglycerides] = useState('');
    const [hemoglobin, setHemoglobin] = useState('');
    const [urineAlbuminACR, setUrineAlbuminACR] = useState('');
    const [glucoseInUrine, setGlucoseInUrine] = useState(false);
    const [fev1FvcRatio, setFev1FvcRatio] = useState('');

    // --- NEW: FORM VALIDATION LOGIC ---
    const isFormValid = useMemo(() => {
        // List all text input fields that must be filled
        const requiredFields = [
            age, bmi, systolicBP, diastolicBP, heartRate, respiratoryRate,
            fbs, hba1c, serumCreatinine, egfr, bun, totalCholesterol,
            ldlCholesterol, hdlCholesterol, triglycerides, hemoglobin,
            urineAlbuminACR, fev1FvcRatio
        ];
        // Check if every field in the list is not empty
        return requiredFields.every(field => field.trim() !== '');
    }, [
        age, bmi, systolicBP, diastolicBP, heartRate, respiratoryRate, fbs, hba1c,
        serumCreatinine, egfr, bun, totalCholesterol, ldlCholesterol, hdlCholesterol,
        triglycerides, hemoglobin, urineAlbuminACR, fev1FvcRatio
    ]);


    const handleAnalysis = async () => {
        // This check is a fallback, but the button will be disabled anyway
        if (!isFormValid) {
            Alert.alert('Missing Data', 'Please fill in all mandatory fields.');
            return;
        }

        const patientDataForModel = {
            'Gender': gender, 'Age': parseInt(age) || 0, 'BMI': parseFloat(bmi) || 0,
            'Smoking_Status': smokingStatus, 'History_of_Stroke': historyOfStroke ? 1 : 0,
            'Systolic_BP': parseInt(systolicBP) || 0, 'Diastolic_BP': parseInt(diastolicBP) || 0,
            'Heart_Rate': parseInt(heartRate) || 0, 'Respiratory_Rate': parseInt(respiratoryRate) || 0,
            'FBS': parseFloat(fbs) || 0, 'HbA1c': parseFloat(hba1c) || 0,
            'Serum_Creatinine': parseFloat(serumCreatinine) || 0, 'eGFR': parseFloat(egfr) || 0,
            'BUN': parseFloat(bun) || 0, 'Total_Cholesterol': parseFloat(totalCholesterol) || 0,
            'LDL_Cholesterol': parseFloat(ldlCholesterol) || 0, 'HDL_Cholesterol': parseFloat(hdlCholesterol) || 0,
            'Triglycerides': parseFloat(triglycerides) || 0, 'Hemoglobin': parseFloat(hemoglobin) || 0,
            'Urine_Albumin_ACR': parseFloat(urineAlbuminACR) || 0, 'Glucose_in_Urine': glucoseInUrine ? 1 : 0,
            'FEV1_FVC_Ratio': parseFloat(fev1FvcRatio) || 0,
        };
        
        setIsSubmitting(true);
        try {
    const response = await ApiService.generateReport(patientDataForModel);
    if (!response.ok) { 
      throw new Error(response.data.message || response.data.error || 'Server error'); 
    }
    
    const report = response.data;
    const diseaseNames = report.predicted_conditions.map((p: PredictedCondition) => p.disease);
    if (diseaseNames) {
      updatePredictions(diseaseNames);
    }
    
    router.push({ pathname: '/report', params: { reportData: JSON.stringify(report) } });
  } catch (error: any) {
    console.error("API Call failed:", error);
    Alert.alert(
      "Analysis Failed", 
      error.message || "Could not connect to the analysis server. Please try again."
    );
  } finally {
    setIsSubmitting(false);
  }
    };

    return (
        <SafeAreaView style={styles.container}>
            <Stack.Screen options={{ presentation: 'modal', title: 'New Health Entry' }} />
            <ScrollView contentContainerStyle={styles.contentContainer}>
                <Text style={styles.headerTitle}>New Health Entry</Text>
                
                <DashboardCard icon="person-outline" title="Basic Information">
                    {/* ... (Your form fields are unchanged) ... */}
                    <Text style={styles.label}>Gender</Text>
                    <View style={styles.choiceContainer}>
                        <ChoiceButton label="Male" value="Male" selectedValue={gender} onSelect={setGender} />
                        <ChoiceButton label="Female" value="Female" selectedValue={gender} onSelect={setGender} />
                    </View>
                    <Text style={styles.label}>Age (years)</Text><TextInput style={styles.input} value={age} onChangeText={setAge} keyboardType="numeric" />
                    <Text style={styles.label}>BMI</Text><TextInput style={styles.input} value={bmi} onChangeText={setBmi} keyboardType="numeric" />
                    <Text style={styles.label}>Smoking Status</Text>
                    <View style={styles.choiceContainer}>
                        <ChoiceButton label="Never" value="Never" selectedValue={smokingStatus} onSelect={setSmokingStatus} />
                        <ChoiceButton label="Former" value="Former" selectedValue={smokingStatus} onSelect={setSmokingStatus} />
                        <ChoiceButton label="Current" value="Current" selectedValue={smokingStatus} onSelect={setSmokingStatus} />
                    </View>
                    <View style={styles.switchContainer}><Text style={styles.label}>History of Stroke?</Text><Switch value={historyOfStroke} onValueChange={setHistoryOfStroke} /></View>
                </DashboardCard>
                
                <DashboardCard icon="pulse-outline" title="Vitals">
                    <Text style={styles.label}>Blood Pressure (Systolic)</Text><TextInput style={styles.input} value={systolicBP} onChangeText={setSystolicBP} keyboardType="numeric" placeholder="e.g., 120" />
                    <Text style={styles.label}>Blood Pressure (Diastolic)</Text><TextInput style={styles.input} value={diastolicBP} onChangeText={setDiastolicBP} keyboardType="numeric" placeholder="e.g., 80" />
                    <Text style={styles.label}>Heart Rate (bpm)</Text><TextInput style={styles.input} value={heartRate} onChangeText={setHeartRate} keyboardType="numeric" placeholder="e.g., 72" />
                    <Text style={styles.label}>Respiratory Rate (breaths/min)</Text><TextInput style={styles.input} value={respiratoryRate} onChangeText={setRespiratoryRate} keyboardType="numeric" placeholder="e.g., 16" />
                </DashboardCard>
                
                <DashboardCard icon="beaker-outline" title="Lab Results">
                    <Text style={styles.label}>Fasting Blood Sugar (mg/dL)</Text><TextInput style={styles.input} value={fbs} onChangeText={setFbs} keyboardType="numeric" />
                    <Text style={styles.label}>HbA1c (%)</Text><TextInput style={styles.input} value={hba1c} onChangeText={setHba1c} keyboardType="numeric" />
                    <Text style={styles.label}>Serum Creatinine (mg/dL)</Text><TextInput style={styles.input} value={serumCreatinine} onChangeText={setSerumCreatinine} keyboardType="numeric" />
                    <Text style={styles.label}>eGFR (mL/min/1.73m²)</Text><TextInput style={styles.input} value={egfr} onChangeText={setEgfr} keyboardType="numeric" />
                    <Text style={styles.label}>Blood Urea Nitrogen (mg/dL)</Text><TextInput style={styles.input} value={bun} onChangeText={setBun} keyboardType="numeric" />
                    <Text style={styles.label}>Total Cholesterol (mg/dL)</Text><TextInput style={styles.input} value={totalCholesterol} onChangeText={setTotalCholesterol} keyboardType="numeric" />
                    <Text style={styles.label}>LDL Cholesterol (mg/dL)</Text><TextInput style={styles.input} value={ldlCholesterol} onChangeText={setLdlCholesterol} keyboardType="numeric" />
                    <Text style={styles.label}>HDL Cholesterol (mg/dL)</Text><TextInput style={styles.input} value={hdlCholesterol} onChangeText={setHdlCholesterol} keyboardType="numeric" />
                    <Text style={styles.label}>Triglycerides (mg/dL)</Text><TextInput style={styles.input} value={triglycerides} onChangeText={setTriglycerides} keyboardType="numeric" />
                    <Text style={styles.label}>Hemoglobin (g/dL)</Text><TextInput style={styles.input} value={hemoglobin} onChangeText={setHemoglobin} keyboardType="numeric" />
                    <Text style={styles.label}>Urine Albumin-to-Creatinine Ratio (mg/g)</Text><TextInput style={styles.input} value={urineAlbuminACR} onChangeText={setUrineAlbuminACR} keyboardType="numeric" />
                    <Text style={styles.label}>FEV1/FVC Ratio</Text><TextInput style={styles.input} value={fev1FvcRatio} onChangeText={setFev1FvcRatio} keyboardType="numeric" placeholder="e.g., 0.80" />
                    <View style={styles.switchContainer}><Text style={styles.label}>Glucose present in urine?</Text><Switch value={glucoseInUrine} onValueChange={setGlucoseInUrine} /></View>
                </DashboardCard>

                {/* --- UPDATED: Button is now disabled and styled based on isFormValid --- */}
                <TouchableOpacity
                  style={[styles.saveButton, !isFormValid && styles.saveButtonDisabled]}
                  onPress={handleAnalysis}
                  disabled={!isFormValid || isSubmitting}
                >
                    {isSubmitting ? ( <ActivityIndicator color="#FFFFFF" /> ) : (
                    <>
                        <Ionicons name="analytics-outline" size={24} color="#FFFFFF" />
                        <Text style={styles.saveButtonText}>Submit for Analysis</Text>
                    </>
                    )}
                </TouchableOpacity>
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#F0F4F8' },
    contentContainer: { padding: 10, paddingBottom: 40 },
    headerTitle: { fontSize: 28, fontWeight: 'bold', color: Colors.text, marginBottom: 20, paddingHorizontal: 10 },
    label: { fontSize: 16, color: '#666', marginBottom: 8, marginLeft: 5, fontWeight: '500' },
    input: { backgroundColor: Colors.surface, borderRadius: 10, padding: 15, fontSize: 16, marginBottom: 15, color: Colors.text, borderWidth: 1, borderColor: '#E0E0E0' },
    saveButton: { backgroundColor: Colors.primary, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', padding: 18, borderRadius: 15, margin: 10, marginTop: 20 },
    // --- NEW: Style for the disabled button ---
    saveButtonDisabled: {
        backgroundColor: '#a9a9a9', // A gray color
    },
    saveButtonText: { color: '#FFFFFF', fontSize: 18, fontWeight: 'bold', marginLeft: 10 },
    choiceContainer: { flexDirection: 'row', marginBottom: 15, },
    choiceButton: { flex: 1, paddingVertical: 12, borderWidth: 1, borderColor: '#CCC', borderRadius: 8, alignItems: 'center', marginHorizontal: 5, },
    choiceButtonSelected: { backgroundColor: Colors.primary, borderColor: Colors.primary, },
    choiceButtonText: { color: '#333', fontSize: 14, fontWeight: '600' },
    choiceButtonTextSelected: { color: '#FFFFFF' },
    switchContainer: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 10, }
});

