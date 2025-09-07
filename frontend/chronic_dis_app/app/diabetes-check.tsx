import React, { useState, useMemo } from 'react';
import { View, Text, StyleSheet, ScrollView, TextInput, TouchableOpacity, Alert, Switch, ActivityIndicator, SafeAreaView } from 'react-native';
import { Stack, useRouter } from 'expo-router';
import { Colors } from '../constants/Colors';
import DashboardCard from '../components/DashboardCard';
import { Ionicons } from '@expo/vector-icons';
import ApiService from '../services/ApiService';

// Helper component for choice buttons
type ChoiceButtonProps = { label: string; value: any; selectedValue: any; onSelect: (value: any) => void; };
const ChoiceButton = ({ label, value, selectedValue, onSelect }: ChoiceButtonProps) => (
    <TouchableOpacity
      style={[styles.choiceButton, selectedValue === value && styles.choiceButtonSelected]}
      onPress={() => onSelect(value)}
    >
      <Text style={[styles.choiceButtonText, selectedValue === value && styles.choiceButtonTextSelected]}>{label}</Text>
    </TouchableOpacity>
);

export default function DiabetesCheckScreen() {
    const router = useRouter();
    const [isSubmitting, setIsSubmitting] = useState(false);

    // State for the form fields
    const [isPregnant, setIsPregnant] = useState(false);
    const [ageAtDiagnosis, setAgeAtDiagnosis] = useState('');
    const [bmiAtDiagnosis, setBmiAtDiagnosis] = useState('');
    const [familyHistory, setFamilyHistory] = useState('None');
    const [hba1c, setHba1c] = useState('');
    const [cPeptideLevel, setCPeptideLevel] = useState('');
    const [autoantibodiesStatus, setAutoantibodiesStatus] = useState('Negative');
    const [geneticTestResult, setGeneticTestResult] = useState('Negative');

    // --- NEW: FORM VALIDATION LOGIC ---
    const isFormValid = useMemo(() => {
        // List all text input fields that must be filled
        const requiredFields = [ageAtDiagnosis, bmiAtDiagnosis, hba1c, cPeptideLevel];
        // Check if every field is not empty
        return requiredFields.every(field => field.trim() !== '');
    }, [ageAtDiagnosis, bmiAtDiagnosis, hba1c, cPeptideLevel]);


    const handleSubmit = async () => {
        if (!isFormValid) {
            Alert.alert("Missing Data", "Please fill in all required fields.");
            return;
        }
        setIsSubmitting(true);
        const dataForModel = {
            'Is_Pregnant': isPregnant ? 1 : 0,
            'Age_at_Diagnosis': parseInt(ageAtDiagnosis),
            'BMI_at_Diagnosis': parseFloat(bmiAtDiagnosis),
            'Family_History': familyHistory,
            'HbA1c': parseFloat(hba1c),
            'C_Peptide_Level': parseFloat(cPeptideLevel),
            'Autoantibodies_Status': autoantibodiesStatus,
            'Genetic_Test_Result': geneticTestResult
        };

        
  try {
    const response = await ApiService.predictDiabetes(dataForModel); // or ApiService.predictHypertension
    if (!response.ok) throw new Error(response.data.error || response.data.message || 'Prediction failed');
    
    const result = response.data;
    router.push({
      pathname: '/diabetes-report', // or '/hypertension-report'
      params: { reportData: JSON.stringify(result) }
    });

  } catch (error: any) {
    Alert.alert('Error', `Could not get prediction. ${error.message}`);
  } finally {
    setIsSubmitting(false);
  }
    };

    return (
        <SafeAreaView style={styles.container}>
            <Stack.Screen options={{ presentation: 'modal', title: 'Diabetes Subtype Analysis' }} />
            <ScrollView contentContainerStyle={styles.contentContainer}>
                <Text style={styles.headerTitle}>Diabetes Subtype Analysis</Text>

                <DashboardCard icon='information-circle-outline' title='Patient Details'>
                    <View style={styles.switchContainer}><Text style={styles.label}>Is Pregnant?</Text><Switch value={isPregnant} onValueChange={setIsPregnant} /></View>
                    <Text style={styles.label}>Age at Diagnosis</Text><TextInput style={styles.input} value={ageAtDiagnosis} onChangeText={setAgeAtDiagnosis} keyboardType="numeric" placeholder="e.g., 45" />
                    <Text style={styles.label}>BMI at Diagnosis</Text><TextInput style={styles.input} value={bmiAtDiagnosis} onChangeText={setBmiAtDiagnosis} keyboardType="numeric" placeholder="e.g., 27.2" />
                </DashboardCard>

                <DashboardCard icon='people-outline' title='Genetic & Family Factors'>
                    <Text style={styles.label}>Family History</Text>
                    <View style={styles.choiceContainer}>
                        <ChoiceButton label="None" value="None" selectedValue={familyHistory} onSelect={setFamilyHistory} />
                        <ChoiceButton label="Parent/Sibling T1D" value="Parent/Sibling_T1D" selectedValue={familyHistory} onSelect={setFamilyHistory} />
                    </View>
                    <View style={styles.choiceContainer}>
                        <ChoiceButton label="Parent/Sibling T2D" value="Parent/Sibling_T2D" selectedValue={familyHistory} onSelect={setFamilyHistory} />
                        <ChoiceButton label="Strong Multi-Gen" value="Strong_Multi_Generational" selectedValue={familyHistory} onSelect={setFamilyHistory} />
                    </View>
                    
                    <Text style={styles.label}>Genetic Test Result</Text>
                    <View style={styles.choiceContainer}>
                        <ChoiceButton label="Negative" value="Negative" selectedValue={geneticTestResult} onSelect={setGeneticTestResult} />
                        <ChoiceButton label="Known MODY Mutation" value="Known_MODY_Mutation" selectedValue={geneticTestResult} onSelect={setGeneticTestResult} />
                    </View>
                </DashboardCard>

                <DashboardCard icon='beaker-outline' title='Lab Results'>
                    <Text style={styles.label}>HbA1c (%)</Text>
                    <TextInput style={styles.input} value={hba1c} onChangeText={setHba1c} keyboardType="numeric" placeholder="e.g., 8.5" />
                    <Text style={styles.label}>C-Peptide Level (ng/mL)</Text>
                    <TextInput style={styles.input} value={cPeptideLevel} onChangeText={setCPeptideLevel} keyboardType="numeric" placeholder="e.g., 0.6" />
                    <Text style={styles.label}>Autoantibodies Status</Text>
                    <View style={styles.choiceContainer}>
                        <ChoiceButton label="Negative" value="Negative" selectedValue={autoantibodiesStatus} onSelect={setAutoantibodiesStatus} />
                        <ChoiceButton label="GAD65 Positive" value="GAD65_Positive" selectedValue={autoantibodiesStatus} onSelect={setAutoantibodiesStatus} />
                    </View>
                     <View style={styles.choiceContainer}>
                        <ChoiceButton label="Multiple Positive" value="Multiple_Positive" selectedValue={autoantibodiesStatus} onSelect={setAutoantibodiesStatus} />
                    </View>
                </DashboardCard>

                {/* --- UPDATED: Button is now disabled and styled based on isFormValid --- */}
                <TouchableOpacity 
                    style={[styles.saveButton, !isFormValid && styles.saveButtonDisabled]} 
                    onPress={handleSubmit} 
                    disabled={!isFormValid || isSubmitting}
                >
                    {isSubmitting ? <ActivityIndicator color="#FFFFFF" /> : <Text style={styles.saveButtonText}>Analyze Diabetes Subtype</Text>}
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
    saveButton: { backgroundColor: Colors.primary, padding: 18, borderRadius: 15, alignItems: 'center', margin: 10, marginTop: 20 },
    // --- NEW: Style for the disabled button ---
    saveButtonDisabled: {
        backgroundColor: '#a9a9a9', // A gray color
    },
    saveButtonText: { color: '#FFFFFF', fontSize: 18, fontWeight: 'bold' },
    switchContainer: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 10 },
    choiceContainer: { flexDirection: 'row', marginBottom: 15, },
    choiceButton: { flex: 1, paddingVertical: 12, borderWidth: 1, borderColor: '#CCC', borderRadius: 8, alignItems: 'center', marginHorizontal: 5, },
    choiceButtonSelected: { backgroundColor: Colors.primary, borderColor: Colors.primary, },
    choiceButtonText: { color: '#333', fontSize: 14, fontWeight: '600', textAlign: 'center' },
    choiceButtonTextSelected: { color: '#FFFFFF' },
});