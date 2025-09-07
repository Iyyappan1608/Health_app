import React, { useState, useMemo } from 'react';
import { View, Text, StyleSheet, ScrollView, TextInput, TouchableOpacity, Alert, Switch, ActivityIndicator, SafeAreaView } from 'react-native';
import { Stack, useRouter } from 'expo-router';
import { Colors } from '../constants/Colors';
import DashboardCard from '../components/DashboardCard';
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

export default function HypertensionCheckScreen() {
    const router = useRouter();
    const [isSubmitting, setIsSubmitting] = useState(false);

    // State for the hypertension form fields
    const [age, setAge] = useState('');
    const [sex, setSex] = useState(1); // 1 for Male, 0 for Female
    const [bmi, setBmi] = useState('');
    const [familyHistory, setFamilyHistory] = useState(false);
    const [creatinine, setCreatinine] = useState('');
    const [systolicBP, setSystolicBP] = useState('');
    const [diastolicBP, setDiastolicBP] = useState('');

    // --- NEW: FORM VALIDATION LOGIC ---
    const isFormValid = useMemo(() => {
        const requiredFields = [age, bmi, creatinine, systolicBP, diastolicBP];
        return requiredFields.every(field => field.trim() !== '');
    }, [age, bmi, creatinine, systolicBP, diastolicBP]);

    const handleSubmit = async () => {
        if (!isFormValid) {
            Alert.alert("Missing Data", "Please fill in all fields.");
            return;
        }

        setIsSubmitting(true);
        const dataForModel = {
            'age': parseInt(age) || 0,
            'sex': sex,
            'bmi': parseFloat(bmi) || 0,
            'family_history': familyHistory ? 1 : 0,
            'creatinine': parseFloat(creatinine) || 0,
            'systolic_bp': parseInt(systolicBP) || 0,
            'diastolic_bp': parseInt(diastolicBP) || 0
        };

        try {
            const response = await ApiService.predictHypertension(dataForModel);
            if (!response.ok) throw new Error(response.data.error || 'Prediction failed');
            
            const result = response.data;
            router.push({
                pathname: '/hypertension-report',
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
            <Stack.Screen options={{ presentation: 'modal', title: 'Hypertension Analysis' }} />
            <ScrollView contentContainerStyle={styles.contentContainer}>
                <Text style={styles.headerTitle}>Hypertension Analysis</Text>

                <DashboardCard icon='person-outline' title='Patient Details'>
                    <Text style={styles.label}>Age</Text>
                    <TextInput style={styles.input} value={age} onChangeText={setAge} keyboardType="numeric" placeholder="e.g., 55" />
                    <Text style={styles.label}>Sex</Text>
                    <View style={styles.choiceContainer}>
                        <ChoiceButton label="Male" value={1} selectedValue={sex} onSelect={setSex} />
                        <ChoiceButton label="Female" value={0} selectedValue={sex} onSelect={setSex} />
                    </View>
                    <Text style={styles.label}>BMI</Text>
                    <TextInput style={styles.input} value={bmi} onChangeText={setBmi} keyboardType="numeric" placeholder="e.g., 28.7"/>
                    <View style={styles.switchContainer}>
                        <Text style={styles.label}>Family History of HTN?</Text>
                        <Switch value={familyHistory} onValueChange={setFamilyHistory} />
                    </View>
                </DashboardCard>
                
                <DashboardCard icon='pulse-outline' title='Vitals & Labs'>
                     <Text style={styles.label}>Systolic BP (mmHg)</Text>
                    <TextInput style={styles.input} value={systolicBP} onChangeText={setSystolicBP} keyboardType="numeric" placeholder="e.g., 145" />
                     <Text style={styles.label}>Diastolic BP (mmHg)</Text>
                    <TextInput style={styles.input} value={diastolicBP} onChangeText={setDiastolicBP} keyboardType="numeric" placeholder="e.g., 92" />
                    <Text style={styles.label}>Creatinine (mg/dL)</Text>
                    <TextInput style={styles.input} value={creatinine} onChangeText={setCreatinine} keyboardType="numeric" placeholder="e.g., 1.1" />
                </DashboardCard>

                {/* --- UPDATED: Button is now disabled and styled based on isFormValid --- */}
                <TouchableOpacity 
                    style={[styles.saveButton, !isFormValid && styles.saveButtonDisabled]} 
                    onPress={handleSubmit} 
                    disabled={!isFormValid || isSubmitting}
                >
                    {isSubmitting ? <ActivityIndicator color="#FFFFFF" /> : <Text style={styles.saveButtonText}>Analyze Hypertension</Text>}
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
    choiceButtonText: { color: '#333', fontSize: 14, fontWeight: '600' },
    choiceButtonTextSelected: { color: '#FFFFFF' },
});