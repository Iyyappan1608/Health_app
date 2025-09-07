import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { Stack, useLocalSearchParams, useRouter, Link } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '../constants/Colors';
import DashboardCard from '../components/DashboardCard';

// Type definitions (remain the same)
type PredictedCondition = { disease: string; explanation: string; };
type RiskAssessment = { disease: string; risk_score: number; risk_level: 'Low' | 'Medium' | 'High'; primary_drivers: string; };
type Report = { predicted_conditions: PredictedCondition[]; risk_assessment: RiskAssessment[]; };

export default function ReportScreen() {
    const router = useRouter();
    const params = useLocalSearchParams();
    const { reportData } = params;

    const report: Report | null = typeof reportData === 'string' ? JSON.parse(reportData) as Report : null;

    if (!report) {
        return ( <View style={styles.container}><Text>Error: Could not display report.</Text></View> );
    }

    // --- THIS IS THE FIX ---
    // We make the check case-insensitive by converting the disease name to lowercase
     const hasDiabetes = report.predicted_conditions.some(condition => condition.disease.toLowerCase().includes('diabetes'));
const hasHypertension = report.predicted_conditions.some(condition => condition.disease.toLowerCase().includes('hypertension'));
    // --- END OF FIX ---

    console.log("Parsed Report:", report.predicted_conditions);
    console.log("Has Diabetes:", hasDiabetes);
    console.log("Has Hypertension:", hasHypertension);
    
    const getRiskColor = (level: 'Low' | 'Medium' | 'High') => {
        if (level === 'High') return Colors.danger;
        if (level === 'Medium') return '#F39C12';
        return Colors.accent;
    };

    return (
        <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
            <Stack.Screen options={{ title: 'AI Diagnostic Report' }} />
            <Text style={styles.headerTitle}>AI Diagnostic Report</Text>

            <DashboardCard icon="medkit-outline" title="Predicted Conditions">
                {report.predicted_conditions.map((condition, index) => (
                    <View key={index} style={styles.conditionItem}>
                        <Text style={styles.diseaseTitle}>{condition.disease}</Text>
                        <Text style={styles.explanationText}>{condition.explanation}</Text>
                    </View>
                ))}
            </DashboardCard>

            {report.risk_assessment.length > 0 && (
                <DashboardCard icon="shield-checkmark-outline" title="Detailed Risk Assessment">
                    {report.risk_assessment.map((risk, index) => (
                        <View key={index} style={styles.riskItem}>
                            <Text style={styles.diseaseTitle}>{risk.disease} Risk</Text>
                            <View style={styles.riskScoreContainer}>
                                <Text style={[styles.riskScore, { color: getRiskColor(risk.risk_level) }]}>
                                    {risk.risk_score}/100
                                </Text>
                                <Text style={[styles.riskLevel, { color: getRiskColor(risk.risk_level) }]}>
                                    ({risk.risk_level})
                                </Text>
                            </View>
                            <Text style={styles.explanationText}>
                                <Text style={{fontWeight: 'bold'}}>Primary Drivers:</Text> {risk.primary_drivers}
                            </Text>
                        </View>
                    ))}
                </DashboardCard>
            )}
            
            {(hasDiabetes || hasHypertension) && (
                <DashboardCard icon="stats-chart-outline" title="Next Steps: Drill-down Analysis">
                     {hasDiabetes && (
                        <Link href="/diabetes-check" asChild>
                            <TouchableOpacity style={styles.drilldownButton}>
                                <Text style={styles.drilldownButtonText}>Check for Diabetes Subtype</Text>
                                <Ionicons name="chevron-forward-outline" size={20} color="#FFFFFF" />
                            </TouchableOpacity>
                        </Link>
                    )} 
                    {hasHypertension && (
                        <Link href="/hypertension-check" asChild>
                            <TouchableOpacity style={styles.drilldownButton}>
                                <Text style={styles.drilldownButtonText}>Check for Hypertension Stage</Text>
                                <Ionicons name="chevron-forward-outline" size={20} color="#FFFFFF" />
                            </TouchableOpacity>
                        </Link>
                    )}
                </DashboardCard>
            )}

            <Text style={styles.disclaimer}>
                Disclaimer: This is an AI-generated prediction and not a substitute for professional medical advice.
            </Text>

            <TouchableOpacity style={styles.doneButton} onPress={() => router.back()}>
                <Text style={styles.doneButtonText}>Done</Text>
            </TouchableOpacity>
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#F0F4F8' },
    contentContainer: { padding: 20, paddingBottom: 40 },
    headerTitle: { fontSize: 28, fontWeight: 'bold', color: Colors.text, marginBottom: 20 },
    conditionItem: { marginBottom: 15 },
    diseaseTitle: { fontSize: 18, fontWeight: 'bold', color: Colors.primary, marginBottom: 5 },
    explanationText: { fontSize: 15, color: Colors.textSecondary, lineHeight: 22 },
    riskItem: { marginBottom: 20, borderTopWidth: 1, borderTopColor: '#EEE', paddingTop: 15 },
    riskScoreContainer: { flexDirection: 'row', alignItems: 'baseline', marginBottom: 5 },
    riskScore: { fontSize: 24, fontWeight: 'bold' },
    riskLevel: { fontSize: 16, fontWeight: '600', marginLeft: 8 },
    disclaimer: { fontSize: 12, color: '#999', textAlign: 'center', marginVertical: 20, fontStyle: 'italic' },
    doneButton: { backgroundColor: Colors.primary, padding: 18, borderRadius: 15, alignItems: 'center' },
    doneButtonText: { color: '#FFFFFF', fontSize: 18, fontWeight: 'bold' },
    drilldownButton: { backgroundColor: '#5D6D7E', padding: 15, borderRadius: 10, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',marginVertical: 5 },
    drilldownButtonText: { color: '#FFFFFF', fontSize: 16, fontWeight: '600' },
}); 