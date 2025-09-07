import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, StatusBar, ActivityIndicator } from 'react-native';
import { Stack } from 'expo-router';
import { Colors } from '../constants/Colors';
import { Ionicons } from '@expo/vector-icons';

// --- TYPE DEFINITIONS ---
type StatCardProps = { icon: keyof typeof Ionicons.glyphMap; label: string; value: string | number; unit: string; color: string; };
type RemedyItemProps = { icon: keyof typeof Ionicons.glyphMap; text: string; };
type Analysis = { Disease: string; Remedies: string[]; };
type Vitals = { glucose_level: number; heart_rate: number; systolic_bp: number; diastolic_bp: number; steps_per_min: number; stress_level: number; sleep_hours: number; };

// --- STANDALONE SIMULATED DATA ---
const simulatedCases = [
    {
        "caseName": "Stable Condition",
        "vitals": { "glucose_level": 95, "heart_rate": 72, "systolic_bp": 118, "diastolic_bp": 78, "steps_per_min": 25, "stress_level": 2, "sleep_hours": 7.8 },
        "analysis": { "Disease": "Normal", "Remedies": ["Maintain current diet and exercise routine.", "Ensure you get at least 7-8 hours of sleep.", "Stay hydrated throughout the day."]}
    },
    {
        "caseName": "High Glucose & Stress",
        "vitals": { "glucose_level": 185, "heart_rate": 95, "systolic_bp": 135, "diastolic_bp": 88, "steps_per_min": 10, "stress_level": 8, "sleep_hours": 5.2 },
        "analysis": { "Disease": "Hyperglycemia & High Stress", "Remedies": ["Consider a 15-minute brisk walk to help lower blood sugar.", "Practice deep breathing exercises to reduce acute stress.", "Avoid sugary drinks and refined carbohydrates for your next meal."]}
    },
    {
        "caseName": "Pre-Hypertension",
        "vitals": { "glucose_level": 110, "heart_rate": 88, "systolic_bp": 138, "diastolic_bp": 85, "steps_per_min": 5, "stress_level": 6, "sleep_hours": 6.5 },
        "analysis": { "Disease": "Elevated Blood Pressure", "Remedies": ["Reduce sodium intake for the remainder of the day.", "Incorporate potassium-rich foods like bananas or spinach.", "Monitor blood pressure again in a few hours."]}
    },
    {
        "caseName": "Hypoglycemia Risk",
        "vitals": { "glucose_level": 65, "heart_rate": 85, "systolic_bp": 110, "diastolic_bp": 70, "steps_per_min": 15, "stress_level": 5, "sleep_hours": 7.1 },
        "analysis": { "Disease": "Hypoglycemia Risk", "Remedies": ["Consume 15 grams of fast-acting carbohydrates (e.g., juice or glucose tablets).", "Re-check blood sugar in 15 minutes."]}
    },
    {
        "caseName": "Poor Sleep & Sedentary",
        "vitals": { "glucose_level": 120, "heart_rate": 80, "systolic_bp": 125, "diastolic_bp": 82, "steps_per_min": 2, "stress_level": 7, "sleep_hours": 4.5 },
        "analysis": { "Disease": "Lifestyle Risk Factors", "Remedies": ["Prioritize a consistent sleep schedule, aiming for 7+ hours.", "Incorporate at least 30 minutes of moderate activity, like walking."]}
    }
];

// --- UI HELPER COMPONENTS ---
const StatCard = ({ icon, label, value, unit, color }: StatCardProps) => ( <View style={styles.statCard}><View style={{flexDirection: 'row', alignItems: 'center', marginBottom: 8}}><Ionicons name={icon} size={20} color={color} /><Text style={[styles.statLabel, {color}]}>{label}</Text></View><Text style={[styles.statValue, {color}]}>{value}</Text><Text style={[styles.statUnit, {color: Colors.textSecondary}]}>{unit}</Text></View> );
const RemedyItem = ({ icon, text }: RemedyItemProps) => ( <View style={[styles.remedyItem, { borderLeftColor: Colors.primary }]}><Ionicons name={icon} size={24} color={Colors.primary} style={{marginRight: 15}} /><Text style={styles.remedyAction}>{text}</Text></View> );

export default function LiveMonitorScreen() {
    const [isConnected, setIsConnected] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [currentCaseName, setCurrentCaseName] = useState('N/A');
    const [vitals, setVitals] = useState<Vitals | null>(null);
    const [analysis, setAnalysis] = useState<Analysis | null>(null);

    // This function now works offline by picking a random case from the list above.
    const fetchLiveUpdate = useCallback(() => {
        setIsLoading(true);
        // Simulate a small network delay for better user experience
        setTimeout(() => {
            const result = simulatedCases[Math.floor(Math.random() * simulatedCases.length)];
            setVitals(result.vitals);
            setAnalysis(result.analysis);
            setCurrentCaseName(result.caseName);
            setIsLoading(false);
        }, 500); // 0.5 second delay
    }, []);

    // The useEffect hook for the timer has been removed.

    const handleConnectToggle = () => {
        const newConnectionState = !isConnected;
        setIsConnected(newConnectionState);
        
        if (newConnectionState) {
            // Data will only be fetched here, once, when the button is pressed.
            fetchLiveUpdate();
        } else {
            // Reset everything when disconnecting
            setVitals(null);
            setAnalysis(null);
            setCurrentCaseName('N/A');
        }
    };

    return (
        <ScrollView style={styles.container}>
            <Stack.Screen options={{ title: 'Live Monitor' }} />
            <StatusBar barStyle="dark-content" />
            <View style={styles.header}>
                <View>
                    <Text style={styles.headerTitle}>Live Health Monitor</Text>
                    <Text style={styles.headerSubtitle}>Real-time AI health analysis</Text>
                </View>
                <TouchableOpacity onPress={handleConnectToggle} style={[styles.connectButton, isConnected ? styles.connected : styles.disconnected]}>
                    <Ionicons name="watch-outline" size={20} color={isConnected ? Colors.accent : Colors.textSecondary} />
                    <Text style={[styles.connectButtonText, isConnected ? {color: Colors.accent} : {color: Colors.textSecondary}]}>{isConnected ? 'Connected' : 'Connect'}</Text>
                </TouchableOpacity>
            </View>
            <View style={styles.card}>
                <View style={styles.cardHeader}>
                    <Text style={styles.cardTitle}>Live Vitals (Case: {currentCaseName})</Text>
                    {isLoading && <ActivityIndicator size="small" color={Colors.primary} />}
                </View>
                {vitals ? (
                    <View style={styles.statsGrid}>
                        <StatCard icon="water-outline" label="Glucose" value={vitals.glucose_level.toFixed(0)} unit="mg/dL" color={vitals.glucose_level > 180 || vitals.glucose_level < 70 ? Colors.danger : vitals.glucose_level > 140 ? '#F39C12' : Colors.accent} />
                        <StatCard icon="heart-outline" label="Heart Rate" value={vitals.heart_rate.toFixed(0)} unit="bpm" color={Colors.primary} />
                        <StatCard icon="pulse-outline" label="Blood Pressure" value={`${vitals.systolic_bp.toFixed(0)}/${vitals.diastolic_bp.toFixed(0)}`} unit="mmHg" color={vitals.systolic_bp > 130 ? '#F39C12' : Colors.primary} />
                        <StatCard icon="walk-outline" label="Activity" value={vitals.steps_per_min.toFixed(0)} unit="steps/min" color={Colors.accent} />
                        <StatCard icon="moon-outline" label="Sleep" value={vitals.sleep_hours.toFixed(1)} unit="hours" color={vitals.sleep_hours < 6 ? '#F39C12' : Colors.primary} />
                        <StatCard icon="leaf-outline" label="Stress Level" value={vitals.stress_level.toFixed(0)} unit="/ 10" color={vitals.stress_level > 7 ? Colors.danger : Colors.accent} />
                    </View>
                ) : <Text style={styles.centeredText}>{isConnected ? "Loading vitals..." : "Connect to view vitals."}</Text>}
            </View>
            <View style={styles.card}>
                <View style={styles.cardHeader}>
                    <Text style={styles.cardTitle}>AI Smart Recommendations</Text>
                </View>
                {analysis ? (
                    <>
                        <View style={styles.predictionSummary}>
                            <Text style={styles.predictionLabel}>Condition:</Text>
                            <Text style={styles.predictionValue}>{analysis.Disease}</Text>
                        </View>
                        {analysis.Remedies.map((remedy, index) => (
                           <RemedyItem key={index} text={remedy} icon="medkit-outline" />
                        ))}
                    </>
                ) : (
                    <View style={styles.allGoodContainer}>
                       <Ionicons name={"information-circle-outline"} size={24} color={Colors.textSecondary} />
                       <Text style={styles.allGoodText}>{isConnected ? "Awaiting analysis..." : "Connect to a device to begin."}</Text>
                    </View>
                )}
            </View>
        </ScrollView>
    );
}

// --- STYLES ---
const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: Colors.background, },
    header: { padding: 20, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
    headerTitle: { fontSize: 24, fontWeight: 'bold', color: Colors.text },
    headerSubtitle: { fontSize: 14, color: Colors.textSecondary },
    connectButton: { flexDirection: 'row', alignItems: 'center', paddingVertical: 8, paddingHorizontal: 12, borderRadius: 10 },
    connected: { backgroundColor: '#E8F5E9' },
    disconnected: { backgroundColor: '#F5F5F5' },
    connectButtonText: { fontWeight: '600', marginLeft: 8 },
    card: { backgroundColor: Colors.surface, marginHorizontal: 15, marginBottom: 15, borderRadius: 15, padding: 15, shadowColor: "#000", shadowOffset: { width: 0, height: 2, }, shadowOpacity: 0.1, shadowRadius: 8, elevation: 5, },
    cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15, },
    cardTitle: { fontSize: 18, fontWeight: 'bold', color: Colors.text },
    statsGrid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between' },
    statCard: { width: '48%', backgroundColor: '#F8F9FA', padding: 15, borderRadius: 10, marginBottom: 10, },
    statLabel: { fontSize: 14, fontWeight: '500', marginLeft: 5, },
    statValue: { fontSize: 24, fontWeight: 'bold', marginTop: 5 },
    statUnit: { fontSize: 12, },
    remedyItem: { backgroundColor: '#F8F9FA', padding: 15, borderRadius: 10, marginBottom: 10, borderLeftWidth: 5, flexDirection: 'row', alignItems: 'center'},
    remedyAction: { fontSize: 14, color: Colors.textSecondary, flex: 1, },
    allGoodContainer: { flexDirection: 'row', alignItems: 'center', padding: 10, },
    allGoodText: { fontSize: 16, fontWeight: '600', color: Colors.textSecondary, marginLeft: 10, },
    centeredText: { textAlign: 'center', color: Colors.textSecondary, paddingVertical: 20 },
    predictionSummary: { flexDirection: 'row', alignItems: 'center', marginBottom: 15, padding: 10, backgroundColor: '#F8F9FA', borderRadius: 10, flexWrap: 'wrap' },
    predictionLabel: { fontSize: 16, color: Colors.textSecondary, marginRight: 5 },
    predictionValue: { fontSize: 16, fontWeight: 'bold', color: Colors.text, marginRight: 15 }
});