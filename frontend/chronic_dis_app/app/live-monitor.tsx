import React, { useState, useCallback, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, StatusBar, ActivityIndicator, Alert } from 'react-native';
import { Stack } from 'expo-router';
import { Colors } from '../constants/Colors';
import { Ionicons } from '@expo/vector-icons';
import ApiService from '../services/ApiService';

// --- TYPE DEFINITIONS ---
type StatCardProps = { icon: keyof typeof Ionicons.glyphMap; label: string; value: string | number; unit: string; color: string; };
type Analysis = { 
  Disease: string; 
  Remedies: string[];
  Severity?: string;
};
type Vitals = { 
  glucose_level: number; 
  heart_rate: number; 
  systolic_bp: number; 
  diastolic_bp: number; 
  steps_per_min: number; 
  stress_level: number; 
  sleep_hours: number; 
};
type ApiResponse = {
    input_data: Vitals;
    prediction: string;
    severity: string;
    remedies: string;
    case_name?: string;
    actual_label?: string;
};

// --- UI HELPER COMPONENTS ---
const StatCard = ({ icon, label, value, unit, color }: StatCardProps) => ( 
  <View style={styles.statCard}>
    <View style={{flexDirection: 'row', alignItems: 'center', marginBottom: 8}}>
      <Ionicons name={icon} size={20} color={color} />
      <Text style={[styles.statLabel, {color}]}>{label}</Text>
    </View>
    <Text style={[styles.statValue, {color}]}>{value}</Text>
    <Text style={[styles.statUnit, {color: Colors.textSecondary}]}>{unit}</Text>
  </View> 
);

const DiseaseCard = ({ disease, remedies }: { disease: string; remedies: string[] }) => (
  <View style={styles.diseaseCard}>
    <View style={styles.diseaseHeader}>
      <Ionicons name="medical" size={20} color={Colors.danger} />
      <Text style={styles.diseaseTitle}>Disease: {disease}</Text>
    </View>
    
    <View style={styles.remediesContainer}>
      <Text style={styles.remedyTitle}>Recommended Actions:</Text>
      {remedies.map((remedy, index) => (
        <View key={index} style={styles.remedyItem}>
          <Ionicons name="caret-forward" size={16} color={Colors.primary} style={styles.remedyIcon} />
          <Text style={styles.remedyText}>{remedy}</Text>
        </View>
      ))}
    </View>
  </View>
);

export default function LiveMonitorScreen() {
    const [isConnected, setIsConnected] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [currentCaseName, setCurrentCaseName] = useState('N/A');
    const [vitals, setVitals] = useState<Vitals | null>(null);
    const [analysis, setAnalysis] = useState<Analysis | null>(null);
    const [lastUpdated, setLastUpdated] = useState<string>('');

const fetchLiveUpdate = useCallback(async () => {
    if (!isConnected) return;
    
    setIsLoading(true);
    try {
        const response = await ApiService.wearableConnect();
        
        console.log('Wearable API Response:', response);
        
        if (response.ok && response.data) {
            const data: ApiResponse = response.data;
            
            // Set the vitals from the API response
            setVitals(data.input_data);
            
            // Parse remedies (they come as a string from the API, pipe-separated)
            const remediesArray = data.remedies.split('|').map(r => r.trim()).filter(r => r.length > 0);
            
            // Set the analysis from the API response
            setAnalysis({
                Disease: data.prediction,
                Remedies: remediesArray,
                Severity: data.severity
            });
            
            // Use the case_name from the API response
            setCurrentCaseName(data.case_name || `Case: ${data.prediction} (${data.severity} severity)`);
            
            // Update timestamp
            setLastUpdated(new Date().toLocaleTimeString());
        } else {
            Alert.alert('Error', response.data?.message || 'Failed to fetch data from wearable');
            setIsConnected(false);
        }
    } catch (error) {
        console.error('Error fetching wearable data:', error);
        Alert.alert('Error', 'Failed to connect to wearable device');
        setIsConnected(false);
    } finally {
        setIsLoading(false);
    }
}, [isConnected]);

    // Fetch data when connected state changes
    useEffect(() => {
        if (isConnected) {
            fetchLiveUpdate();
            
            // Set up interval to fetch data every 10 seconds while connected
            const interval = setInterval(fetchLiveUpdate, 10000);
            return () => clearInterval(interval);
        }
    }, [isConnected, fetchLiveUpdate]);

    const handleConnectToggle = () => {
        const newConnectionState = !isConnected;
        setIsConnected(newConnectionState);
        
        if (!newConnectionState) {
            // Reset everything when disconnecting
            setVitals(null);
            setAnalysis(null);
            setCurrentCaseName('N/A');
            setLastUpdated('');
        }
    };

    // Helper function to determine color based on value ranges
    const getValueColor = (type: string, value: number): string => {
        switch (type) {
            case 'glucose':
                if (value > 250) return Colors.danger;
                if (value > 180 || value < 70) return '#F39C12';
                return Colors.accent;
            case 'bp_systolic':
                if (value > 160) return Colors.danger;
                if (value > 130) return '#F39C12';
                return Colors.primary;
            case 'bp_diastolic':
                if (value > 100) return Colors.danger;
                if (value > 90) return '#F39C12';
                return Colors.primary;
            case 'heart_rate':
                if (value > 120 || value < 50) return Colors.danger;
                if (value > 100 || value < 60) return '#F39C12';
                return Colors.primary;
            case 'sleep':
                return value < 6 ? '#F39C12' : Colors.primary;
            case 'stress':
                return value > 7 ? Colors.danger : Colors.accent;
            default:
                return Colors.primary;
        }
    };

    const getSeverityColor = (severity: string): string => {
        switch (severity?.toLowerCase()) {
            case 'high': return Colors.danger;
            case 'medium': return '#F39C12';
            case 'low': return Colors.accent;
            default: return Colors.textSecondary;
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
                    {lastUpdated && (
                        <Text style={styles.lastUpdated}>Last updated: {lastUpdated}</Text>
                    )}
                </View>
                <TouchableOpacity 
                    onPress={handleConnectToggle} 
                    style={[styles.connectButton, isConnected ? styles.connected : styles.disconnected]}
                    disabled={isLoading}
                >
                    <Ionicons 
                        name={isConnected ? "watch" : "watch-outline"} 
                        size={20} 
                        color={isConnected ? Colors.accent : Colors.textSecondary} 
                    />
                    <Text style={[styles.connectButtonText, isConnected ? {color: Colors.accent} : {color: Colors.textSecondary}]}>
                        {isConnected ? 'Connected' : 'Connect'}
                    </Text>
                    {isLoading && <ActivityIndicator size="small" color={Colors.primary} style={{marginLeft: 8}} />}
                </TouchableOpacity>
            </View>
            
            <View style={styles.card}>
                <View style={styles.cardHeader}>
                    <Text style={styles.cardTitle}>Live Vitals</Text>
                    {isConnected && !isLoading && (
                        <TouchableOpacity onPress={fetchLiveUpdate}>
                            <Ionicons name="refresh" size={20} color={Colors.primary} />
                        </TouchableOpacity>
                    )}
                </View>
                
                {vitals ? (
                    <>
                        <Text style={styles.caseName}>{currentCaseName}</Text>
                        <View style={styles.statsGrid}>
                            <StatCard 
                                icon="water-outline" 
                                label="Glucose" 
                                value={vitals.glucose_level} 
                                unit="mg/dL" 
                                color={getValueColor('glucose', vitals.glucose_level)} 
                            />
                            <StatCard 
                                icon="heart-outline" 
                                label="Heart Rate" 
                                value={vitals.heart_rate} 
                                unit="bpm" 
                                color={getValueColor('heart_rate', vitals.heart_rate)} 
                            />
                            <StatCard 
                                icon="pulse-outline" 
                                label="Systolic BP" 
                                value={vitals.systolic_bp} 
                                unit="mmHg" 
                                color={getValueColor('bp_systolic', vitals.systolic_bp)} 
                            />
                            <StatCard 
                                icon="pulse-outline" 
                                label="Diastolic BP" 
                                value={vitals.diastolic_bp} 
                                unit="mmHg" 
                                color={getValueColor('bp_diastolic', vitals.diastolic_bp)} 
                            />
                            <StatCard 
                                icon="walk-outline" 
                                label="Activity" 
                                value={vitals.steps_per_min} 
                                unit="steps/min" 
                                color={Colors.accent} 
                            />
                            <StatCard 
                                icon="moon-outline" 
                                label="Sleep" 
                                value={vitals.sleep_hours} 
                                unit="hours" 
                                color={getValueColor('sleep', vitals.sleep_hours)} 
                            />
                            <StatCard 
                                icon="leaf-outline" 
                                label="Stress Level" 
                                value={vitals.stress_level} 
                                unit="/ 10" 
                                color={getValueColor('stress', vitals.stress_level)} 
                            />
                        </View>
                    </>
                ) : (
                    <Text style={styles.centeredText}>
                        {isConnected ? "Loading vitals..." : "Connect to view real-time vitals."}
                    </Text>
                )}
            </View>
            
            <View style={styles.card}>
                <View style={styles.cardHeader}>
                    <Text style={styles.cardTitle}>AI Model Analysis</Text>
                </View>
                
                {analysis ? (
                    <DiseaseCard 
                        disease={analysis.Disease} 
                        remedies={analysis.Remedies} 
                    />
                ) : (
                    <View style={styles.allGoodContainer}>
                       <Ionicons name={"information-circle-outline"} size={24} color={Colors.textSecondary} />
                       <Text style={styles.allGoodText}>
                           {isConnected ? "Awaiting analysis..." : "Connect to a device to begin."}
                       </Text>
                    </View>
                )}
            </View>
        </ScrollView>
    );
}

// --- STYLES ---
const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: Colors.background, },
    header: { padding: 20, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
    headerTitle: { fontSize: 24, fontWeight: 'bold', color: Colors.text, marginBottom: 4 },
    headerSubtitle: { fontSize: 14, color: Colors.textSecondary, marginBottom: 4 },
    lastUpdated: { fontSize: 12, color: Colors.textSecondary, fontStyle: 'italic' },
    connectButton: { 
        flexDirection: 'row', 
        alignItems: 'center', 
        paddingVertical: 8, 
        paddingHorizontal: 12, 
        borderRadius: 10,
        marginTop: 5
    },
    connected: { backgroundColor: '#E8F5E9' },
    disconnected: { backgroundColor: '#F5F5F5' },
    connectButtonText: { fontWeight: '600', marginLeft: 8 },
    card: { 
        backgroundColor: Colors.surface, 
        marginHorizontal: 15, 
        marginBottom: 15, 
        borderRadius: 15, 
        padding: 15, 
        shadowColor: "#000", 
        shadowOffset: { width: 0, height: 2, }, 
        shadowOpacity: 0.1, 
        shadowRadius: 8, 
        elevation: 5, 
    },
    cardHeader: { 
        flexDirection: 'row', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        marginBottom: 15, 
    },
    cardTitle: { fontSize: 18, fontWeight: 'bold', color: Colors.text },
    caseName: { fontSize: 16, color: Colors.textSecondary, marginBottom: 15, fontStyle: 'italic' },
    statsGrid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between' },
    statCard: { width: '48%', backgroundColor: '#F8F9FA', padding: 15, borderRadius: 10, marginBottom: 10, },
    statLabel: { fontSize: 14, fontWeight: '500', marginLeft: 5, },
    statValue: { fontSize: 24, fontWeight: 'bold', marginTop: 5 },
    statUnit: { fontSize: 12, },
    
    // New styles for disease card (matching WhatsApp image format)
    diseaseCard: {
        backgroundColor: '#F8F9FA',
        borderRadius: 10,
        padding: 15,
        borderLeftWidth: 4,
        borderLeftColor: Colors.danger
    },
    diseaseHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 15,
        paddingBottom: 10,
        borderBottomWidth: 1,
        borderBottomColor: '#E0E0E0'
    },
    diseaseTitle: {
        fontSize: 18,
        fontWeight: 'bold',
        color: Colors.text,
        marginLeft: 10
    },
    remediesContainer: {
        marginTop: 5
    },
    remedyTitle: {
        fontSize: 16,
        fontWeight: '600',
        color: Colors.text,
        marginBottom: 12,
        marginLeft: 5
    },
    remedyItem: {
        flexDirection: 'row',
        alignItems: 'flex-start',
        marginBottom: 10,
        paddingLeft: 5
    },
    remedyIcon: {
        marginTop: 2,
        marginRight: 10
    },
    remedyText: {
        fontSize: 14,
        color: Colors.textSecondary,
        flex: 1,
        lineHeight: 20
    },
    
    allGoodContainer: { flexDirection: 'row', alignItems: 'center', padding: 10, },
    allGoodText: { fontSize: 16, fontWeight: '600', color: Colors.textSecondary, marginLeft: 10, },
    centeredText: { textAlign: 'center', color: Colors.textSecondary, paddingVertical: 20 },
});