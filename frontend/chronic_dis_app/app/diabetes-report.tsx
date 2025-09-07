import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, SafeAreaView, StatusBar } from 'react-native';
import { Stack, useLocalSearchParams, useRouter } from 'expo-router';
import { Colors } from '../constants/Colors';
import DashboardCard from '../components/DashboardCard';

type ClusterStats = {
    mean_HbA1c: number;
    mean_BMI_at_Diagnosis: number;
    mean_Age_at_Diagnosis: number;
};

type DiabetesReport = {
    predicted_type: string;
    confidence_score: number;
    risk_level: string;
    prediction_explanation: string;
    risk_explanation: string;
    cluster_stats: ClusterStats | null;
};

// Define valid diabetes types
type DiabetesType = 'MODY' | 'Type 1' | 'Type 2' | 'LADA';

export default function DiabetesReportScreen() {
    const router = useRouter();
    const params = useLocalSearchParams();
    const { reportData } = params;

    const report: DiabetesReport | null = typeof reportData === 'string' ? JSON.parse(reportData) as DiabetesReport : null;

    if (!report) {
        return (
            <SafeAreaView style={[styles.container, styles.centerContent]}>
                <Text style={styles.errorText}>Error: Could not display report data.</Text>
            </SafeAreaView>
        );
    }

    const getRiskColor = (level: string) => {
        if (level === 'High') return Colors.danger;
        if (level === 'Medium') return '#F39C12';
        return Colors.accent;
    };

    // Fallback explanation if not provided by model
    const getFallbackExplanation = (type: string) => {
        const explanations: Record<string, string> = {
            "MODY": "Based on your clinical features, you most closely match the MODY subtype.",
            "Type 1": "Based on your clinical features, you most closely match Type 1 diabetes.",
            "Type 2": "Based on your clinical features, you most closely match Type 2 diabetes.",
            "LADA": "Based on your clinical features, you most closely match LADA (Latent Autoimmune Diabetes in Adults)."
        };
        
        return explanations[type] || "Based on your clinical features, a diabetes subtype prediction has been made.";
    };

    const predictionExplanation = report.prediction_explanation || getFallbackExplanation(report.predicted_type);

    return (
        <SafeAreaView style={styles.container}>
            <StatusBar barStyle="dark-content" />
            <Stack.Screen options={{ title: 'Diabetes Subtype Report' }} />
            <ScrollView contentContainerStyle={styles.contentContainer}>
                <Text style={styles.headerTitle}>Diabetes Subtype Report</Text>

                <DashboardCard icon="analytics-outline" title="Diagnosis Details">
                    <View style={styles.detailBlock}>
                        <Text style={styles.detailLabel}>Predicted Diabetes Type:</Text>
                        <Text style={styles.detailValue}>{report.predicted_type}</Text>
                    </View>
                    <View style={styles.separator} />
                    <View style={styles.detailBlock}>
                        <Text style={styles.detailLabel}>Confidence Score:</Text>
                        <Text style={styles.detailValue}>{report.confidence_score.toFixed(2)}%</Text>
                    </View>
                </DashboardCard>

                <DashboardCard icon="document-text-outline" title="Prediction Explanation">
                    <Text style={styles.explanationText}>
                        {predictionExplanation}
                    </Text>
                </DashboardCard>
                
                <DashboardCard icon="shield-checkmark-outline" title="Risk Level Assessment">
                     <View style={styles.detailBlock}>
                        <Text style={styles.detailLabel}>Predicted Risk Level:</Text>
                        <Text style={[styles.detailValue, { color: getRiskColor(report.risk_level) }]}>{report.risk_level}</Text>
                    </View>
                    <View style={styles.separator} />
                    {report.cluster_stats && (
                        <View style={styles.detailBlock}>
                           <Text style={styles.detailLabel}>Comparison Cluster Stats:</Text>
                           <Text style={styles.statsText}>- Mean HbA1c: {report.cluster_stats.mean_HbA1c.toFixed(2)}%</Text>
                           <Text style={styles.statsText}>- Mean BMI: {report.cluster_stats.mean_BMI_at_Diagnosis.toFixed(1)}</Text>
                           <Text style={styles.statsText}>- Mean Age: {report.cluster_stats.mean_Age_at_Diagnosis.toFixed(1)}</Text>
                        </View>
                    )}
                    <Text style={styles.explanationText}>{report.risk_explanation}</Text>
                </DashboardCard>

                <TouchableOpacity style={styles.doneButton} onPress={() => router.back()}>
                    <Text style={styles.doneButtonText}>Done</Text>
                </TouchableOpacity>
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: Colors.background },
    contentContainer: { padding: 20, paddingBottom: 40 },
    headerTitle: { fontSize: 28, fontWeight: 'bold', color: Colors.text, marginBottom: 20 },
    detailBlock: { marginBottom: 10 },
    detailLabel: { fontSize: 16, fontWeight: '600', color: Colors.textSecondary, marginBottom: 4, },
    detailValue: { fontSize: 22, fontWeight: 'bold', color: Colors.primary },
    explanationText: { fontSize: 16, color: Colors.text, lineHeight: 24, fontStyle: 'italic' },
    statsText: { fontSize: 14, color: Colors.textSecondary, lineHeight: 22 },
    doneButton: { backgroundColor: Colors.primary, padding: 18, borderRadius: 15, alignItems: 'center', marginTop: 20 },
    doneButtonText: { color: '#FFFFFF', fontSize: 18, fontWeight: 'bold' },
    separator: { height: 1, backgroundColor: '#EFEFEF', marginVertical: 15, },
    
    centerContent: {
        justifyContent: 'center',
        alignItems: 'center',
    },
    errorText: {
        fontSize: 18,
        color: Colors.danger,
        textAlign: 'center',
    },
});