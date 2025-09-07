import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, SafeAreaView, StatusBar } from 'react-native';
import { Stack, useLocalSearchParams, useRouter } from 'expo-router';
import { Colors } from '../constants/Colors';
import DashboardCard from '../components/DashboardCard';

// Updated type to include the new risk data
type HypertensionReport = {
    hypertension_risk: boolean;
    probability: number;
    risk_level: string;
    stage: string;
    subtype: string;
    kidney_risk_1yr: number;
    stroke_risk_1yr: number;
    heart_risk_1yr: number;
    explanation: string;
};

export default function HypertensionReportScreen() {
    const router = useRouter();
    const params = useLocalSearchParams();
    const { reportData } = params;

    const report: HypertensionReport | null = typeof reportData === 'string' ? JSON.parse(reportData) as HypertensionReport : null;

    if (!report) {
        return (
            <SafeAreaView style={[styles.container, styles.centerContent]}>
                <Text style={styles.errorText}>Error: Could not display report data.</Text>
            </SafeAreaView>
        );
    }

    const getRiskColor = (level: string) => {
        if (level === 'High') return Colors.danger;
        if (level === 'Medium') return '#F39C12'; // Orange/Yellow
        return Colors.accent; // Use accent color for Low risk
    };

    const getRiskPercentageColor = (percentage: number) => {
        if (percentage >= 60) return Colors.danger;
        if (percentage >= 40) return '#F39C12'; // Orange/Yellow
        return '#22C55E'; // Green
    };

    return (
        <SafeAreaView style={styles.container}>
            <StatusBar barStyle="dark-content" />
            <Stack.Screen options={{ title: 'Hypertension Analysis Report' }} />
            <ScrollView contentContainerStyle={styles.contentContainer}>
                <Text style={styles.headerTitle}>Hypertension Analysis Report</Text>

                <DashboardCard icon="analytics-outline" title="Diagnosis Results">
                    <View style={styles.detailBlock}>
                        <Text style={styles.detailLabel}>Hypertension Risk:</Text>
                        <Text style={[styles.detailValue, { color: report.hypertension_risk ? Colors.danger : '#22C55E' }]}>
                            {report.hypertension_risk ? 'Detected' : 'Not Detected'}
                        </Text>
                    </View>
                    <View style={styles.separator} />
                    
                    <View style={styles.detailBlock}>
                        <Text style={styles.detailLabel}>Hypertension Stage:</Text>
                        <Text style={styles.detailValue}>{report.stage || 'Not Specified'}</Text>
                    </View>
                    <View style={styles.separator} />
                    
                    <View style={styles.detailBlock}>
                        <Text style={styles.detailLabel}>Subtype:</Text>
                        <Text style={styles.detailValue}>{report.subtype || 'Not Specified'}</Text>
                    </View>
                    <View style={styles.separator} />
                    
                    <View style={styles.detailBlock}>
                        <Text style={styles.detailLabel}>Probability:</Text>
                        <Text style={styles.detailValue}>{(report.probability * 100).toFixed(1)}%</Text>
                    </View>
                    <View style={styles.separator} />
                    
                    <View style={styles.detailBlock}>
                        <Text style={styles.detailLabel}>Risk Level:</Text>
                        <Text style={[styles.detailValue, { color: getRiskColor(report.risk_level) }]}>
                            {report.risk_level}
                        </Text>
                    </View>
                </DashboardCard>

                {/* New Risk Assessment Card */}
                <DashboardCard icon="warning-outline" title="⚠️ 1-Year Risk Assessment">
                    <View style={styles.riskRow}>
                        <Text style={styles.riskLabel}>Kidney Risk:</Text>
                        <Text style={[styles.riskValue, { color: getRiskPercentageColor(report.kidney_risk_1yr) }]}>
                            {report.kidney_risk_1yr.toFixed(2)}%
                        </Text>
                    </View>
                    
                    <View style={styles.riskRow}>
                        <Text style={styles.riskLabel}>Stroke Risk:</Text>
                        <Text style={[styles.riskValue, { color: getRiskPercentageColor(report.stroke_risk_1yr) }]}>
                            {report.stroke_risk_1yr.toFixed(2)}%
                        </Text>
                    </View>
                    
                    <View style={styles.riskRow}>
                        <Text style={styles.riskLabel}>Heart Attack Risk:</Text>
                        <Text style={[styles.riskValue, { color: getRiskPercentageColor(report.heart_risk_1yr) }]}>
                            {report.heart_risk_1yr.toFixed(2)}%
                        </Text>
                    </View>
                </DashboardCard>

                {/* Risk Statements Card */}
                <DashboardCard icon="chatbubble-ellipses-outline" title="💬 Risk Statements">
                    <View style={styles.riskStatement}>
                        <Text style={styles.riskStatementText}>
                            • {report.kidney_risk_1yr.toFixed(2)}% risk of kidney function decline in 1 year if BP not controlled
                        </Text>
                    </View>
                    
                    <View style={styles.riskStatement}>
                        <Text style={styles.riskStatementText}>
                            • {report.stroke_risk_1yr.toFixed(2)}% risk of stroke in 1 year if BP not controlled
                        </Text>
                    </View>
                    
                    <View style={styles.riskStatement}>
                        <Text style={styles.riskStatementText}>
                            • {report.heart_risk_1yr.toFixed(2)}% risk of heart attack in 1 year if BP not controlled
                        </Text>
                    </View>
                </DashboardCard>

                <DashboardCard icon="document-text-outline" title="Explanation">
                    <Text style={styles.explanationText}>{report.explanation}</Text>
                </DashboardCard>

                <TouchableOpacity style={styles.doneButton} onPress={() => router.back()}>
                    <Text style={styles.doneButtonText}>Done</Text>
                </TouchableOpacity>
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: { 
        flex: 1, 
        backgroundColor: Colors.background 
    },
    centerContent: {
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    contentContainer: {
        padding: 20,
        paddingBottom: 40,
    },
    headerTitle: { 
        fontSize: 28, 
        fontWeight: 'bold', 
        color: Colors.text, 
        marginBottom: 20,
        textAlign: 'center'
    },
    errorText: {
        fontSize: 16,
        color: Colors.danger,
        textAlign: 'center',
    },
    detailBlock: {
        marginBottom: 15,
    },
    detailLabel: { 
        fontSize: 16,
        fontWeight: '600',
        color: Colors.textSecondary,
        marginBottom: 4,
    },
    detailValue: { 
        fontSize: 22, 
        fontWeight: 'bold', 
        color: Colors.primary, 
    },
    riskRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingVertical: 10,
        borderBottomWidth: 1,
        borderBottomColor: '#F0F0F0',
    },
    riskLabel: {
        fontSize: 16,
        color: Colors.text,
        fontWeight: '500',
    },
    riskValue: {
        fontSize: 18,
        fontWeight: 'bold',
    },
    riskStatement: {
        paddingVertical: 8,
        borderLeftWidth: 3,
        borderLeftColor: Colors.danger,
        paddingLeft: 12,
        marginBottom: 10,
    },
    riskStatementText: {
        fontSize: 14,
        color: Colors.text,
        lineHeight: 20,
    },
    explanationText: {
        fontSize: 16,
        color: Colors.text,
        lineHeight: 24,
    },
    doneButton: { 
        backgroundColor: Colors.primary, 
        padding: 18, 
        borderRadius: 15, 
        alignItems: 'center',
        marginTop: 20,
    },
    doneButtonText: { 
        color: '#FFFFFF', 
        fontSize: 18, 
        fontWeight: 'bold' 
    },
    separator: {
        height: 1,
        backgroundColor: '#EFEFEF',
        marginVertical: 15,
    }
});