import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React from 'react';
import { ActivityIndicator, Alert, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import AnimatedCard from '../../components/AnimatedCard';
import DashboardCard from '../../components/DashboardCard';
import { Colors } from '../../constants/Colors';
import { useData } from '../../src/context/DataContext';
import AuthService from '../../services/AuthService';

// The ProfileMenuItem component is no longer needed
// const ProfileMenuItem = ...

export default function ProfileScreen() {
    const router = useRouter();
    const { userData, isLoading } = useData();

    const handleLogout = async () => {
        await AuthService.clearAuthData();
        router.replace('/login');
    };

    const confirmLogout = () => {
        Alert.alert(
            "Log Out",
            "Are you sure you want to log out?",
            [
                { text: "Cancel", style: "cancel" },
                { text: "Log Out", onPress: handleLogout, style: "destructive" }
            ]
        );
    };

    if (isLoading || !userData) {
        return (
            <View style={[styles.container, { justifyContent: 'center', alignItems: 'center' }]}>
                <ActivityIndicator size="large" color={Colors.primary} />
            </View>
        );
    }

    return (
        <ScrollView style={styles.container}>
            <View style={styles.header}>
                <Ionicons name="person-circle-outline" size={100} color={Colors.primary} />
                <Text style={styles.nameText}>{userData.name}</Text>
                <Text style={styles.emailText}>{userData.email || 'user@example.com'}</Text>
            </View>

            <AnimatedCard index={0}>
                <DashboardCard icon="medkit-outline" title="Health Profile">
                    <Text style={styles.infoText}>
                        {(userData.predictedConditions || []).join(', ') || 'No conditions analyzed yet.'}
                    </Text>
                </DashboardCard>
            </AnimatedCard>

            {/* --- THE MENU CARD SECTION HAS BEEN REMOVED --- */}

            <TouchableOpacity style={styles.logoutButton} onPress={confirmLogout}>
                <Ionicons name="log-out-outline" size={24} color={Colors.danger} />
                <Text style={styles.logoutButtonText}>Log Out</Text>
            </TouchableOpacity>
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: Colors.background,
    },
    header: {
        alignItems: 'center',
        paddingVertical: 30,
        paddingHorizontal: 20,
        backgroundColor: Colors.surface,
        borderBottomWidth: 1,
        borderBottomColor: '#E0E0E0',
    },
    nameText: {
        fontSize: 24,
        fontWeight: 'bold',
        color: Colors.text,
        marginTop: 10,
    },
    emailText: {
        fontSize: 16,
        color: Colors.textSecondary,
    },
    infoText: {
        fontSize: 16,
        color: Colors.text,
        textAlign: 'center'
    },
    // menuContainer, menuRow, and menuText styles are no longer needed
    logoutButton: {
        flexDirection: 'row',
        backgroundColor: '#FFEBEE',
        margin: 20,
        paddingVertical: 15,
        paddingHorizontal: 40,
        borderRadius: 15,
        alignItems: 'center',
        justifyContent: 'center',
        marginTop: 30, // Added margin top for spacing
    },
    logoutButtonText: {
        color: Colors.danger,
        fontSize: 16,
        fontWeight: 'bold',
        marginLeft: 10,
    },
});