import React from 'react';
import { View, Text, StyleSheet, SafeAreaView, StatusBar, Image, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Colors } from '../constants/Colors';
import { Ionicons } from '@expo/vector-icons';

export default function WelcomeScreen() {
  const router = useRouter();

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient
        colors={['#E0F2F7', Colors.background]}
        style={StyleSheet.absoluteFill}
      />
      <StatusBar barStyle="dark-content" />
      <View style={styles.content}>
        <Image 
          source={require('../assets/app-icon.png')} 
          style={styles.logoImage}
        />
        <Text style={styles.title}>Welcome</Text>
        <Text style={styles.subtitle}>Your personal health companion.</Text>

        <TouchableOpacity style={styles.button} onPress={() => router.push('/login')}>
          <Ionicons name="person-outline" size={32} color={Colors.primary} />
          <Text style={styles.buttonText}>Patient Portal</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: 'transparent',
  },
  logoImage: { 
    width: 120,
    height: 120,
    resizeMode: 'contain',
    marginBottom: 20,
  },
  title: {
    fontSize: 36,
    fontWeight: 'bold',
    color: Colors.text,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 18,
    color: Colors.textSecondary,
    textAlign: 'center',
    marginBottom: 60,
  },
  button: {
    backgroundColor: Colors.surface,
    paddingVertical: 20,
    paddingHorizontal: 20,
    borderRadius: 15,
    width: '100%',
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  buttonText: {
    color: Colors.primary,
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 15,
  },
});