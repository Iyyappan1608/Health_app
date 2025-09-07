import { LinearGradient } from 'expo-linear-gradient';
import { useRouter } from 'expo-router';
import React, { useEffect, useRef } from 'react';
import { Animated, Image, SafeAreaView, StatusBar, StyleSheet, View } from 'react-native';
import { Colors } from '../constants/Colors';

export default function SplashScreen() {
  const router = useRouter();
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 800,
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 5,
        useNativeDriver: true,
      })
    ]).start();

    const timer = setTimeout(() => {
      // UPDATED: Navigate to the new welcome screen
      router.replace('/welcome');
    }, 1500); 

    return () => clearTimeout(timer);
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient
        colors={['#E0F2F7', Colors.background]}
        style={StyleSheet.absoluteFill}
      />
      <StatusBar barStyle="dark-content" />
      <View style={styles.content}>
        <Animated.View style={{ opacity: fadeAnim, transform: [{ scale: scaleAnim }] }}>
          <Image
            source={require('../assets/app-icon.png')}
            style={styles.logoImage}
          />
        </Animated.View>
        <Animated.Text style={[styles.tagline, { opacity: fadeAnim }]}>
          Your Health, Predicted.
        </Animated.Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'transparent',
  },
  logoImage: {
    width: 120,
    height: 120,
    resizeMode: 'contain',
  },
  tagline: {
    fontSize: 18,
    color: Colors.textSecondary,
    marginTop: 20,
    fontWeight: '600',
  },
});
