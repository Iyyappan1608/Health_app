// services/NotificationService.ts
import { Platform } from 'react-native';
import PushNotification from 'react-native-push-notification';

// Flag to track if notification service is configured
let isConfigured = false;

// Configure push notifications (call this once when app starts)
export const configureNotifications = () => {
  if (isConfigured) return;
  
  PushNotification.configure({
    onRegister: function (token) {
      console.log('TOKEN:', token);
    },
    onNotification: function (notification) {
      console.log('NOTIFICATION:', notification);
    },
    permissions: {
      alert: true,
      badge: true,
      sound: true,
    },
    popInitialNotification: true,
    requestPermissions: true,
  });

  // Create notification channel for Android
  if (Platform.OS === 'android') {
    PushNotification.createChannel(
      {
        channelId: 'care-plan-channel',
        channelName: 'Care Plan Notifications',
        channelDescription: 'Notifications for daily care plans',
        soundName: 'default',
        importance: 4,
        vibrate: true,
      },
      (created) => console.log(`Channel created: ${created}`)
    );
  }
  
  isConfigured = true;
};

export const schedulePushNotification = async (dayNumber: number, message: string) => {
  try {
    const title = `Day ${dayNumber} Care Plan`;
    
    if (Platform.OS === 'android') {
      PushNotification.localNotification({
        channelId: 'care-plan-channel',
        title,
        message,
        playSound: true,
        soundName: 'default',
      });
    } else if (Platform.OS === 'ios') {
      PushNotification.localNotification({
        title,
        message,
        playSound: true,
        soundName: 'default',
      });
    }
  } catch (error) {
    console.error('Failed to schedule notification:', error);
  }
};

export const cancelAllScheduledNotifications = () => {
  PushNotification.cancelAllLocalNotifications();
};