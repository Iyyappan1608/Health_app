import React, { useEffect } from 'react';
import Animated, { useAnimatedStyle, useSharedValue, withSpring } from 'react-native-reanimated';

type AnimatedCardProps = {
  children: React.ReactNode;
  index: number; // We'll use an index to stagger the animation
};

const AnimatedCard = ({ children, index }: AnimatedCardProps) => {
  // Shared values for opacity and vertical position
  const opacity = useSharedValue(0);
  const translateY = useSharedValue(50);

  const animatedStyle = useAnimatedStyle(() => {
    return {
      opacity: opacity.value,
      transform: [{ translateY: translateY.value }],
    };
  });

  useEffect(() => {
    // A delay based on the card's index creates a staggered effect
    const delay = index * 100; 

    const timeoutId = setTimeout(() => {
      // Use withSpring for a bouncy, natural animation
      opacity.value = withSpring(1);
      translateY.value = withSpring(0);
    }, delay);

    return () => clearTimeout(timeoutId);
  }, [index]);

  return <Animated.View style={animatedStyle}>{children}</Animated.View>;
};

export default AnimatedCard;