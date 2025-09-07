import React from 'react';
import { View, Dimensions, Text, StyleSheet } from 'react-native';
import { LineChart } from 'react-native-chart-kit';

// Get the width of the screen for a responsive chart
const screenWidth = Dimensions.get('window').width;

// Define the props our new chart component will accept
type VitalsChartProps = {
  labels: string[];
  data: number[];
};

const VitalsChart = ({ labels, data }: VitalsChartProps) => {
  // react-native-chart-kit requires at least one data point to render
  if (data.length === 0) {
    return (
      <View style={styles.container}>
        <Text style={styles.noDataText}>Not enough data to display chart.</Text>
      </View>
    );
  }

  const chartData = {
    labels: labels,
    datasets: [
      {
        data: data,
        color: (opacity = 1) => `rgba(74, 144, 226, ${opacity})`, // Blue line
        strokeWidth: 2,
      },
    ],
    legend: ['Glucose (mg/dL)'], // optional
  };

  const chartConfig = {
    backgroundColor: '#FFFFFF',
    backgroundGradientFrom: '#FFFFFF',
    backgroundGradientTo: '#FFFFFF',
    decimalPlaces: 0,
    color: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
    labelColor: (opacity = 1) => `rgba(100, 100, 100, ${opacity})`,
    style: {
      borderRadius: 16,
    },
    propsForDots: {
      r: '6',
      strokeWidth: '2',
      stroke: '#4A90E2',
    },
  };

  return (
    <View style={styles.container}>
      <LineChart
        data={chartData}
        width={screenWidth - 60} // Adjust width to fit inside the card
        height={220}
        chartConfig={chartConfig}
        bezier // Makes the line curvy
        style={{
          marginVertical: 8,
          borderRadius: 16,
        }}
      />
    </View>
  );
};

const styles = StyleSheet.create({
    container: {
        alignItems: 'center',
        justifyContent: 'center',
    },
    noDataText: {
        fontSize: 14,
        color: '#666',
        padding: 20,
    }
});

export default VitalsChart;