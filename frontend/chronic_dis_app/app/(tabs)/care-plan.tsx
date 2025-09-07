import React, { useState, useMemo, useCallback } from 'react';
import {
  SafeAreaView,
  FlatList,
  StyleSheet,
  Text,
  View,
  StatusBar,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import ApiService from '../../services/ApiService';

// --- 1. DATA TYPES (for TypeScript) ---
interface PlanItemData {
  id: string;
  icon: string;
  type: string;
  title: string;
  description: string;
}

interface DayPlanData {
  id: string;
  title: string;
  items: PlanItemData[];
}

// --- 2. PARSING LOGIC ---
const EMOJI_MAP: Record<string, { type: string; title: string }> = {
  '🏃': { type: 'activity', title: 'Physical Activity' },
  '🧘': { type: 'wellness', title: 'Mental Wellness' },
  '🥗': { type: 'meals', title: 'Meals' },
  '💧': { type: 'hydration', title: 'Hydration' },
  '❌': { type: 'avoid', title: 'Avoid' },
  '✅': { type: 'risk_reduction', title: "Today's Risk Reduction" },
  '⚠': { type: 'consequences', title: 'Consequences if Skipped' },
};

const parseCarePlanText = (text: string): DayPlanData[] => {
  if (!text || typeof text !== 'string') {
    Alert.alert('Error', 'Invalid care plan data received from the server.');
    return [];
  }
  const dayBlocks = text.trim().split(/\n?(?=Day \d+)/).filter(Boolean);

  return dayBlocks.map((block, index) => {
    const lines = block.trim().split('\n');
    const dayTitle = lines.shift() || `Day ${index + 1}`;
    
    const items = lines
      .map((line) => {
        const icon = line.trim().charAt(0);
        const mappedInfo = EMOJI_MAP[icon];
        if (!mappedInfo) return null;
        
        const content = line.substring(1).trim();
        const [action, explanation] = content.split('→').map((s) => s ? s.trim() : '');
        
        return {
          id: `${dayTitle}-${mappedInfo.type}`,
          icon,
          type: mappedInfo.type,
          title: action || mappedInfo.title,
          description: explanation || '',
        };
      })
      .filter((item): item is PlanItemData => item !== null);
      
    return { id: dayTitle, title: dayTitle, items };
  });
};

// --- 3. REUSABLE UI COMPONENTS ---
const PlanItem: React.FC<{ item: PlanItemData }> = React.memo(({ item }) => (
  <View style={styles.itemContainer}>
    <Text style={styles.itemIcon}>{item.icon}</Text>
    <View style={styles.itemTextContainer}>
      <Text style={styles.itemTitle}>{item.title}</Text>
      {item.description ? <Text style={styles.itemDescription}>{item.description}</Text> : null}
    </View>
  </View>
));

const DayPlanCard: React.FC<{ day: DayPlanData }> = ({ day }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // --- THIS IS THE FIX ---
  // We explicitly tell TypeScript the types that useMemo will return.
  // This one change fixes all four errors.
  const { standardItems, extraItems } = useMemo<{
    standardItems: PlanItemData[];
    extraItems: PlanItemData[];
  }>(() => {
    const standard = day.items.filter(
      (item) => !['risk_reduction', 'consequences'].includes(item.type)
    );
    const extra = day.items.filter(
      (item) => ['risk_reduction', 'consequences'].includes(item.type)
    );
    return { standardItems: standard, extraItems: extra };
  }, [day.items]);

  return (
    <View style={styles.card}>
      <Text style={styles.dayTitle}>{day.title}</Text>
      {standardItems.map((item) => (
        <PlanItem key={item.id} item={item} />
      ))}
      {extraItems.length > 0 && (
        <TouchableOpacity style={styles.button} onPress={() => setIsExpanded(!isExpanded)}>
          <Text style={styles.buttonText}>
            {isExpanded ? 'Show Less' : 'Show Risk & Consequences'}
          </Text>
        </TouchableOpacity>
      )}
      {isExpanded && (
        <View style={styles.expandedSection}>
          {extraItems.map((item) => <PlanItem key={item.id} item={item} />)}
        </View>
      )}
    </View>
  );
};

// --- 4. MAIN SCREEN COMPONENT ---
export default function CarePlanScreen() {
  const [planData, setPlanData] = useState<DayPlanData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGeneratePlan = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setPlanData([]);
    try {
      const response = await ApiService.generateCarePlan();
      if (response.ok && response.data.care_plan) {
        const structuredData = parseCarePlanText(response.data.care_plan);
        if (structuredData.length === 0) {
          setError("The generated care plan was empty or could not be read. Please ensure you have recent health reports.");
        } else {
          setPlanData(structuredData);
        }
      } else {
        throw new Error(response.data.error || 'Failed to generate care plan.');
      }
    } catch (err: any) {
      setError(err.message);
      Alert.alert('Generation Failed', err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const renderEmptyComponent = () => (
    <View style={styles.centeredView}>
        {isLoading ? (
            <>
                <ActivityIndicator size="large" color="#1D3557" />
                <Text style={styles.statusText}>Generating your personalized plan...</Text>
            </>
        ) : error ? (
            <>
                <Text style={styles.errorText}>{error}</Text>
                <TouchableOpacity style={styles.primaryButton} onPress={handleGeneratePlan}>
                    <Text style={styles.primaryButtonText}>Retry Generation</Text>
                </TouchableOpacity>
            </>
        ) : (
            <>
                <Text style={styles.emptyTitle}>No Care Plan Available</Text>
                <Text style={styles.emptySubtitle}>
                    Generate a new plan based on your latest health reports.
                </Text>
                <TouchableOpacity style={styles.primaryButton} onPress={handleGeneratePlan}>
                    <Text style={styles.primaryButtonText}>Generate My Care Plan</Text>
                </TouchableOpacity>
            </>
        )}
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <FlatList
        data={planData}
        renderItem={({ item }) => <DayPlanCard day={item} />}
        keyExtractor={(item) => item.id}
        contentContainerStyle={planData.length === 0 ? styles.flexGrow : styles.list}
        ListEmptyComponent={renderEmptyComponent}
        ListHeaderComponent={planData.length > 0 ? (
          <View style={styles.header}>
            <Text style={styles.title}>Your Personalized Care Plan</Text>
            <Text style={styles.subtitle}>Follow these recommendations to improve your health.</Text>
          </View>
        ) : null}
        refreshControl={<RefreshControl refreshing={isLoading} onRefresh={handleGeneratePlan} tintColor="#1D3557" />}
      />
    </SafeAreaView>
  );
}

// --- 5. STYLES ---
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F8F9FA' },
  flexGrow: { flexGrow: 1 },
  header: { paddingHorizontal: 20, paddingTop: 20, paddingBottom: 10 },
  title: { fontSize: 28, fontWeight: 'bold', color: '#1D3557' },
  subtitle: { fontSize: 16, color: '#495057', marginTop: 8 },
  list: { paddingBottom: 32, paddingTop: 10 },
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 20,
    marginHorizontal: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
    elevation: 5,
  },
  dayTitle: { fontSize: 22, fontWeight: 'bold', marginBottom: 20, color: '#1A237E' },
  button: {
    backgroundColor: '#E8EAF6',
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: 'center',
    marginTop: 10,
    marginBottom: 10,
  },
  buttonText: { color: '#3F51B5', fontWeight: 'bold', fontSize: 14 },
  itemContainer: { flexDirection: 'row', alignItems: 'flex-start', marginBottom: 16 },
  itemIcon: { fontSize: 24, marginRight: 12, marginTop: 2, color: '#34495E' },
  itemTextContainer: { flex: 1 },
  itemTitle: { fontSize: 16, fontWeight: 'bold', color: '#2C3E50', marginBottom: 4 },
  itemDescription: { fontSize: 14, color: '#566573', lineHeight: 21 },
  centeredView: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  statusText: {
    marginTop: 12,
    fontSize: 16,
    color: '#495057',
  },
  emptyTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#1D3557',
    textAlign: 'center',
    marginBottom: 12,
  },
  emptySubtitle: {
    fontSize: 16,
    color: '#6c757d',
    textAlign: 'center',
    marginBottom: 24,
  },
  primaryButton: {
    backgroundColor: '#1D3557',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 10,
    elevation: 2,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  errorText: {
    fontSize: 16,
    color: '#D9534F',
    textAlign: 'center',
    marginBottom: 20,
  },
  expandedSection: {
    borderTopWidth: 1,
    borderTopColor: '#EEE',
    marginTop: 10,
    paddingTop: 16,
  }
});