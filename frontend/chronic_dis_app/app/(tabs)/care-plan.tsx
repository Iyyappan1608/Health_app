import React, { useState, useMemo, useCallback, useEffect } from 'react';
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
  ScrollView,
} from 'react-native';
import ApiService from '../../services/ApiService';
import { schedulePushNotification} from "../../services/NotificationService";

interface CarePlanProgress {
  current_day: number;
  last_completed_day: number;
  is_completed: boolean;
  last_accessed?: string;
}

interface DayAccessCheck {
  can_access: boolean;
  current_day: number;
  last_completed_day: number;
}

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
  rawContent?: string; // Add raw content field
}

// --- 2. PARSING LOGIC ---
const EMOJI_MAP: Record<string, { type: string; title: string }> = {
  '🏃': { type: 'activity', title: 'Physical Activity' },
  '🧘': { type: 'wellness', title: 'Mental Wellness' },
  '🥗': { type: 'meals', title: 'Meals' },
  '💧': { type: 'hydration', title: 'Hydration' },
  '⌛': { type: 'avoid', title: 'Avoid' },
  '❌': { type: 'avoid', title: 'Avoid' },
  '✅': { type: 'risk_reduction', title: "Today's Risk Reduction" },
  '⚠️': { type: 'consequences', title: 'Consequences if Skipped' },
  '⚠': { type: 'consequences', title: 'Consequences if Skipped' },
};

// ----------------- Care Plan Parser -----------------
export function parseCarePlanResponse(planData: any): DayPlanData[] {
  if (!planData) {
    return [];
  }
  
  const days: DayPlanData[] = [];
  
  console.log('Raw plan data received:', planData);
  
  // Handle different response formats
  let structuredPlan: Record<string, string> = {};
  
  try {
    if (typeof planData === 'string') {
      const daySections = planData.split(/\n(?=Day \d+)/i);
      
      daySections.forEach(section => {
        const firstLine = section.split('\n')[0]?.trim();
        if (firstLine && firstLine.match(/Day \d+/i)) {
          structuredPlan[firstLine] = section;
        }
      });
    } else if (typeof planData === 'object' && planData !== null) {
      if (Array.isArray(planData)) {
        planData.forEach((day: any, index: number) => {
          if (day && typeof day === 'object') {
            const dayTitle = day.title || `Day ${index + 1}`;
            structuredPlan[dayTitle] = day.content || JSON.stringify(day);
          }
        });
      } else {
        structuredPlan = planData;
      }
    }
    
    // Parse each day
    Object.entries(structuredPlan).forEach(([dayTitle, dayContent]) => {
      const contentString = typeof dayContent === 'string' ? dayContent : JSON.stringify(dayContent);
      const lines = contentString.split('\n').filter(line => line && line.trim());
      
      const items = lines.map((line, index) => {
        const trimmedLine = line.trim();
        if (!trimmedLine) return null;
        
        const iconMatch = trimmedLine.match(/^([\u{1F300}-\u{1F9FF}]|[\u{2600}-\u{26FF}])/u);
        const icon = iconMatch ? iconMatch[0] : '📋';
        const content = iconMatch ? trimmedLine.substring(iconMatch[0].length).trim() : trimmedLine;
        
        const mappedInfo = EMOJI_MAP[icon] || { type: 'general', title: 'General' };
        
        const [action, explanation] = content.split('→').map(s => s ? s.trim() : '');
        
        return {
          id: `${dayTitle}-${mappedInfo.type}-${index}`,
          icon,
          type: mappedInfo.type,
          title: action || mappedInfo.title,
          description: explanation || '',
        };
      }).filter(Boolean) as PlanItemData[];
      
      days.push({
        id: dayTitle,
        title: dayTitle,
        items,
        rawContent: contentString // Store the raw content
      });
    });
    
    return days.sort((a, b) => {
      const dayA = parseInt(a.title.replace(/Day /i, '')) || 0;
      const dayB = parseInt(b.title.replace(/Day /i, '')) || 0;
      return dayA - dayB;
    });
    
  } catch (error) {
    console.error('Error parsing care plan:', error);
    return [];
  }
}
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
PlanItem.displayName = "PlanItem";

// --- 4. MAIN SCREEN COMPONENT ---
export default function CarePlanScreen() {
  const [planData, setPlanData] = useState<DayPlanData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentDay, setCurrentDay] = useState(1);
  const [lastCompletedDay, setLastCompletedDay] = useState(0);
  const [refreshing, setRefreshing] = useState(false);
  const [showRawContent, setShowRawContent] = useState(false); // Toggle for raw content

  useEffect(() => {
    loadProgress();
  }, []);

const loadProgress = async () => {
  try {
    const response = await ApiService.getCarePlanProgress();
    if (response && response.ok) {
      const progress: CarePlanProgress = response.data;
      setCurrentDay(progress.current_day);
      setLastCompletedDay(progress.last_completed_day);
    }
  } catch (error) {
    console.error('Error loading progress:', error);
  }
};

const checkDayAccess = async (day: number): Promise<boolean> => {
  try {
    const response = await ApiService.canAccessCarePlanDay({ day });
    return response && response.ok && response.data.can_access;
  } catch (error) {
    console.error('Error checking day access:', error);
    return false;
  }
};

const updateProgress = async (newCurrentDay: number, newLastCompletedDay: number) => {
  try {
    const response = await ApiService.updateCarePlanProgress({
      current_day: newCurrentDay,
      last_completed_day: newLastCompletedDay
    });
    if (response && response.ok) {
      setCurrentDay(newCurrentDay);
      setLastCompletedDay(newLastCompletedDay);
    }
  } catch (error) {
    console.error('Error updating progress:', error);
  }
};

const recordInteraction = async (dayNumber: number, question: string, response: string, carePlanText?: string) => {
  try {
    await ApiService.recordCarePlanInteraction({
      day_number: dayNumber,
      question,
      response,
      care_plan_text: carePlanText
    });
  } catch (error) {
    console.error('Error recording interaction:', error);
  }
};

  const handleDayCompletion = async (dayNumber: number, followedPlan: boolean) => {
    // Record the interaction
    const dayPlan = planData[dayNumber - 1]?.items.map(item => 
      `${item.icon} ${item.title}: ${item.description}`
    ).join('\n');
    
    await recordInteraction(
      dayNumber,
      'followed_plan',
      followedPlan ? 'yes' : 'no',
      dayPlan
    );

    if (followedPlan) {
      // Move to next day if plan was followed
      const newLastCompleted = Math.max(lastCompletedDay, dayNumber);
      const newCurrentDay = dayNumber < 7 ? dayNumber + 1 : dayNumber;
      
      await updateProgress(newCurrentDay, newLastCompleted);
      
      if (dayNumber < 7) {
        Alert.alert('Success', 'Great job! Your progress has been saved. You can continue with the next day tomorrow.');
      } else {
        Alert.alert('Congratulations!', 'You have completed your 7-day care plan!');
      }
    } else {
      Alert.alert('Noted', 'Your response has been recorded. Please try to follow the plan for better results.');
    }
  };

  const DayPlanCard: React.FC<{ day: DayPlanData }> = ({ day }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const dayNumber = parseInt(day.title.replace('Day ', '').replace('DAY ', '').replace('day ', ''));

    const { standardItems, extraItems } = useMemo(() => {
      const standard = day.items.filter(
        (item) => !['risk_reduction', 'consequences'].includes(item.type)
      );
      const extra = day.items.filter(
        (item) => ['risk_reduction', 'consequences'].includes(item.type)
      );
      return { standardItems: standard, extraItems: extra };
    }, [day.items]);

    // Only show the current day's card
    if (dayNumber !== currentDay) {
      return null;
    }

    return (
      <View style={styles.card}>
        <Text style={styles.dayTitle}>{day.title}</Text>
        
        {/* Toggle between structured and raw view */}
        <TouchableOpacity 
          style={styles.toggleButton}
          onPress={() => setShowRawContent(!showRawContent)}
        >
          <Text style={styles.toggleButtonText}>
            {showRawContent ? 'Show Structured View' : 'Show Raw LLM Output'}
          </Text>
        </TouchableOpacity>

        {showRawContent ? (
          // Show raw LLM content
          <ScrollView style={styles.rawContentContainer}>
            <Text style={styles.rawContentText}>{day.rawContent}</Text>
          </ScrollView>
        ) : (
          // Show structured content
          <>
            {standardItems.length > 0 ? (
              standardItems.map((item) => (
                <PlanItem key={item.id} item={item} />
              ))
            ) : (
              <Text style={styles.emptyDayText}>No activities planned for this day.</Text>
            )}
            
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
          </>
        )}
        
        {/* Completion section - only show for current day */}
        <View style={styles.completionSection}>
          <Text style={styles.completionQuestion}>Did you follow today&#39;s plan?</Text>
          <View style={styles.completionButtons}>
            <TouchableOpacity 
              style={[styles.completionButton, styles.yesButton]}
              onPress={() => handleDayCompletion(dayNumber, true)}
            >
              <Text style={styles.completionButtonText}>Yes</Text>
            </TouchableOpacity>
            <TouchableOpacity 
              style={[styles.completionButton, styles.noButton]}
              onPress={() => handleDayCompletion(dayNumber, false)}
            >
              <Text style={styles.completionButtonText}>No</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    );
  };

// Fixed handleGeneratePlan function in care-plan.tsx

const handleGeneratePlan = useCallback(async () => {
  if (!(await checkDayAccess(currentDay))) {
    Alert.alert('Access Denied', 'Please complete the previous day before accessing this day.');
    return;
  }

  setIsLoading(true);
  setError(null);
  
  try {
    const response = await ApiService.generateCarePlan();
    console.log('API Response:', response);

    // Check if response exists
    if (!response) {
      throw new Error("No response received from server");
    }

    // Handle the response format returned by the fixed server endpoint
    if (response.ok && response.data) {
      console.log('Response data structure:', Object.keys(response.data));
      
      let structuredData: DayPlanData[] = [];
      
      // Handle the corrected response format
      if (response.data.structured_plan) {
        console.log('Using structured_plan');
        structuredData = parseCarePlanResponse(response.data.structured_plan);
      } else if (response.data.care_plan) {
        console.log('Using care_plan text');
        structuredData = parseCarePlanResponse(response.data.care_plan);
      } else {
        console.log('No valid data found:', response.data);
        throw new Error("No valid care plan data received from server");
      }
      
      console.log('Parsed structured data length:', structuredData.length);
      
      if (structuredData.length === 0) {
        throw new Error("Failed to parse care plan data. Please try again.");
      }
      
      setPlanData(structuredData);

      // Send notification for today's plan
      if (currentDay <= structuredData.length) {
        const todayPlan = structuredData[currentDay - 1]?.items
          .map(item => `${item.icon} ${item.title}: ${item.description}`)
          .join("\n");

        if (todayPlan) {
          await schedulePushNotification(currentDay, todayPlan);
        }
      }
      
    } else {
      // Handle error response
      const errorMessage = response.error || 
                          response.data?.error || 
                          "Failed to generate care plan.";
      throw new Error(errorMessage);
    }
  } catch (err: any) {
    console.error('Care plan generation error:', err);
    const errorMsg = err.message || "An unexpected error occurred";
    setError(errorMsg);
    Alert.alert('Generation Failed', errorMsg);
  } finally {
    setIsLoading(false);
    setRefreshing(false);
  }
}, [currentDay]);

const onRefresh = useCallback(() => {
    setRefreshing(true);
    handleGeneratePlan();
  }, [handleGeneratePlan]);

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
            <Text style={styles.currentDayText}>Current Day: {currentDay}</Text>
          </View>
        ) : null}
        refreshControl={
          <RefreshControl 
            refreshing={refreshing} 
            onRefresh={onRefresh} 
            tintColor="#1D3557" 
          />
        }
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
  currentDayText: { fontSize: 16, color: '#1D3557', fontWeight: '600', marginTop: 5 },
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
  toggleButton: {
    backgroundColor: '#E3F2FD',
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 15,
    alignItems: 'center',
    marginBottom: 15,
  },
  toggleButtonText: {
    color: '#1976D2',
    fontWeight: '600',
    fontSize: 14,
  },
  rawContentContainer: {
    maxHeight: 300,
    borderWidth: 1,
    borderColor: '#E0E0E0',
    borderRadius: 8,
    padding: 12,
    marginBottom: 15,
    backgroundColor: '#FAFAFA',
  },
  rawContentText: {
    fontSize: 14,
    color: '#424242',
    lineHeight: 20,
  },
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
  },
  completionSection: {
    marginTop: 20,
    paddingTop: 15,
    borderTopWidth: 1,
    borderTopColor: '#EEE',
  },
  completionQuestion: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#2C3E50',
  },
  completionButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  completionButton: {
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
    minWidth: 80,
    alignItems: 'center',
  },
  yesButton: {
    backgroundColor: '#27ae60',
  },
  noButton: {
    backgroundColor: '#e74c3c',
  },
  completionButtonText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
  },
  emptyDayText: {
    fontSize: 14,
    color: '#6c757d',
    fontStyle: 'italic',
    textAlign: 'center',
    marginVertical: 10,
  },
});