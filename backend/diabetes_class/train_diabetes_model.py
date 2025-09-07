import pandas as pd
import numpy as np
import os
import joblib
import sklearn
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# --- 1. Load Dataset ---
try:
    df = pd.read_excel(r"C:\Users\IYYAPPAN\Desktop\Health_app\backend\diabetes_class\Diabetestype.xlsx")
    print(f"--- Loaded dataset with {len(df)} records ---")
    
    # Data validation
    print("\n--- Missing Values Check ---")
    print(df.isnull().sum())
    
    print("\n--- Class Distribution ---")
    print(df['Diabetes_Type'].value_counts())
    
except FileNotFoundError:
    print("Error: File not found.")
    print("Please ensure the file path is correct.")
    exit()

# --- 2. Prepare Features and Labels ---
X = df.drop('Diabetes_Type', axis=1)
y = df['Diabetes_Type']

categorical_features = ['Family_History', 'Autoantibodies_Status', 'Genetic_Test_Result']
numerical_features = X.select_dtypes(include=np.number).columns

# --- 3. Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

# --- 4. K-Means Clustering for Risk Prediction ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
risk_clusters = kmeans.fit_predict(X_processed)
df['Risk_Cluster'] = risk_clusters

# --- 5. Analyze Cluster Centers for Risk Labeling ---
cluster_centers_num = kmeans.cluster_centers_[:, :len(numerical_features)]
scaler = preprocessor.named_transformers_['num']
cluster_centers_orig = scaler.inverse_transform(cluster_centers_num)

cluster_stats = []
for i, center in enumerate(cluster_centers_orig):
    stats = {
        'cluster': i,
        'mean_Is_Pregnant': center[0],
        'mean_Age_at_Diagnosis': center[1],
        'mean_BMI_at_Diagnosis': center[2],
        'mean_HbA1c': center[3],
        'mean_C_Peptide_Level': center[4]
    }
    cluster_stats.append(stats)

sorted_clusters = sorted(cluster_stats, key=lambda x: x['mean_HbA1c'])
risk_labels = ['Low', 'Medium', 'High']
cluster_risk_map = {c['cluster']: risk_labels[i] for i, c in enumerate(sorted_clusters)}

df['Risk_Level'] = df['Risk_Cluster'].map(cluster_risk_map)

print("--- Cluster Centers and Risk Levels ---")
for c in sorted_clusters:
    print(f"Cluster {c['cluster']}: HbA1c={c['mean_HbA1c']:.2f}, "
          f"Age={c['mean_Age_at_Diagnosis']:.1f}, "
          f"BMI={c['mean_BMI_at_Diagnosis']:.1f}, "
          f"Assigned Risk = {cluster_risk_map[c['cluster']]}")

# --- 6. Build Classification Pipeline ---
lgbm_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=42, n_estimators=200,
                                  max_depth=10, learning_rate=0.05))
])

# --- 7. Train Classification Model ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\n--- Training LightGBM classifier ---")
lgbm_model.fit(X_train, y_train)
print("--- Classification model trained ---\n")

# --- 8. Model Evaluation ---
y_pred = lgbm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy*100:.2f}%")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(lgbm_model, X, y, cv=5, scoring='accuracy')
print(f"\n--- Cross-Validation Scores (5-fold) ---")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance
feature_names = numerical_features.tolist()
cat_encoder = preprocessor.named_transformers_['cat']
feature_names.extend(cat_encoder.get_feature_names_out(categorical_features).tolist())

importances = lgbm_model.named_steps['classifier'].feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n--- Top 10 Feature Importances ---")
print(feature_importance_df.head(10))

# --- 9. Save Models to PKL Files ---
save_dir = r"C:\Users\IYYAPPAN\Desktop\Health_app\backend\diabetes_class"
os.makedirs(save_dir, exist_ok=True)

joblib.dump(preprocessor, os.path.join(save_dir, "preprocessor.pkl"))
joblib.dump(lgbm_model, os.path.join(save_dir, "lgbm_classifier.pkl"))
joblib.dump(kmeans, os.path.join(save_dir, "kmeans_model.pkl"))
joblib.dump(cluster_stats, os.path.join(save_dir, "cluster_stats.pkl"))
joblib.dump(cluster_risk_map, os.path.join(save_dir, "cluster_risk_map.pkl"))

# Also save feature names and metadata
metadata = {
    'feature_names': feature_names,
    'categorical_features': categorical_features,
    'numerical_features': numerical_features.tolist(),
    'model_version': '1.0',
    'scikit_learn_version': sklearn.__version__
}
joblib.dump(metadata, os.path.join(save_dir, "model_metadata.pkl"))

print(f"\n--- Models saved successfully in {save_dir} ---")