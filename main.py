

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('covtype.csv')

print("Dataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nTarget variable distribution:")
print(df['Cover_Type'].value_counts().sort_index())
print("\nMissing values:")
print(df.isnull().sum().sum())

plt.figure(figsize=(10, 6))
df['Cover_Type'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Cover Type')
plt.ylabel('Count')
plt.title('Distribution of Forest Cover Types')
plt.xticks(rotation=0)
plt.savefig('task3_target_distribution.png')
plt.show()

X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type'] - 1

print(f"\nNumber of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")


if len(X) > 100000:
    print("\nUsing a sample of 50,000 records for faster training...")
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=50000, stratify=y, random_state=42)
    X, y = X_sample, y_sample

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")


print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n=== Random Forest Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(y.unique()), 
            yticklabels=sorted(y.unique()))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.savefig('task3_confusion_matrix.png')
plt.show()

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== Top 15 Most Important Features ===")
print(feature_importance.head(15))

plt.figure(figsize=(12, 8))
plt.barh(feature_importance.head(15)['Feature'], feature_importance.head(15)['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 15 Feature Importance - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('task3_feature_importance.png')
plt.show()

print("\n=== BONUS: Training XGBoost model ===")
xgb_model = XGBClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

print("\n=== XGBoost Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'Accuracy': [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_xgb)]
})

print("\n=== Model Comparison ===")
print(comparison)

plt.figure(figsize=(8, 6))
plt.bar(comparison['Model'], comparison['Accuracy'], color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Model Comparison: Random Forest vs XGBoost')
plt.ylim([0, 1])
for i, v in enumerate(comparison['Accuracy']):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
plt.savefig('task3_model_comparison.png')
plt.show()

print("\nTask 3 completed successfully!")
