import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
file_path = r"C:\Users\ASUS\Downloads\churn_boosting.csv"
df = pd.read_csv(file_path)
print("S.RESHMA 24BAD097")
print("Initial rows:", len(df))
print("Original Churn values:", df['Churn'].unique())
df['Churn'] = df['Churn'].astype(str).str.strip().str.lower()
df['Churn'] = df['Churn'].replace({
    'yes': 1, 'no': 0,
    '1': 1, '0': 0,
    'true': 1, 'false': 0})
df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')
print("After mapping:", df['Churn'].unique())
df = df.dropna(subset=['Churn'])
print("Rows after cleaning:", len(df))
if len(df) == 0:
    raise ValueError("Dataset became empty after cleaning. Check Churn values.")
df = df.fillna(df.mean(numeric_only=True))
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
X = df.drop("Churn", axis=1)
y = df["Churn"]
if len(X) == 0:
    raise ValueError("No data available after preprocessing.")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
y_prob_ada = ada_model.predict_proba(X_test)[:, 1]
ada_accuracy = accuracy_score(y_test, y_pred_ada)
print("AdaBoost Accuracy:", ada_accuracy)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
y_prob_gb = gb_model.predict_proba(X_test)[:, 1]
gb_accuracy = accuracy_score(y_test, y_pred_gb)
print("Gradient Boosting Accuracy:", gb_accuracy)
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_prob_ada)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)
auc_ada = auc(fpr_ada, tpr_ada)
auc_gb = auc(fpr_gb, tpr_gb)
plt.figure()
plt.plot(fpr_ada, tpr_ada, label=f"AdaBoost (AUC = {auc_ada:.2f})")
plt.plot(fpr_gb, tpr_gb, label=f"Gradient Boosting (AUC = {auc_gb:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
importances = gb_model.feature_importances_
features = X.columns
indices = np.argsort(importances)
plt.figure()
plt.barh(features[indices], importances[indices])
plt.xlabel("Importance")
plt.title("Feature Importance - Gradient Boosting")
plt.show()