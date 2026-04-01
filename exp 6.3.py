import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
file_path = r"C:\Users\ASUS\Downloads\income_random_forest.csv"
df = pd.read_csv(file_path)
print("S.RESHMA 24BAD097")
df['Income'] = df['Income'].astype(str).str.strip().str.lower()
df['Income'] = df['Income'].replace({
    '>50k': 1, '<=50k': 0,
    '1': 1, '0': 0})
df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
df = df.dropna(subset=['Income'])
df = df.fillna(df.mean(numeric_only=True))
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
X = df.drop("Income", axis=1)
y = df["Income"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
trees = [10, 50, 100, 150, 200]
accuracies = []
for n in trees:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
print("Accuracies:", accuracies)
plt.figure()
plt.plot(trees, accuracies, marker='o')
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Trees")
plt.show()
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)
importances = final_model.feature_importances_
features = X.columns
indices = np.argsort(importances)
plt.figure()
plt.barh(features[indices], importances[indices])
plt.xlabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.show()