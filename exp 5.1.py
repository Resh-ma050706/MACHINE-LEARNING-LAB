# 1. Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# -------------------------------
# 2. Load Dataset
# -------------------------------
print("S.RESHMA 24BAD097")
path = r"C:\Users\ASUS\Downloads\exp5.1\breast-cancer.csv"
df = pd.read_csv(path)
print("\nFirst 5 rows of dataset:")
print(df.head())
# -------------------------------
# 3. Data Inspection & Preprocessing
# -------------------------------
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
# Drop unwanted columns if present
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
# -------------------------------
# 4. Encode Target Labels
# -------------------------------
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])
# M = 1 (Malignant), B = 0 (Benign)
print("\nEncoded Target Classes:")
print(df['diagnosis'].value_counts())
# -------------------------------
# 5. Feature Selection
# -------------------------------
features = ['radius_mean', 'texture_mean', 'perimeter_mean',
            'area_mean', 'smoothness_mean']
X = df[features]
y = df['diagnosis']
# -------------------------------
# 6. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# -------------------------------
# 7. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)
# -------------------------------
# 8. Train KNN Classifier & Experiment with K
# -------------------------------
k_values = range(1, 21)
accuracy_list = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, y_pred))
# -------------------------------
# 9. Best K Value
# -------------------------------
best_k = k_values[np.argmax(accuracy_list)]
print("\nBest K value:", best_k)
print("Best Accuracy:", max(accuracy_list))
# -------------------------------
# 10. Final Model Training
# -------------------------------
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
y_pred_final = knn_final.predict(X_test)
# -------------------------------
# 11. Performance Evaluation
# -------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred_final))
print("\nClassification Report:\n", classification_report(y_test, y_pred_final))
# -------------------------------
# 12. Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign','Malignant'],
            yticklabels=['Benign','Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
# -------------------------------
# 13. Accuracy vs K Plot
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(k_values, accuracy_list, marker='o')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.grid(True)
plt.show()
# -------------------------------
# 14. Identify Misclassified Samples
# -------------------------------
misclassified = np.where(y_test != y_pred_final)[0]
print("\nNumber of Misclassified Samples:", len(misclassified))
print("\nSample Misclassified Cases:")
print(misclassified[:10])
# -------------------------------
# 15. Decision Boundary Plot (2 Features Only)
# -------------------------------
# Using only radius & texture for visualization
X2 = df[['radius_mean','texture_mean']]
y2 = df['diagnosis']
X2_scaled = scaler.fit_transform(X2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2_scaled, y2, test_size=0.25, random_state=42
)
knn_vis = KNeighborsClassifier(n_neighbors=best_k)
knn_vis.fit(X_train2, y_train2)
# Meshgrid
x_min, x_max = X2_scaled[:, 0].min() - 1, X2_scaled[:, 0].max() + 1
y_min, y_max = X2_scaled[:, 1].min() - 1, X2_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X2_scaled[:,0], X2_scaled[:,1], c=y2, edgecolor='k')
plt.xlabel("Radius")
plt.ylabel("Texture")
plt.title("Decision Boundary (KNN)")
plt.show()
