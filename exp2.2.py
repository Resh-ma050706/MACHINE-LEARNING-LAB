import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc)
print("S.RESHMA 24BAD097")
data = pd.read_csv(r"C:\Users\ASUS\Downloads\exp2.2\LICI - 10 minute data.csv")
print("Dataset Loaded Successfully!")
print(data.head())
print("\nColumns Available:\n", data.columns)
data.columns = [col.strip().capitalize() for col in data.columns]
data["Price_Movement"] = np.where(
    data["Close"] > data["Open"], 1, 0
)
features = ["Open", "High", "Low", "Volume"]
X = data[features]
y = data["Price_Movement"]
X.fillna(X.mean(), inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("\nModel Performance Metrics")
print("-------------------------")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix (10-Min Data)")
plt.show()
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label="AUC = %.2f" % roc_auc)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (10-Min Data)")
plt.legend()
plt.show()
importance = pd.Series(model.coef_[0], index=features)
plt.figure(figsize=(6, 4))
importance.plot(kind="bar")
plt.title("Feature Importance (10-Min Data)")
plt.ylabel("Coefficient Value")
plt.show()
optimized_model = LogisticRegression(
    C=0.5,
    penalty="l2",
    solver="liblinear")
optimized_model.fit(X_train, y_train)
opt_pred = optimized_model.predict(X_test)
print("\nOptimized Model Accuracy:",
      accuracy_score(y_test, opt_pred))