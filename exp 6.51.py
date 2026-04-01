import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, accuracy_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
file_path = r"C:\Users\ASUS\Downloads\fraud_smote.csv"
df = pd.read_csv(file_path)
print("S.RESHMA 24BAD097")
X = df.drop("Fraud", axis=1)
y = df["Fraud"]
print("Class distribution before SMOTE:\n", y.value_counts())
plt.figure()
y.value_counts().plot(kind='bar')
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
model_before = LogisticRegression(max_iter=1000)
model_before.fit(X_train, y_train)
y_prob_before = model_before.predict_proba(X_test)[:, 1]
y_pred_before = model_before.predict(X_test)
acc_before = accuracy_score(y_test, y_pred_before)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:\n", pd.Series(y_train_sm).value_counts())
plt.figure()
pd.Series(y_train_sm).value_counts().plot(kind='bar')
plt.title("Class Distribution After SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
model_after = LogisticRegression(max_iter=1000)
model_after.fit(X_train_sm, y_train_sm)
y_prob_after = model_after.predict_proba(X_test)[:, 1]
y_pred_after = model_after.predict(X_test)
acc_after = accuracy_score(y_test, y_pred_after)
print("Accuracy Before SMOTE:", acc_before)
print("Accuracy After SMOTE:", acc_after)
precision_b, recall_b, _ = precision_recall_curve(y_test, y_prob_before)
precision_a, recall_a, _ = precision_recall_curve(y_test, y_prob_after)
auc_b = auc(recall_b, precision_b)
auc_a = auc(recall_a, precision_a)
plt.figure()
plt.plot(recall_b, precision_b, label=f"Before SMOTE (AUC = {auc_b:.2f})")
plt.plot(recall_a, precision_a, label=f"After SMOTE (AUC = {auc_a:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()