import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("S.RESHMA 24BAD097")
# 2. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target
# Convert to DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
print("Dataset Preview:")
print(df.head())
# 3. Data Inspection
print("\nDataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())
# 4. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
# 6. Train Gaussian Naïve Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
# 7. Predictions
y_pred = gnb.predict(X_test)
# 8. Model Evaluation
print("\nGaussian Naïve Bayes Performance:\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Gaussian NB")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# 9. Compare Predictions with Actual
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted (First 10 Samples):\n")
print(comparison.head(10))
# 10. Analyze Class Probabilities
probs = gnb.predict_proba(X_test[:5])
print("\nClass Probabilities for first 5 samples:\n")
print(probs)
# 11. Decision Boundary Plot (Using two features)
from matplotlib.colors import ListedColormap
X_two = X_scaled[:, [2, 3]]   # Petal length & Petal width
y_two = y
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_two, y_two, test_size=0.2, random_state=42
)
model2 = GaussianNB()
model2.fit(X_train2, y_train2)
x_min, x_max = X_two[:, 0].min() - 1, X_two[:, 0].max() + 1
y_min, y_max = X_two[:, 1].min() - 1, X_two[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = model2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(6,5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['red','green','blue']))
plt.scatter(X_two[:,0], X_two[:,1], c=y_two, edgecolor='k',
            cmap=ListedColormap(['red','green','blue']))
plt.xlabel("Petal Length (scaled)")
plt.ylabel("Petal Width (scaled)")
plt.title("Decision Boundary - Gaussian NB")
plt.show()
# 12. Probability Distribution Plots
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(X[:,2], kde=True)
plt.title("Petal Length Distribution")
plt.subplot(1,2,2)
sns.histplot(X[:,3], kde=True)
plt.title("Petal Width Distribution")
plt.show()
# 13. Compare Gaussian NB with Logistic Regression (Optional)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print("\nComparison with Logistic Regression:\n")
print("Gaussian NB Accuracy   :", accuracy_score(y_test, y_pred))
print("Logistic Reg Accuracy :", accuracy_score(y_test, y_pred_log))
