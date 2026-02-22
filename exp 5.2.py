# 1. Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# -------------------------------
# 2. Load Dataset
# -------------------------------
print("S.RESHMA 24BAD097")
path = r"C:\Users\ASUS\Downloads\exp5.2\train_u6lujuX_CVtuZ9i (1).csv"
df = pd.read_csv(path)
print("\nFirst 5 rows:")
print(df.head())
# -------------------------------
# 3. Data Inspection
# -------------------------------
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
# -------------------------------
# 4. Handle Missing Values
# -------------------------------
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
# -------------------------------
# 5. Encode Categorical Variables
# -------------------------------

le = LabelEncoder()

cat_cols = ['Gender','Married','Dependents','Education',
            'Self_Employed','Property_Area','Loan_Status']

for col in cat_cols:
    df[col] = le.fit_transform(df[col])
# -------------------------------
# 6. Feature Selection
# -------------------------------
features = ['ApplicantIncome','LoanAmount',
            'Credit_History','Education','Property_Area']
X = df[features]
y = df['Loan_Status']
# -------------------------------
# 7. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)# -------------------------------
# 8. Train Decision Tree Classifier
# -------------------------------
dt = DecisionTreeClassifier(criterion='gini', random_state=42)
dt.fit(X_train, y_train)
# -------------------------------
# 9. Predictions
# -------------------------------
y_pred = dt.predict(X_test)
# -------------------------------
# 10. Performance Evaluation
# -------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# -------------------------------
# 11. Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Rejected','Approved'],
            yticklabels=['Rejected','Approved'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
# -------------------------------
# 12. Feature Importance Plot
# -------------------------------
importances = dt.feature_importances_
plt.figure(figsize=(7,4))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
# -------------------------------
# 13. Tree Depth Experiment (Detect Overfitting)
# -------------------------------
train_acc = []
test_acc = []
depth_values = range(1, 21)
for d in depth_values:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    train_acc.append(model.score(X_train, y_train))
    test_acc.append(model.score(X_test, y_test))
# Plot
plt.figure(figsize=(8,5))
plt.plot(depth_values, train_acc, label='Training Accuracy')
plt.plot(depth_values, test_acc, label='Testing Accuracy')
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Tree Depth vs Model Performance")
plt.legend()
plt.grid(True)
plt.show()
# -------------------------------
# 14. Compare Shallow vs Deep Tree
# -------------------------------
shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
deep = DecisionTreeClassifier(max_depth=15, random_state=42)
shallow.fit(X_train, y_train)
deep.fit(X_train, y_train)
print("\nShallow Tree Accuracy:", shallow.score(X_test, y_test))
print("Deep Tree Accuracy:", deep.score(X_test, y_test))
# -------------------------------
# 15. Tree Structure Plot
# -------------------------------
plt.figure(figsize=(20,10))
plot_tree(shallow, feature_names=features,
          class_names=['Rejected','Approved'],
          filled=True)
plt.title("Decision Tree Structure (Depth=3)")
plt.show()