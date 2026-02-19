import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
print("S.RESHMA 24BAD097")
# 2. Load Dataset
file_path = r"C:\Users\ASUS\Downloads\exp4.1\spam.csv"
data = pd.read_csv(file_path, encoding='latin-1')
# Keep required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
print("Dataset Preview:")
print(data.head())
# 3. Text Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text
data['message'] = data['message'].apply(clean_text)
# 4. Convert Text to Numerical Features (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])
# 5. Encode Labels
encoder = LabelEncoder()
y = encoder.fit_transform(data['label'])  # ham=0, spam=1
# 6. Train-Test Split (store indices for misclassification)
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, y, data.index, test_size=0.2, random_state=42
)
# 7. Train Multinomial Naïve Bayes
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)
# 8. Predictions
y_pred = model.predict(X_test)
# 9. Model Evaluation
print("\nModel Performance:\n")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Multinomial NB")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# 10. Misclassified Samples
misclassified = data.loc[test_idx][y_test != y_pred]
print("\nMisclassified Messages:\n")
print(misclassified.head())

# 11. Feature Importance (Top Spam Words)
feature_names = vectorizer.get_feature_names_out()
top_spam_words = np.argsort(model.feature_log_prob_[1])[-10:]

print("\nTop Words Influencing Spam Prediction:")
for i in top_spam_words:
    print(feature_names[i])

# 12. Word Frequency Comparison (Spam vs Ham)
spam_words = data[data['label'] == 'spam']['message']
ham_words  = data[data['label'] == 'ham']['message']
spam_vec = vectorizer.transform(spam_words)
ham_vec  = vectorizer.transform(ham_words)
spam_freq = np.asarray(spam_vec.sum(axis=0)).flatten()
ham_freq  = np.asarray(ham_vec.sum(axis=0)).flatten()
top_spam_idx = spam_freq.argsort()[-10:]
top_ham_idx  = ham_freq.argsort()[-10:]
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.barh([feature_names[i] for i in top_spam_idx], spam_freq[top_spam_idx])
plt.title("Top 10 Frequent Words in Spam")
plt.subplot(1,2,2)
plt.barh([feature_names[i] for i in top_ham_idx], ham_freq[top_ham_idx])
plt.title("Top 10 Frequent Words in Ham")
plt.show()
# 13. Laplace Smoothing Impact Comparison
model_no_smooth = MultinomialNB(alpha=0)
model_no_smooth.fit(X_train, y_train)
y_pred_no = model_no_smooth.predict(X_test)
print("\nWithout Laplace Smoothing (alpha=0):")
print("Accuracy :", accuracy_score(y_test, y_pred_no))
print("Precision:", precision_score(y_test, y_pred_no))
print("Recall   :", recall_score(y_test, y_pred_no))
print("F1 Score :", f1_score(y_test, y_pred_no))
print("\nWith Laplace Smoothing (alpha=1):")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

