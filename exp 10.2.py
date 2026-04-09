print("S.RESHMA 24BAD097")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
ratings = pd.read_csv(r"C:\Users\ASUS\Downloads\exp 10\ml-100k\u.data", sep='\t',
                      names=['user_id', 'item_id', 'rating', 'timestamp'])
movies = pd.read_csv(r"C:\Users\ASUS\Downloads\exp 10\ml-100k\u.item", sep='|',
                     encoding='latin-1', names=['item_id', 'title'], usecols=[0, 1])
data = pd.merge(ratings, movies, on='item_id').drop('timestamp', axis=1)
user_item_matrix = data.pivot_table(index='user_id',
                                    columns='title',
                                    values='rating',
                                    aggfunc='mean')
user_item_matrix_filled = user_item_matrix.fillna(0)
k = 20
nmf_model = NMF(n_components=k, init='random', random_state=42)
W = nmf_model.fit_transform(user_item_matrix_filled)
H = nmf_model.components_
reconstructed = np.dot(W, H)
predicted_ratings = pd.DataFrame(reconstructed,
                                 columns=user_item_matrix.columns,
                                 index=user_item_matrix.index)
actual = user_item_matrix.values
predicted = predicted_ratings.values
mask = actual > 0
rmse = np.sqrt(mean_squared_error(actual[mask], predicted[mask]))
print("RMSE:", rmse)
def recommend_movies(user_id, n=5):
    user_pred = predicted_ratings.loc[user_id].sort_values(ascending=False)
    rated = user_item_matrix.loc[user_id].dropna().index
    return user_pred.drop(rated).head(n)
top_movies = recommend_movies(1, 5)
print("\nTop Recommended Movies:\n", top_movies)
def precision_recall_at_k(user_id, k=5, threshold=3.5):
    recommended = recommend_movies(user_id, k).index
    actual_rated = user_item_matrix.loc[user_id]
    relevant = actual_rated[actual_rated >= threshold].index
    hits = len(set(recommended) & set(relevant))
    precision = hits / k
    recall = hits / len(relevant) if len(relevant) > 0 else 0
    return precision, recall
precision, recall = precision_recall_at_k(1, 5)
print("Precision@5:", precision)
print("Recall@5:", recall)
plt.figure(figsize=(8,5))
sns.heatmap(W[:20, :10])
plt.title("User Latent Features")
plt.show()
plt.figure(figsize=(10,4))
sns.heatmap(user_item_matrix_filled.iloc[:20, :20])
plt.title("Original Matrix")
plt.show()
plt.figure(figsize=(10,4))
sns.heatmap(predicted_ratings.iloc[:20, :20])
plt.title("Reconstructed Matrix (NMF)")
plt.show()
plt.figure()
top_movies.plot(kind='bar')
plt.title("Top Recommended Movies (NMF)")
plt.ylabel("Predicted Rating")
plt.xticks(rotation=60, ha='right')
plt.show()