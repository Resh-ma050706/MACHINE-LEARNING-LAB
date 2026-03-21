import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
data_path = r"C:\Users\ASUS\Downloads\exp9.2\ml-latest-small"
print("S.RESHMA 24BAD097")
ratings = pd.read_csv(data_path + r"\ratings.csv")
movies = pd.read_csv(data_path + r"\movies.csv")
df = pd.merge(ratings, movies, on="movieId")
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
user_item_matrix = df.pivot_table(index="userId",
                                 columns="title",
                                 values="rating")
user_item_filled = user_item_matrix.fillna(0)
user_similarity = cosine_similarity(user_item_filled)

user_similarity_df = pd.DataFrame(user_similarity,
                                 index=user_item_matrix.index,
                                 columns=user_item_matrix.index)
def get_similar_users(user_id, n=5):
    sim_users = user_similarity_df[user_id].sort_values(ascending=False)
    return sim_users.iloc[1:n+1]
print("\nTop similar users for User 1:")
print(get_similar_users(1))
def predict_ratings(user_id, n_neighbors=5):
    similar_users = get_similar_users(user_id, n_neighbors)
    user_ratings = user_item_matrix.loc[user_id]
    predictions = {}
    for movie in user_item_matrix.columns:
        if np.isnan(user_ratings[movie]):  # unseen movie
            weighted_sum = 0
            similarity_sum = 0
            for sim_user, sim_score in similar_users.items():
                rating = user_item_matrix.loc[sim_user, movie]
                if not np.isnan(rating):
                    weighted_sum += sim_score * rating
                    similarity_sum += sim_score
            if similarity_sum != 0:
                predictions[movie] = weighted_sum / similarity_sum
    return predictions
def recommend_movies(user_id, n=5):
    preds = predict_ratings(user_id)
    sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
    return sorted_preds[:n]
print("\nTop Recommendations for User 1:")
recs = recommend_movies(1, 5)
for movie, rating in recs:
    print(movie, "-> Predicted Rating:", round(rating, 2))
actual = []
predicted = []
for user in user_item_matrix.index[:50]:  # sample users
    preds = predict_ratings(user)
    for movie, pred in preds.items():
        actual_rating = user_item_matrix.loc[user, movie]
        if not np.isnan(actual_rating):
            actual.append(actual_rating)
            predicted.append(pred)
if len(actual) > 0:
    rmse = sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    print("\nEvaluation Metrics:")
    print("RMSE:", rmse)
    print("MAE:", mae)
total_values = user_item_matrix.size
missing_values = user_item_matrix.isnull().sum().sum()
sparsity = (missing_values / total_values) * 100
print("\nMatrix Sparsity:", round(sparsity, 2), "%")
plt.figure(figsize=(10,6))
sns.heatmap(user_item_matrix.iloc[:20, :20])
plt.title("User-Item Matrix Heatmap")
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(user_similarity_df.iloc[:20, :20])
plt.title("User Similarity Matrix")
plt.show()
movies_list = [x[0] for x in recs]
ratings_list = [x[1] for x in recs]
plt.figure()
plt.barh(movies_list, ratings_list)
plt.xlabel("Predicted Rating")
plt.title("Top Recommendations for User 1")
plt.gca().invert_yaxis()
plt.show()