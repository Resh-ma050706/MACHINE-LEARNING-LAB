import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt
data_path = r"C:\Users\ASUS\Downloads\exp9.2\ml-latest-small"
print("S.RESHMA 24BAD097")
ratings = pd.read_csv(data_path + r"\ratings.csv")
movies = pd.read_csv(data_path + r"\movies.csv")
df = pd.merge(ratings, movies, on="movieId")
print(df.head())
item_user_matrix = df.pivot_table(index="title",
                                 columns="userId",
                                 values="rating")
item_user_filled = item_user_matrix.fillna(0)
item_similarity = cosine_similarity(item_user_filled)
item_similarity_df = pd.DataFrame(item_similarity,
                                 index=item_user_matrix.index,
                                 columns=item_user_matrix.index)
def get_similar_items(movie_name, n=5):
    similar = item_similarity_df[movie_name].sort_values(ascending=False)
    return similar.iloc[1:n+1]
print("\nTop similar movies to 'Toy Story (1995)':")
print(get_similar_items("Toy Story (1995)"))
def recommend_items(user_id, n=5):
    user_data = df[df['userId'] == user_id]
    watched_movies = user_data['title'].tolist()
    scores = {}
    for movie in watched_movies:
        similar_movies = get_similar_items(movie, n=5)
        for sim_movie, score in similar_movies.items():
            if sim_movie not in watched_movies:
                if sim_movie not in scores:
                    scores[sim_movie] = 0
                scores[sim_movie] += score
    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return recommended[:n]
print("\nTop recommendations for User 1:")
recs = recommend_items(1)
for movie, score in recs:
    print(movie, "-> Score:", round(score, 3))
actual = []
predicted = []
for user in df['userId'].unique()[:50]:  # sample users
    user_data = df[df['userId'] == user]
    for _, row in user_data.iterrows():
        movie = row['title']
        actual_rating = row['rating']
        try:
            sim_movies = get_similar_items(movie, 5)
            pred_rating = sim_movies.mean()
            actual.append(actual_rating)
            predicted.append(pred_rating)
        except:
            continue
if len(actual) > 0:
    rmse = sqrt(mean_squared_error(actual, predicted))
    print("\nRMSE:", rmse)
def precision_at_k(user_id, k=5):
    recommended = [x[0] for x in recommend_items(user_id, k)]
    actual_movies = df[df['userId'] == user_id]['title'].tolist()
    relevant = 0
    for movie in recommended:
        if movie in actual_movies:
            relevant += 1
    return relevant / k
print("\nPrecision@5 for User 1:", precision_at_k(1))
plt.figure(figsize=(10,6))
sns.heatmap(item_similarity_df.iloc[:20, :20])
plt.title("Item Similarity Heatmap")
plt.show()
movie_name = "Toy Story (1995)"
similar_items = get_similar_items(movie_name, 5)
plt.figure()
plt.barh(similar_items.index, similar_items.values)
plt.title("Top Similar Movies to " + movie_name)
plt.xlabel("Similarity Score")
plt.gca().invert_yaxis()
plt.show()
movies_list = [x[0] for x in recs]
scores = [x[1] for x in recs]
plt.figure()
plt.barh(movies_list, scores)
plt.title("Recommended Movies for User 1")
plt.xlabel("Score")
plt.gca().invert_yaxis()
plt.show()