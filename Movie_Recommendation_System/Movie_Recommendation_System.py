import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
ratings_url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.data'
movies_url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.item'
ratings = pd.read_csv(ratings_url, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
movies = pd.read_csv(movies_url, sep='|', encoding='latin-1', usecols=[0, 1], names=['movie_id', 'title'], header=None)

user_item_matrix = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def get_user_based_recommendations(user_id, num_recommendations=10):
    user_ratings = user_item_matrix.loc[user_id]
    sim_scores = user_similarity_df[user_id]
    weighted_sum = user_item_matrix.T.dot(sim_scores)
    sim_sum = sim_scores.sum()
    weighted_average = weighted_sum / sim_sum
    unrated_movies = user_ratings[user_ratings == 0].index
    recommendations = weighted_average[unrated_movies].sort_values(ascending=False).head(num_recommendations)
    return movies[movies['movie_id'].isin(recommendations.index)]

def get_item_based_recommendations(user_id, num_recommendations=10):
    user_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings == 0].index
    predicted_ratings = {}
    for item in unrated_items:
        sim_scores = item_similarity_df[item]
        weighted_sum = user_ratings.dot(sim_scores)
        sim_sum = sim_scores[user_ratings > 0].sum()
        if sim_sum > 0:
            predicted_ratings[item] = weighted_sum / sim_sum
        else:
            predicted_ratings[item] = 0
    recommendations = pd.Series(predicted_ratings).sort_values(ascending=False).head(num_recommendations)
    return movies[movies['movie_id'].isin(recommendations.index)]
user_based_recommendations = get_user_based_recommendations(1, 10)
print("User-Based Recommendations:")
print(user_based_recommendations)
item_based_recommendations = get_item_based_recommendations(1, 10)
print("Item-Based Recommendations:")
print(item_based_recommendations)
