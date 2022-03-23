"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import random
from utils import movies, ratings, create_user_vector, lookup_movieId, create_R_matrix, create_model
from scipy.sparse import csr_matrix

def recommend_random(k=10):
    """
    return k random unseen movies for user 
    (note: the version below also considers seen movies)
    """
    all_movies = list(movies['title'])
    random_movies = random.choices(all_movies,k=k)
    return random_movies

def recommend_most_popular(k=10):
    """
    return k movies from list of 50 best rated movies unseen for user
    """
    movie_ids = list(ratings.sort_values("rating", ascending=False).head(k)['movieId'])
    popular_movies = [lookup_movieId(movies, movie_id) for movie_id in movie_ids]
    return popular_movies

def recommend_with_user_similarity(user_item_matrix, user_rating, k=5):
    pass

def recommend_nmf_transform(query, model, n_matrix_cols, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model.
    Returns a list of k movie ids.
    """
    # 1. candiate generation
    user_vec, movie_ids = create_user_vector(query, movies, n_matrix_cols)
    # 2. scoring
    # calculate the score with the NMF model
    scores = model.inverse_transform(model.transform(user_vec))
    scores=pd.Series(scores[0])
    scores[movie_ids] = 0
    # 3. ranking
    scores = scores.sort_values(ascending=False)
    # filter out movies allready seen by the user
    # allready_seen=scores.index.isin(query.keys())
    # scores.loc[allready_seen]=0
    # return the top-k highst rated movie ids or titles
    recommendations=scores.head(k).index
    recommendations_df=movies.set_index("movieId").loc[recommendations]['title']
    return recommendations_df

def recommend_nmf(user_rating, k = 10) :
    R = create_R_matrix()
    model = create_model(R, n_components=55, overwrite=False)
    return recommend_nmf_transform(user_rating, model, R.shape[1], k)


