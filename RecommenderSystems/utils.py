"""
UTILS 
- Helper functions to use for your recommender funcions, etc
- Data: import files/models here e.g.
    - movies: list of movie titles and assigned cluster
    - ratings
    - user_item_matrix
    - item-item matrix 
- Models:
    - nmf_model: trained sklearn NMF model
"""
import os.path
import pickle

import numpy as np
import pandas as pd
from fuzzywuzzy import process
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

curr_file_dir = os.path.dirname(os.path.realpath(__file__))

movies = pd.read_csv('data/ml-latest-small/movies.csv')  
ratings = pd.read_csv('data/ml-latest-small/ratings.csv')

def create_R_matrix() :

    return csr_matrix((ratings['rating'], (ratings['userId'], ratings['movieId'])))

def create_model(R, n_components = 55, overwrite = False) :

    file_path = f'{curr_file_dir}/nmf_recommender.pkl'

    if(os.path.exists(file_path) and not overwrite) :
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
    else:
        model = NMF(n_components, init='nndsvd', max_iter=10000, tol=0.0001)
        # fit it to the user-item rating matrix
        model.fit(R)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

    return model


def match_movie_title(input_title, movie_titles):
    """
    Matches inputed movie title to existing one in the list with fuzzywuzzy
    """
    matched_title = process.extractOne(input_title, movie_titles)[0]

    return matched_title


def print_movie_titles(movie_titles):
    """
    Prints list of movie titles in cli app
    """    
    for movie_title in movie_titles:
        print(f'> {movie_title}')
    pass


def create_user_vector(user_rating, movies, n_matrix_cols):
    """
    Convert dict of user_ratings to a user_vector
    """       
    # generate the user vector

    user_ratings_mod  = dict()
    print(user_rating)
    for movie_title in user_rating.keys():
        # movie_title_mod  =  match_movie_title(input_title = movie_title, movie_titles=movies['title'])
        user_ratings_mod[movies.loc[movies['title'] == movie_title]['movieId'].iloc[0]]= int(user_rating[movie_title])

     # 1. candiate generation
    data = list(user_ratings_mod.values())            # the ratings of the new user
    row_ind = [0]*len(data)        # we use just a single row 0 for this user
    col_ind = list(user_ratings_mod.keys())          # the columns (=movieId) of the ratings

    print(user_ratings_mod.keys())

    # construct a user vector
    return csr_matrix((data, (row_ind, col_ind)), shape=(1,n_matrix_cols)), user_ratings_mod.keys()



def lookup_movieId(movies, movieId):
    """
    Convert output of recommendation to movie title
    """
    # match movieId to title
    # match movieId to title
    movies = movies.reset_index()
    boolean = movies["movieId"] == movieId
    movie_title = list(movies[boolean]["title"])[0]
    return movie_title

    return movie_title
