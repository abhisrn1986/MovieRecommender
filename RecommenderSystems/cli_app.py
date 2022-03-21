# example input of web application
from re import A
from recommender import recommend_most_popular, recommend_nmf, recommend_random
from utils import movies, ratings, print_movie_titles, create_R_matrix, create_model, print_movie_titles, match_movie_title

user_rating = {
    'star trek': 5,
    'terminator': 5,
    'star wars': 4,
    'Independence day': 4,
    'The lion king': 1,
    'Toy story': 2
}

# Terminal recommender:

print('>>>> Here are some movie recommendations for you:')

print('>> Based on most popular movies:')
recommended_movies = recommend_most_popular(user_rating, movies, ratings, k=5)
print_movie_titles(recommended_movies)

print('\n\n>> Random recommendations:')
recommended_movies = recommend_random(user_rating, movies, k=5)
print_movie_titles(recommended_movies)

R = create_R_matrix(ratings)
model = create_model(R, n_components=55, overwrite=False)

print('\n\n>> NMF recommendations:')
print_movie_titles(recommend_nmf(user_rating, model, R.shape[1], k=20))