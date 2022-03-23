import imdb
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

from recommender import recommend_most_popular, recommend_nmf, recommend_random
from utils import create_model, create_R_matrix, movies

# Default poster used for movies when no poster is available.
imdb_logo_url = 'https://ia.media-imdb.com/images/M/MV5BMTczNjM0NDY0Ml5BMl5BcG5nXkFtZTgwMTk1MzQ2OTE@._V1_.png'

# cache the movies dictionary for later usage
# https://discuss.streamlit.io/t/save-user-input-into-a-dataframe-and-access-later/2527
@st.cache(allow_output_mutation=True)
def get_movies() :
    return dict()

# Create a data frame form movies dictionary which is needed for
# creating the streamlit table from dataframe.
def get_movies_df() :
    movie_rating_list = {"Movie Title" : get_movies().keys(), "Rating" : get_movies().values() }
    return pd.DataFrame(movie_rating_list)


# This function creates the list of links of imdb objects formatted
# for the streamlit markdown objects.
def create_links_markdown(imdb_objs) :
    list_imdb_links = ""
    for element in imdb_objs:
        url = cinemagoer.get_imdbURL(element)
        name = element['name']
        list_imdb_links += f'[{name}]({url}), '
    list_imdb_links = list_imdb_links[:-2]
    return list_imdb_links

if __name__ == '__main__':

    movies_list=movies['title'].tolist()
    # create an instance of the Cinemagoer class
    cinemagoer = imdb.Cinemagoer()

    # Form for adding movies and setting up the.
    main_form = st.form(key='my-form')
    recommender_type = main_form.selectbox(
        'Movie Recommender Type',
        ('NMF', 'MostPopular', 'Random'))
    n_movie_recommendations = main_form.number_input('Number of Recommendations', value = 5)
    selected_movie = main_form.selectbox('Movie', tuple(movies_list))
    movie_rating = main_form.slider(label='Rating', min_value=1, max_value=5, key=4)

    add_more_movies = main_form.form_submit_button('Add Movie')
    delete_movies = main_form.form_submit_button('Clear Movies')
    submit = main_form.form_submit_button('Recommend Movies')

    # These are the streamlit elements for the showing the current
    # state of the user movies and the remcommended movies.
    current_movies_heading = st.empty()
    current_movies_heading.markdown("## Current User Movie Ratings:")
    current_movies = st.empty()
    recommended_movies = st.empty()

    # Update the user movies ratings list when a Add Movie button is 
    # clicked.
    if add_more_movies :
        get_movies()[selected_movie] = movie_rating
        current_movies = st.empty()
        current_movies.table(get_movies_df())

    # Clear the movies rating list when the Clear Movies Button is
    # clicked.
    if delete_movies :
        current_movies = st.empty()
        get_movies().clear()
        current_movies.table(get_movies_df())
        


    if submit:

        current_movies.table(get_movies_df())

        user_rating = get_movies()
        recommended_movie_list = ""

        # Get the recommendations based on the recommender type.
        if(recommender_type == 'Random'):
            recs = recommend_random(k = n_movie_recommendations)
        elif(recommender_type == 'MostPopular'):
            recs = recommend_most_popular(k = n_movie_recommendations)
        elif(recommender_type == 'NMF'):
            if len(user_rating.values()) != 0:
                R = create_R_matrix()
                model = create_model(R, n_components=55, overwrite=False)
                recs = recommend_nmf(user_rating, model, R.shape[1], k = n_movie_recommendations)
            else:
                recs = []
        else:
            recs = recommend_most_popular(n_movie_recommendations)

        
        recommended_movies.markdown(f"""## Recommended Movies Using {recommender_type}: """)

        # When NMF type is used it needs the user movie ratings list 
        # otherwise this condition is used to give feedback to the
        # user to add some movies.
        if len(recs) == 0:
            st.markdown(f"No movies added. Add movies for {recommender_type} recommender to work on something")

        for movie_index, movie in enumerate(recs):

            # Default values of all the details of the current 
            # recommended movie. The value is used when the imdb
            # api is not able to get some details.
            movie_url = ""
            movie_summary = "No Summary"
            movie_director = "Unknown"
            movie_cast = "Unknown"
            movie_cover_url = imdb_logo_url
            movie_genre = "Unknown"

            movie_info = cinemagoer.search_movie(movie)
            
            if len(movie_info) > 0:
                movie_url = cinemagoer.get_imdbURL(movie_info[0])
                movie_obj = cinemagoer.get_movie(movie_info[0].movieID)

                # These conditions are needed because some times imdb doesn't
                # have some realted imformation for a movie
                if 'genre' in movie_obj:
                    movie_genre = movie_obj['genre']
                if 'cover url' in movie_obj:
                    movie_cover_url = movie_obj.data['cover url']
                if 'plot' in movie_obj:
                    movie_summary = movie_obj['plot'][0]
                if 'director' in movie_obj:
                    movie_director = create_links_markdown(movie_obj['director'])
                if 'cast' in movie_obj:
                    movie_cast = create_links_markdown(movie_obj['cast'])

            # st.image(movie_cover_url, width = 100)
            st.markdown(f"  \n  {movie_index + 1}. [{movie}]({movie_url})  \n  ")
            st.markdown(f'''
            <a href="{movie_url}">
                <img src="{movie_cover_url}" width="150" height="225"/>
            </a>''',
            unsafe_allow_html=True
            )
            st.markdown(f'  \n  Genre : {movie_genre}')
            st.markdown(f'  \n  Summary : {movie_summary}')
            st.markdown(f'  \n  Director : {movie_director}')
            st.markdown(f'  \n  Cast : {movie_cast}')

            st.markdown("#")
            st.markdown("#")


