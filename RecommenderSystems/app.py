from flask import Flask, render_template, request
from recommender import recommend_most_popular, recommend_nmf, recommend_random
from utils import movies, ratings, print_movie_titles, create_R_matrix, create_model, print_movie_titles, match_movie_title

# construct our flask instance, pass name of module
app = Flask(__name__)

# example input of web application
user_rating = {
    'the lion king': 5,
    'terminator': 5,
    'star wars': 2
}

# route decorator for mapping urls to functions


@app.route('/')
def hello_world():
    # jinja is the templating engine
    return render_template('index.html', name='Beatiful people', movies=movies['title'].tolist())

# ?title=star+wars&rating=5


@app.route('/recommendations')
def recommendations():
    # read user input from url
    print(request.args)

    titles = request.args.getlist("title")
    inp_ratings = request.args.getlist("rating")

    print(titles, ratings)

    user_rating = dict(zip(titles, inp_ratings))

    print(user_rating)

    # recs = recommend_random(movies, user_rating, k=5)
    R = create_R_matrix(ratings)
    model = create_model(R, n_components=55, overwrite=False)

    recs = recommend_nmf(user_rating, model, R.shape[1], k=20)
    # recs = recommend_most_popular(user_rating, movies, ratings, k=20)

    return render_template('recommendations.html', recs=recs)


# only run the app if we are in the main module
if (__name__ == '__main__'):
    app.run(debug=True, port=5000)
