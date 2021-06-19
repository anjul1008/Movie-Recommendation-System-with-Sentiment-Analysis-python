import pandas as pd
from flask import Flask, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from flasgger import Swagger

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

streamHandlerFormatter = logging.Formatter('[ %(asctime)s - %(funcName)s():%(module)s.py:%(lineno)d - %(levelname)5s ] : %(message)s ', datefmt='%m/%d/%Y %I:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(streamHandlerFormatter)
logger.addHandler(stream_handler)

app = Flask(__name__)
Swagger(app)

def create_sim():
    data = pd.read_csv('transformed_combined_final.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix)
    return data,sim

def rcmd(in_movie):
    in_movie = in_movie.lower()

    data, sim = create_sim()
    if in_movie not in data['movie_title'].unique():
        return False, ('Sorry! The movie your searched is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==in_movie].index[0]
        lst = list(enumerate(sim[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11]
        segested_movie = []
        for i in range(len(lst)):
            a = lst[i][0]
            segested_movie.append(data['movie_title'][a])
        return True, segested_movie


@app.route("/", methods=['GET'])
def home():
    """Movie Recommendation System 
    It's very intresting, Let's try with a movie name.
    It will work only for Hollywood Movies.
    ---
    parameters:  
      - name: movie_name
        in: query
        type: string
        required: true
    responses:
        200:
            description: The output values
        
    """

    movie_name = request.args.get('movie_name')
    logger.info('Searched movie:{}'.format(movie_name))
    # movie_name = 'thor'
    sucess, recommended_movies = rcmd(movie_name)

    if sucess: 
        logger.info('recommended_movies:[{}]'.format(recommended_movies))
        return '\n'.join(recommended_movies)
    else: 
        logger.warning('{}'.format(recommended_movies))
        return recommended_movies
        


if __name__ == '__main__':
    PORT = 8000
    IP = '127.0.0.1'
    DEBUG = False

    # load the nlp model and tfidf vectorizer from disk
    logger.info('Loading classifier model: model_nlp.pkl ')
    filename = 'model_nlp.pkl'
    clf = pickle.load(open(filename, 'rb'))

    logger.info('Loading vectorizer model: model_tranform.pkl ')
    vectorizer = pickle.load(open('model_tranform.pkl','rb'))

    logger.info('Server started on: http://{}:{}/apidocs/'.format(IP, PORT))
    app.run(debug=True)
    
