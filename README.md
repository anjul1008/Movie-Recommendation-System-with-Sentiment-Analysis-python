# Content-Based-Movie-Recommender-System-with-sentiment-analysis

Content Based Recommender System recommends movies similar to the movie user likes and analyses the sentiments on the reviews given by the user for that movie.

## How to run the project?

1. Install all the libraries mentioned in the [requirements.txt](https://anjul1008/Movie-Recommendation-System-with-Sentiment-Analysis-python/blob/master/requirements.txt) file.
2. Clone this repository in your local system.
3. Open the command prompt from your project directory and run the command `python app.py`.
4. Go to your browser and type `http://127.0.0.1:5000/apidocs/` in the address bar.
6. Hurray! That's it.

## Dockerize your Application
1.Use **Dockerfile** to create a Docker Image.

```sh
FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ \
        bzip2 \
        unzip \
        make \
        wget \
        git \
        python3 \
        python3-dev \
        python3-websockets \
        python3-setuptools \
        python3-pip \
        python3-wheel \
        zlib1g-dev \
        patch \
        ca-certificates \
        swig \
        cmake \
        xz-utils \
        automake \
        autoconf \
        libtool \
        pkg-config 

COPY . /app/server/.

WORKDIR /app/server

RUN pip3 install -r requirements.txt

CMD python3 app_flasgger.py

```

2.Build the Docker Image

`docker build -t {Name_of_your_Image}`

3.Run your Docker Image

`docker run -p 5000:5000 bank_auth`

**Note**: bank_auth is a docker image name, you can name it anything!!

Now your Docker Container is running at ` http://127.0.0.1:5000/docs` and this is the Swagger UI.


## Similarity Score : 

   How does it decide which item is most similar to the item user likes? Here we use the similarity scores.
   
   It is a numerical value ranges between zero to one which helps to determine how much two items are similar to each other on a scale of zero to one. This similarity score is obtained measuring the similarity between the text details of both of the items. So, similarity score is the measure of similarity between given text details of two items. This can be done by cosine-similarity.
   
## How Cosine Similarity works?
  Cosine similarity is a metric used to measure how similar the documents are irrespective of their size. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance (due to the size of the document), chances are they may still be oriented closer together. The smaller the angle, higher the cosine similarity.
  
  ![image](https://user-images.githubusercontent.com/36665975/70401457-a7530680-1a55-11ea-9158-97d4e8515ca4.png)

  
More about Cosine Similarity : [Understanding the Math behind Cosine Similarity](https://www.machinelearningplus.com/nlp/cosine-similarity/)

### Sources of the datasets 

1. [IMDB 5000 Movie Dataset](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset)
2. [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset)
3. [List of movies in 2018](https://en.wikipedia.org/wiki/List_of_American_films_of_2018)
4. [List of movies in 2019](https://en.wikipedia.org/wiki/List_of_American_films_of_2019)
5. [List of movies in 2020](https://en.wikipedia.org/wiki/List_of_American_films_of_2020)

