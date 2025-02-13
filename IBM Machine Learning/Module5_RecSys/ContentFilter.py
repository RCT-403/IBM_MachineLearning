import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False) # expand=False means that we extract a series
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False) # Remove the parenthesis in 'year'
movies_df['title'] = movies_df.title.str.replace(r'(\(\d{4}\))', '', regex=True) # Remove the (year) in the titles
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip()) # Remove the white space in title

movies_df['genres'] = movies_df.genres.str.split('|')

# Create a one hot encoding of the genres
moviesWithGenres_df = movies_df.copy()

# For every row, genre add a 1
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
        
# Fill the rest of the empty cells with 0
moviesWithGenres_df = moviesWithGenres_df.fillna(0)

# Drop the genres col
moviesWithGenres_df = moviesWithGenres_df.drop('genres', axis=1)

# Drop the time stamp
ratings_df = ratings_df.drop('timestamp', axis=1)

# Create user df
userInput = [
            {'title':'Toy Story', 'rating':5},
            {'title':'Cars', 'rating':4.5},
            {'title':'Lion King, The', 'rating':5},
            {'title':'The Lego Movie', 'rating':5},
            {'title':'Inside Out', 'rating': 4.5}, 
            {'title':'Toy Story 3', 'rating': 5},
            {'title':'Spirited Away (Sen to Chihiro no kamikakushi)', 'rating': 4.5},
            {'title':'WALL·E', 'rating':5}
        ] 
inputMovies = pd.DataFrame(userInput)

# inputMovies gets the movieIds and titles
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
inputMovies = pd.merge(inputId, inputMovies)
inputMovies = inputMovies.drop('genres', axis=1).drop('year', axis=1)

# Gets just the genres of the user movies
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies = userMovies.reset_index(drop=True)
userGenreTable = userMovies.drop('movieId', axis=1).drop('title', axis=1).drop('year', axis=1)

# Get the weighted genre matrix by multiplying the ratings vector
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

# Get the genres of all the movies with the movie ID as the idx
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
genreTable = genreTable.drop('movieId', axis=1).drop('title', axis=1).drop('year', axis=1)

# Create the recommendation list
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
recommendationTable_df = recommendationTable_df.to_frame()
recommendationTable_df.columns = ['weighted average']
recommendationTable_df = recommendationTable_df.merge(movies_df, how='inner', on = 'movieId')

print(recommendationTable_df.head(10))