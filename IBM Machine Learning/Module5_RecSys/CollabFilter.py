import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# WE MAKE USE OF USER-BASED COLLAB FILTER

movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# The titles in movies_df contains the year inside a parenthesis, so we extract this to a new col with year
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False) # expand=False means that we extract a series
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False) # Remove the parenthesis in 'year'
movies_df['title'] = movies_df.title.str.replace(r'(\(\d{4}\))', '', regex=True) # Remove the (year) in the titles
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip()) # Remove the white space in title

# Drop the genre row since we dont need it here
movies_df = movies_df.drop('genres', axis=1)

# Create a dummy user
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

# Add the movieId to the user_df
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())] # Create df from movies_df with the specific titles
inputMovies = pd.merge(inputId, inputMovies, on='title') # Merge based on the title
inputMovies = inputMovies.drop('year', axis=1) 

# Filter the ratings df to only include movies user has watched
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]

# Group the ratings by each user, this is not a df but a groupby object
userSubsetGroup = userSubset.groupby('userId') 

# Rearrange the subset group by the number of movies
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)

# Get the top 100 people
userSubsetGroup = userSubsetGroup[0:100]

# Store the Pearson Correlation in a dictionary
# Where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

# We calculate the pearson correlation for everyone to the user
for name, group in userSubsetGroup:
    
    # Sort the group and user_df the same way
    group = group.sort_values(by='movieId') 
    inputMovies = inputMovies.sort_values(by='movieId')
    
    # Only include the common movies in inputMovies
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    
    # Change into a list to manage easier
    tempRatingList = temp_df['rating'].tolist()
    tempGroupList = group['rating'].tolist()
    
    # Calculate the Pear Cor based on the two lists
    nRatings = len(group)
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0
        
# Translate the Pearson Cor from a dict to a df
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index') # Creates a df with the index as the userID and col with the cor
pearsonDF.columns = ['similarityIndex'] # Names the column
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))

# Get the top 50 users to use for the model
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]

# Find all of the ratings of the top 50 users
topUsersRating = topUsers.merge(ratings_df, how='inner', left_on='userId', right_on='userId')

# Add the weighted ratings 
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']

# Create a data frame by aggregating the groupBy object and only get movies that have at least 10 reviews
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating = tempTopUsersRating[tempTopUsersRating['sum_weightedRating'] > 25]

# Create the final sorted df with all the recommended movies 
recommendation_df = pd.DataFrame()
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.index = range(len(recommendation_df))
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)

# List the top 10 movies that the user might like
num_movies = 50
final_df = movies_df.merge(recommendation_df, how='inner', on = 'movieId').sort_values(by='weighted average recommendation score', ascending=False)
final_df = final_df[~final_df['movieId'].isin(inputMovies['movieId'].tolist())]
final_df.index = range(len(final_df))

# Print the final arrangement
print(final_df.head(num_movies))
