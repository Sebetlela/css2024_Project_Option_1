# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:11:30 2024

@author: DELL
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from sklearn.metrics import r2_score

df = pd.read_csv('movie_dataset.csv')
df.columns = df.columns.str.replace(' ', '_')
df.dropna(inplace = True)
df = df.reset_index(drop=True)

movie_column = 'Title'
rating_column = 'Rating'

# Find the row with the highest rating
highest_rated_movie = df.loc[df[rating_column].idxmax()]

# Print the result
print(f"The highest-rated movie is '{highest_rated_movie[movie_column]}' with a rating of {highest_rated_movie[rating_column]}")

revenue_column = 'Revenue_(Millions)'

# Calculate the average revenue
average_revenue = df[revenue_column].mean()

# Print the result
print(f"The average revenue of all movies is: {average_revenue}")

# Assuming the dataset has columns 'Release_Year' and 'Revenue', replace them with your actual column names
release_year_column = 'Year'
revenue_column = 'Revenue_(Millions)'

# Filter movies released from 2015 to 2017
filtered_df = df[(df[release_year_column] >= 2015) & (df[release_year_column] <= 2017)]

# Calculate the average revenue for the filtered dataset
average_revenue_2015_to_2017 = filtered_df[revenue_column].mean()

# Print the result
print(f"The average revenue of movies from 2015 to 2017 is: {average_revenue_2015_to_2017}")

# Assuming the dataset has a 'Release_Year' column, replace it with your actual column name
release_year_column = 'Year'

# Count the number of movies released in the year 2016
movies_2016_count = len(df[df[release_year_column] == 2016])

# Print the result
print(f"The number of movies released in the year 2016 is: {movies_2016_count}")

# Assuming the dataset has a 'Director' column, replace it with your actual column name
director_column = 'Director'

# Count the number of movies directed by Christopher Nolan
nolan_movies_count = len(df[df[director_column] == 'Christopher Nolan'])

# Print the result
print(f"The number of movies directed by Christopher Nolan is: {nolan_movies_count}")

# Assuming the dataset has a 'Rating' column, replace it with your actual column name
rating_column = 'Rating'

# Count the number of movies with a rating of at least 8.0
highly_rated_movies_count = len(df[df[rating_column] >= 8.0])

# Print the result
print(f"The number of movies with a rating of at least 8.0 is: {highly_rated_movies_count}")

director_column = 'Director'
rating_column = 'Rating'

# Filter movies directed by Christopher Nolan
nolan_movies = df[df[director_column] == 'Christopher Nolan']

# Calculate the median rating for Christopher Nolan's movies
median_rating_nolan = nolan_movies[rating_column].median()

# Print the result
print(f"The median rating of movies directed by Christopher Nolan is: {median_rating_nolan}")

# Assuming the dataset has 'Year' and 'Rating' columns, replace them with your actual column names
release_year_column = 'Year'
rating_column = 'Rating'

# Calculate the average rating for each year
average_ratings_by_year = df.groupby(release_year_column)[rating_column].mean()

# Find the year with the highest average rating
year_highest_avg_rating = average_ratings_by_year.idxmax()
highest_avg_rating = average_ratings_by_year.max()

# Print the result
print(f"The year with the highest average rating is {year_highest_avg_rating} with an average rating of {highest_avg_rating}")

# Assuming the dataset has an 'Actors' column, replace it with your actual column name
actors_column = 'Actors'

# Concatenate all actor names into a single string
all_actors_str = ', '.join(df[actors_column].dropna())

# Split the concatenated string into a list of actors
all_actors_list = [actor.strip() for actor in all_actors_str.split(',')]

# Find the most common actor
most_common_actor = max(set(all_actors_list), key=all_actors_list.count)

# Print the result
print(f"The most common actor in all movies is: {most_common_actor}")

# Assuming the dataset has a 'Genres' column, replace it with your actual column name
genres_column = 'Genre'

# Extract unique genres from the dataset
unique_genres = df[genres_column].str.split(',').explode().str.strip().unique()

# Count the number of unique genres
num_unique_genres = len(unique_genres)

# Print the result
print(f"The number of unique genres in the dataset is: {num_unique_genres}")

# Fit a linear regression model
model = np.polyfit(df['Runtime_(Minutes)'], df['Rating'], 1)
predict = np.poly1d(model)

# Calculate R-squared
r2 = r2_score(df['Rating'], predict(df['Runtime_(Minutes)']))
print('Runtime vs Rating')
print("R-squared:", r2)

# Scatter plot
plt.scatter(df['Runtime_(Minutes)'], df['Rating'], label='Actual data')

# Regression line plot
plt.plot(df['Runtime_(Minutes)'], predict(df['Runtime_(Minutes)']), color='red', label='Regression line')

# Labels and title
plt.xlabel('Runtime_(Minutes)')
plt.ylabel('Rating')
plt.title('Scatter Plot with Regression Line')

# Show legend
plt.legend()

# Display the plot
plt.show()

# Fit a linear regression model
model = np.polyfit(df['Runtime_(Minutes)'], df['Votes'], 1)
predict = np.poly1d(model)

# Calculate R-squared
r2 = r2_score(df['Votes'], predict(df['Runtime_(Minutes)']))
print('Runtime vs Votes')
print("R-squared:", r2)

# Scatter plot
plt.scatter(df['Runtime_(Minutes)'], df['Votes'], label='Actual data')

# Regression line plot
plt.plot(df['Runtime_(Minutes)'], predict(df['Runtime_(Minutes)']), color='red', label='Regression line')

# Labels and title
plt.xlabel('Runtime_(Minutes)')
plt.ylabel('Votes')
plt.title('Scatter Plot with Regression Line')

# Show legend
plt.legend()

# Fit a linear regression model
model = np.polyfit(df['Runtime_(Minutes)'], df['Metascore'], 1)
predict = np.poly1d(model)

# Calculate R-squared
r2 = r2_score(df['Metascore'], predict(df['Runtime_(Minutes)']))
print('Runtime vs Revenue')
print("R-squared:", r2)

# Scatter plot
plt.scatter(df['Runtime_(Minutes)'], df['Metascore'], label='Actual data')

# Regression line plot
plt.plot(df['Runtime_(Minutes)'], predict(df['Metascore']), color='red', label='Regression line')

# Labels and title
plt.xlabel('Runtime_(Minutes)')
plt.ylabel('Metascore')
plt.title('Scatter Plot with Regression Line')

# Show legend
plt.legend()

# Display the plot
plt.show()

# Fit a linear regression model
model = np.polyfit(df['Runtime_(Minutes)'], df['Metascore'], 1)
predict = np.poly1d(model)

# Calculate R-squared
r2 = r2_score(df['Metascore'], predict(df['Runtime_(Minutes)']))
print('Runtime vs Metascore')
print("R-squared:", r2)

# Scatter plot
plt.scatter(df['Runtime_(Minutes)'], df['Metascore'], label='Actual data')

# Regression line plot
plt.plot(df['Runtime_(Minutes)'], predict(df['Runtime_(Minutes)']), color='red', label='Regression line')

# Labels and title
plt.xlabel('Runtime_(Minutes)')
plt.ylabel('Metascore')
plt.title('Scatter Plot with Regression Line')

# Show legend
plt.legend()

# Display the plot
plt.show()


# Fit a linear regression model
model = np.polyfit(df['Rating'], df['Revenue_(Millions)'], 1)
predict = np.poly1d(model)

# Calculate R-squared
r2 = r2_score(df['Revenue_(Millions)'], predict(df['Rating']))
print('Rating vs Revenue')
print("R-squared:", r2)

# Scatter plot
plt.scatter(df['Rating'], df['Revenue_(Millions)'], label='Actual data')

# Regression line plot
plt.plot(df['Rating'], predict(df['Rating']), color='red', label='Regression line')

# Labels and title
plt.xlabel('Rating')
plt.ylabel('Revenue_(Millions)')
plt.title('Scatter Plot with Regression Line')

# Show legend
plt.legend()

# Display the plot
plt.show()

# Fit a linear regression model
model = np.polyfit(df['Rating'], df['Votes'], 1)
predict = np.poly1d(model)

# Calculate R-squared
r2 = r2_score(df['Votes'], predict(df['Rating']))
print('Votes vs Rating')
print("R-squared:", r2)

# Scatter plot
plt.scatter(df['Rating'], df['Votes'], label='Actual data')

# Regression line plot
plt.plot(df['Rating'], predict(df['Rating']), color='red', label='Regression line')

# Labels and title
plt.xlabel('Rating')
plt.ylabel('Votes')
plt.title('Scatter Plot with Regression Line')

# Show legend
plt.legend()

# Display the plot
plt.show()

# Fit a linear regression model
model = np.polyfit(df['Rating'], df['Metascore'], 1)
predict = np.poly1d(model)

# Calculate R-squared
r2 = r2_score(df['Metascore'], predict(df['Rating']))
print('Rating vs Metascore')
print("R-squared:", r2)

# Scatter plot
plt.scatter(df['Rating'], df['Metascore'], label='Actual data')

# Regression line plot
plt.plot(df['Rating'], predict(df['Rating']), color='red', label='Regression line')

# Labels and title
plt.xlabel('Rating')
plt.ylabel('Metascore')
plt.title('Scatter Plot with Regression Line')

# Show legend
plt.legend()

# Display the plot
plt.show()

# Fit a linear regression model
model = np.polyfit(df['Votes'], df['Revenue_(Millions)'], 1)
predict = np.poly1d(model)

# Calculate R-squared
r2 = r2_score(df['Revenue_(Millions)'], predict(df['Votes']))
print('Votes vs Revenue')
print("R-squared:", r2)

# Scatter plot
plt.scatter(df['Votes'], df['Revenue_(Millions)'], label='Actual data')

# Regression line plot
plt.plot(df['Votes'], predict(df['Votes']), color='red', label='Regression line')

# Labels and title
plt.xlabel('Votes')
plt.ylabel('Revenue_(Millions)')
plt.title('Scatter Plot with Regression Line')

# Show legend
plt.legend()

# Display the plot
plt.show()

