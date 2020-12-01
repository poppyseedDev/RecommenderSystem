
import numpy as np
import pandas as pd


def read_user_ratings_data(file_name):
    # reads user ratings data needs the name of the file as input
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(file_name, sep='\t', names=column_names)
    return df


def read_names_data(df, file_name):
    # reads movie titles
    movie_titles = pd.read_csv(file_name)
    movie_titles.head()

    df = pd.merge(df,movie_titles,on='item_id')
    return df



if (__name__ == '__main__'):
    df = read_user_ratings_data('u.data')
    df = read_names_data(df, "Movie_Id_Titles")
    print(df.head)
