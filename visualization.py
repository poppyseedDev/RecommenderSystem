# Uvozi za vizualizacijo
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_style('white')
#%matplotlib inline
import numpy as np
import pandas as pd


class VisualizationOfData:
    def __init__(self, df):
        self._df = df

    def set_df(self, df):
        self._df = df

    def all_ratings_histogram(self):
        # histogram of ratings
        plt.figure(figsize=(10, 4))
        self._df['rating'].hist(bins=70)
        plt.xlabel("ratings")
        plt.ylabel("nb. of ratings")
        plt.title("Amazon fashion histogram")
        plt.show()

    def unique_itemID_histogram(self):
        # group by item_id
        unique_item_id = pd.DataFrame(self._df.groupby('item_id')['rating'].mean())
        unique_item_id['num of ratings'] = pd.DataFrame(self._df.groupby('item_id')['rating'].count())

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Histogram of Amazon Fashion data category grouped by item id")
        ax1.hist(unique_item_id['num of ratings'], bins=70)
        ax1.set_xlabel("number of ratings per item")
        ax1.set_ylabel("amounts of items")

        ax2.hist(unique_item_id['rating'], bins=70)
        ax2.set_xlabel("average rating of item")
        ax2.set_ylabel("amounts of items")

        p = sns.jointplot(x='rating', y='num of ratings', data=unique_item_id, alpha=0.5)
        p.fig.suptitle("Scatter plot of average rating vs rating per item on Amazon Fashion data")


        plt.show()

    def unique_userID_histogram(self):
        #group by user_id
        unique_user_id = pd.DataFrame(self._df.groupby('user_id')['rating'].mean())
        unique_user_id['num of ratings'] = pd.DataFrame(self._df.groupby('user_id')['rating'].count())

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Histogram of Amazon Fashion data category grouped by user id")
        ax1.hist(unique_user_id['num of ratings'], bins=70)
        ax1.set_xlabel("number of ratings per user")
        ax1.set_ylabel("amounts of users")

        ax2.hist(unique_user_id['rating'], bins=70)
        ax2.set_xlabel("users average rating")
        ax2.set_ylabel("amounts of users")

        p = sns.jointplot(x='rating', y='num of ratings', data=unique_user_id, alpha=0.5)
        p.fig.suptitle("Scatter plot of average rating vs rating per user on Amazon Fashion data")

        plt.show()
