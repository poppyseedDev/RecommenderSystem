# Uvozi za vizualizacijo
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_style('white')
#%matplotlib inline
import numpy as np
import pandas as pd


class VisualizationOfData:
    def __init__(self, df):
        self.df = df

    def all_ratings_histogram(self):
        # histogram of ratings
        plt.figure(figsize=(10, 4))
        self.df['rating'].hist(bins=70)
        plt.xlabel("ratings")
        plt.ylabel("nb. of ratings")
        plt.title("Amazon fashion histogram")
        plt.show()

    def unique_itemID_histogram(self):
        # group by item_id
        unique_item_id = pd.DataFrame(self.df.groupby('item_id')['rating'].mean())
        unique_item_id['num of ratings'] = pd.DataFrame(self.df.groupby('item_id')['rating'].count())

        fig, (ax1, ax2) = plt.subplots(2)
        ax1.hist(unique_item_id['num of ratings'], bins=70)

        ax2.hist(unique_item_id['rating'], bins=70)

        sns.jointplot(x='rating', y='num of ratings', data=unique_item_id, alpha=0.5)

        plt.show()
