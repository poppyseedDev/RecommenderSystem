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

    def number_of_ratings_histogram(self):
        # histogram of ratings
        plt.figure(figsize=(10, 4))
        self.df['rating'].hist(bins=70)
        plt.xlabel("ratings")
        plt.ylabel("nb. of ratings")
        plt.title("Amazon fashion histogram")
        plt.show()

