# for reading .json.gz data frames


import pandas as pd
import gzip
import json

import matplotlib.pyplot as plt

class PreProcessing:
    def __init__(self, df):
        self.df = df
        self.TRESH_ITEMS = 50
        self.TRESH_USERS = 2

    def calculateSparsity(self, printState = False):
        num_of_ratings = self.df.size
        num_of_items = len(pd.unique(self.df['item_id']))
        num_of_users = len(pd.unique(self.df['user_id']))
        num_of_elements = num_of_items * num_of_users

        sparsity = num_of_ratings/num_of_elements * 100

        if (printState):
            print("Nb. of ratings: {}".format(num_of_ratings))
            print("Nb. of items: {}".format(num_of_items))
            print("Nb. of users: {}".format(num_of_users))
            print("Matrix sparsity: {}%".format(sparsity))

        return sparsity

    def getRidOfUsersWhoAreNotEnoughtActive(self):
        unique_item_id = pd.DataFrame(self.df.groupby('item_id')['rating'].mean())
        unique_item_id['num_of_item_ratings'] = pd.DataFrame(self.df.groupby('item_id')['rating'].count())

        unique_user_id = pd.DataFrame(self.df.groupby('user_id')['rating'].mean())
        unique_user_id['num_of_user_ratings'] = pd.DataFrame(self.df.groupby('user_id')['rating'].count())

        # Reduce values
        unique_item_id = unique_item_id[unique_item_id['num_of_item_ratings'] >= self.TRESH_ITEMS]
        unique_user_id = unique_user_id[unique_user_id['num_of_user_ratings'] >= self.TRESH_USERS]

        print(self.df.size)
        self.df = pd.merge(self.df, unique_item_id, on="item_id")
        print(self.df.size)
        self.df = pd.merge(self.df, unique_user_id, on="user_id")
        print(self.df.size)

        return self.df







