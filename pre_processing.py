# for reading .json.gz data frames


import pandas as pd
import gzip
import json

import matplotlib.pyplot as plt

class PreProcessing:
    def __init__(self, df):
        self.df = df
        self.TRESH_ITEMS = 300
        self.TRESH_USERS = 2

    def calculateSparsity(self, printState = False):
        num_of_ratings = self.df.index.size
        num_of_items = len(pd.unique(self.df['item_id']))
        num_of_users = len(pd.unique(self.df['user_id']))
        num_of_elements = num_of_items * num_of_users

        sparsity = 100 - num_of_ratings/num_of_elements * 100

        if (printState):
            print("Nb. of ratings: {}".format(num_of_ratings))
            print("Nb. of items: {}".format(num_of_items))
            print("Nb. of users: {}".format(num_of_users))
            print("Matrix sparsity: {}%".format(sparsity))

        return sparsity

    def getRidOfUsersWhoAreNotEnoughtActive(self):
        unique_item_id = pd.DataFrame(self.df.groupby('item_id')['rating'].mean())
        unique_item_id['num_of_item_ratings'] = pd.DataFrame(self.df.groupby('item_id')['rating'].count())
        unique_item_id.rename(columns={'rating': 'avg_item_rating'}, inplace=True)


        unique_user_id = pd.DataFrame(self.df.groupby('user_id')['rating'].mean())
        unique_user_id['num_of_user_ratings'] = pd.DataFrame(self.df.groupby('user_id')['rating'].count())
        unique_user_id.rename(columns={'rating': 'avg_user_rating'}, inplace=True)      #renaming to prevent naming issues down the line

        #print(unique_user_id.sort_values('num_of_user_ratings',ascending=False).head(10))

        # Reduce values
        unique_item_id = unique_item_id[unique_item_id['num_of_item_ratings'] >= self.TRESH_ITEMS]
        unique_user_id = unique_user_id[unique_user_id['num_of_user_ratings'] >= self.TRESH_USERS]

        size_before = self.df.size
        self.df = pd.merge(self.df, unique_item_id, on="item_id")       #merging previous data with reduced data

        self.df = pd.merge(self.df, unique_user_id, on="user_id")
        size_now = self.df.size

        print("Reducing size from {} to {}.".format(size_before, size_now))


        return self.df

    def createUserItemMatrix(self, index, columns):
        # Create a user-items matrix from columns
        #print(self.df.size)
        item_matrix = pd.pivot_table(self.df, values="rating", index=index, columns=columns)

        spars_count = item_matrix.isnull().values.sum()
        full_count = item_matrix.size
        print("Sparsity: ", spars_count/full_count * 100, "%")


        return item_matrix








