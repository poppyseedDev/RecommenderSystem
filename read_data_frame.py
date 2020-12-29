# for reading .json.gz data frames


import pandas as pd
import gzip
import json

import matplotlib.pyplot as plt

class LoadDataFrames:
    def __init__(self, file_name):
        self.file_name = file_name

    def read_user_ratings_data(self):
        # reads user ratings data needs the name of the file as input
        column_names = ['item_id', 'user_id', 'rating', 'timestamp']
        df = pd.read_csv(self.file_name, sep=',', names=column_names)
        return df
