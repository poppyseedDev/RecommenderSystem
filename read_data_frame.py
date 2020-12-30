# for reading .json.gz data frames


import pandas as pd
import gzip
import json

import matplotlib.pyplot as plt

class LoadDataFrames:
    def __init__(self, dataProp):
        self.file_name = dataProp['filename']
        self.column_names = dataProp['columnNames']


    def read_user_ratings_data(self):
        # reads user ratings data needs the name of the file as input
        df = pd.read_csv(self.file_name, sep=',', names=self.column_names)
        return pd.DataFrame(df)
