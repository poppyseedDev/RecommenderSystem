# for reading .json.gz data frames


import pandas as pd
import gzip
import json

import matplotlib.pyplot as plt

class PreProcessing:
    def __init__(self, df):
        self.df = df

    def calculateSparsity(self):
        


