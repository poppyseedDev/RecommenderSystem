import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import pre_processing
import visualization

import read_data_frame

datasetPropreties = {
    "filename": "C:\\Users\\auror\\Desktop\\koda\\sola\\podatkovne vede\\data\\AMAZON_FASHION.csv",
    "columnNames": ['item_id', 'user_id', 'rating', 'timestamp']
}

def Main():
    load_data = read_data_frame.LoadDataFrames(datasetPropreties)   #initialize class to load data

    #load the data
    df = load_data.read_user_ratings_data()

    visualize = visualization.VisualizationOfData(df)               #initialize class to visulaize data
    preProcessed = pre_processing.PreProcessing(df)                 #initialize class to pre-process data
    visualize.unique_itemID_histogram()     # display the histograms with unique titles


    #make a user-items matrix from columns
    #print(df.head())
    #item_matrix = pd.pivot_table(df, values="rating", index="user_id", columns="item_id")

    #print(item_matrix.head())

    #preProcessed.calculateSparsity(printState = True)


    # Reduce the matrix for only
    reduced_df = preProcessed.getRidOfUsersWhoAreNotEnoughtActive()
    preProcessed.calculateSparsity(printState=True)
    visualize.set_df(reduced_df)
    visualize.unique_itemID_histogram()     # display the histograms with unique titles



if (__name__ == '__main__'):
    Main()
