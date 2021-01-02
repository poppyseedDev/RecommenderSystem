import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pre_processing
import visualization
import algorithms

import read_data_frame

datasetPropreties = {
    "filename": "C:\\Users\\auror\\Desktop\\koda\\sola\\podatkovne vede\\data\\AMAZON_FASHION.csv",
    "columnNames": ['item_id', 'user_id', 'rating', 'timestamp']
}

def testCorrelation(df):
    # Initialize classes
    visualize = visualization.VisualizationOfData(df)               #initialize class to visulaize data
    preProcessed = pre_processing.PreProcessing(df)                 #initialize class to pre-process data

    # Reduce the matrix for given item and user thresholds
    reduced_df = preProcessed.getRidOfUsersWhoAreNotEnoughtActive()
    #preProcessed.calculateSparsity(printState=True)
    visualize.set_df(reduced_df)

    # Create a user-items matrix from columns
    user_item_matrix = preProcessed.createUserItemMatrix(index='user_id', columns='item_id')
    item_user_matrix = preProcessed.createUserItemMatrix(index='item_id', columns='user_id')

    algo = algorithms.Algorithms()
    algo.setMatrix(user_item_matrix)
    algo.calculateCorrelation('B01FQ114LG', printState=True)

    algo.setMatrix(item_user_matrix)
    algo.calculateCorrelation('A2PBHVTPTIIGKR', printState=True)


def Main():
    load_data = read_data_frame.LoadDataFrames(datasetPropreties)   #initialize class to load data

    # Load the data
    df = load_data.read_user_ratings_data()

    








if (__name__ == '__main__'):
    Main()
