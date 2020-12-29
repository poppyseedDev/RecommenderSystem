import numpy as np
import pandas as pd

import visualization

import read_data_frame



if (__name__ == '__main__'):
    load_data = read_data_frame.LoadDataFrames("C:\\Users\\auror\\Desktop\\koda\\sola\\podatkovne vede\\data\\AMAZON_FASHION.csv")
    df = load_data.read_user_ratings_data()
    print(df)
    visualize = visualization.VisualizationOfData(df)    # initialize the class
    visualize.number_of_ratings_histogram()
