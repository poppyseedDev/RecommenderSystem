import pandas as pd
import random

import pre_processing
import visualization
import algorithms

import read_data_frame

from sklearn.metrics.pairwise import cosine_similarity

datasetPropreties = {
    "filename": "C:\\Users\\auror\\Desktop\\koda\\sola\\podatkovne vede\\data\\AMAZON_FASHION.csv",
    "columnNames": ['item_id', 'user_id', 'rating', 'timestamp']
}

def TestCorrelation(item_user_matrix, user):
    algo = algorithms.Algorithms()
    algo.setMatrix(item_user_matrix)

    # get user items
    all_user_items = item_user_matrix.iloc[item_user_matrix.index.get_loc(user),:]
    unique_user_items = all_user_items.index[all_user_items.notnull()]

    all_recommended_items = []
    # go over each item the user has rated
    for item in unique_user_items:
        results = algo.calculateCorrelation(item)

        # add only those results that are unique
        for result in results:
            if result not in all_recommended_items:
                all_recommended_items.append(result)

    # remove items the user has already seen
    for already_seen_item in unique_user_items:
        if already_seen_item in all_recommended_items:
            all_recommended_items.remove(already_seen_item)

    return all_recommended_items


def CenteredCosineSimilarity(item_user_matrix, user):
    #initate class
    algo = algorithms.Algorithms()
    algo.setMatrix(item_user_matrix)

    # get user items
    unique_user_items = item_user_matrix[user][item_user_matrix[user].notnull()]
    unique_user_items = unique_user_items.index

    all_recommended_items = []
    for item in unique_user_items:
        results = algo.calculateCosineSimilarity(item)
        # add only those results that are unique
        for result in results[0]:
            if result not in all_recommended_items:
                all_recommended_items.append(result)

    # remove items the user has already seen
    for already_seen_item in unique_user_items:
        if already_seen_item in all_recommended_items:
            all_recommended_items.remove(already_seen_item)

    return all_recommended_items



def Main():
    #----------------------- LOAD DATA --------------------------------------
    load_data = read_data_frame.LoadDataFrames(datasetPropreties)   #initialize class to load data

    # Load the data
    df = load_data.read_user_ratings_data()
    # -----------------------------------------------------------------------

    # ----------------- INITIALIZE CLASSES ----------------------------------
    # Initialize classes
    visualize = visualization.VisualizationOfData(df)               #initialize class to visulaize data
    preProcessed = pre_processing.PreProcessing(df)                 #initialize class to pre-process data
    # -----------------------------------------------------------------------

    # ----------------- REDUCE DATASET --------------------------------------
    # Reduce the matrix for given item and user thresholds
    reduced_df = preProcessed.getRidOfUsersWhoAreNotEnoughtActive()
    #preProcessed.calculateSparsity(printState=True)

    # ---------------- VISUALIZE DATA ---------------------------------------
    #visualize.set_df(reduced_df)
    #visualize = visualization.VisualizationOfData(df)               #initialize class to visulaize data
    #visualize.unique_itemID_histogram()
    # -----------------------------------------------------------------------


    #print(reduced_df.sort_values('num_of_item_ratings', ascending=False))

    # ------------- TRYING OUT DIFFERENT ALGORITHMS -------------------------
    testCORR = False
    testCOSINE = True

    TEST_USER = 'A30WK4BBLSXER7'

    all_users = pd.unique(reduced_df['user_id'])
    #randomly shuffle users
    random.shuffle(all_users)
    #all_users = ['A1C365S9OS5HFY', 'A30WK4BBLSXER7', 'AP6SE0LUVSOOF']

    number_of_recomendations_Cos = []
    number_of_recomendations_Corr = []


    if (testCOSINE):
        # ------------- TRYING OUT DIFFERENT ALGORITHMS -------------------------
        # Create a user-items matrix from columns
        item_user_matrix = preProcessed.createUserItemMatrix(index='item_id', columns='user_id')
        # -----------------------------------------------------------------------
        for idx, user in enumerate(all_users):
            recomendedCos = CenteredCosineSimilarity(item_user_matrix, user)
            if (idx % 10 == 0):
                print(idx)
            if (idx == 10000):
                break
            print(recomendedCos)
            number_of_recomendations_Cos.append(len(recomendedCos))


    if (testCORR):
        # ------------- TRYING OUT DIFFERENT ALGORITHMS -------------------------
        # Create a user-items matrix from columns
        item_user_matrix = preProcessed.createUserItemMatrix(index='user_id', columns='item_id')
        # -----------------------------------------------------------------------
        for idx, user in enumerate(all_users):
            recomendedCorr = TestCorrelation(item_user_matrix, user)
            if (idx % 100 == 0):
                print(idx)
            if (idx == 10000):
                break
            print(recomendedCorr)
            number_of_recomendations_Corr.append(len(recomendedCorr))
    # ---------------------------------------------------------------------------

    print(number_of_recomendations_Corr)
    print(number_of_recomendations_Cos)
    visualize.plotResultsHistogram(number_of_recomendations_Cos, number_of_recomendations_Corr)


if (__name__ == '__main__'):
    Main()
