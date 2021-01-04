import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



class Algorithms:
    def __init__(self, matrix=None):
        self.matrix = matrix

    def setMatrix(self, matrix):
        self.matrix = matrix

    def calculateCorrelation(self, item, printState=False):
        if (self.matrix.any):
            # Correlation - find similaries to item
            # you have to remove users who have not rated this item to find out which items match
            user_item_matrix_random_item = self.matrix[self.matrix[item].notnull()]
            user_item_matrix_random_item = user_item_matrix_random_item.dropna(axis='columns', thresh=2)

            random_item_ratings = user_item_matrix_random_item[item]

            similar_to_random_item = user_item_matrix_random_item.corrwith(random_item_ratings)

            if (printState):
                print(similar_to_random_item.sort_values(ascending=False).head(100))

            return similar_to_random_item
        else:
            print("Call this class with setMatrix to set a matrix")
            return 0

    def calculateCosineSimilarity(self):
        mean_users = self.matrix.mean(axis='index')
        # user_item_matrix.mean(axis='index').hist(bins=70)
        # plt.show()

        user_item_matrix = self.matrix - mean_users

        user_item_matrix.fillna(value=0, inplace=True)

        # find similar users
        user_item_matrix = user_item_matrix.to_numpy()
        similar_users = cosine_similarity([user_item_matrix[0][:]], user_item_matrix)

        locations_of_similar_users = np.where(similar_users[0] > 0.3)

        # get names of the array
        loc_names_sim_users = []
        for location in locations_of_similar_users:
            loc_names_sim_users.append(self.matrix.index[location])

        return loc_names_sim_users
