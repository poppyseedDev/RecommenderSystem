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
            item_user_matrix_random_item = self.matrix[self.matrix[item].notnull()]
            item_user_matrix_random_item = item_user_matrix_random_item.dropna(axis='columns', thresh=2)

            random_item_ratings = self.matrix[item]
            if (item_user_matrix_random_item.any):
                similar_to_random_item = item_user_matrix_random_item.corrwith(random_item_ratings)
                if (printState):
                    print(similar_to_random_item)

                list_of_similar_items = similar_to_random_item.index[similar_to_random_item > 0.3]

                return list(list_of_similar_items)

            else:
                print('No matching items to be found.')
                return 0

        else:
            print("Call this class with setMatrix method to set a matrix")
            return 0

    def calculateCosineSimilarity(self, item):
        if (self.matrix.any):
            indexOfItem = self.matrix.index.get_loc(item)

            mean_items = self.matrix.mean(axis='index')
            # item_user_matrix.mean(axis='index').hist(bins=70)
            # plt.show()

            item_user_matrix = self.matrix - mean_items

            item_user_matrix.fillna(value=0, inplace=True)

            # find similar users
            item_user_matrix = item_user_matrix.to_numpy()
            similar_users = cosine_similarity([item_user_matrix[indexOfItem][:]], item_user_matrix)

            locations_of_similar_items = np.where(similar_users[0] > 0.3)

            # get names of the array
            loc_names_sim_users = []
            for location in locations_of_similar_items:
                loc_names_sim_users.append(list(self.matrix.index[location]))

            return list(loc_names_sim_users)
        else:
            print("Call this class with setMatrix method to set a matrix")
            return 0