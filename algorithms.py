


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