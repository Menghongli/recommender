import tool
import recommender as rc
from similarity import *
from sklearn.metrics import mean_squared_error
from math import sqrt

class Evaluation:
    def __init__(self, recommender):
        self.recommender = recommender
        self.dataMat = tool.loadUserMat()
        self.testMat = tool.loadUserTestMat()

    def evalByAccuracy(self):
        y_true = []
        y_pred = []

        for user, ratings in self.testMat.items():
            for book, rating in ratings.items():
                y_true.append(float(self.dataMat[user][book]))
                y_pred.append(self.recommender.predict(user, book))

        rmse = sqrt(mean_squared_error(y_true, y_pred))
        print("Root Mean Squared Error: %f" % rmse)

evaluation = Evaluation(recommender = rc.ItemBased(simMeasure=cosSim))
evaluation.evalByAccuracy()

evaluation1 = Evaluation(recommender = rc.ItemBased(simMeasure=euclidSim))
evaluation1.evalByAccuracy()
