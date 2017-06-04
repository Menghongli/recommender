import tool
import recommender as rc
from similarity import *
from sklearn.metrics import mean_squared_error
from math import sqrt

class Evaluation:
    def __init__(self):
        self.dataMat = tool.loadUserMat()
        self.testMat = tool.loadUserTestMat()

    def evalByAccuracy(self, recommender):
        y_true = []
        y_pred = []

        for user, ratings in self.testMat.items():
            for book, rating in ratings.items():
                y_true.append(float(self.dataMat[user][book]))
                y_pred.append(recommender.predict(user, book, self.dataMat[user][book]))

        rmse = sqrt(mean_squared_error(y_true, y_pred))
        print("Root Mean Squared Error: %f" % rmse)

evaluation = Evaluation()
# evaluation.evalByAccuracy(recommender = rc.ItemBased(simMeasure=cosSim))
# evaluation.evalByAccuracy(recommender = rc.ItemBased(simMeasure=euclidSim))
# evaluation.evalByAccuracy(recommender = rc.ItemBased(simMeasure=pearsSim))
# evaluation.evalByAccuracy(recommender = rc.UserBased(simMeasure=cosSim))
# evaluation.evalByAccuracy(recommender = rc.UserBased(simMeasure=euclidSim))
# evaluation.evalByAccuracy(recommender = rc.UserBased(simMeasure=pearsSim))
evaluation.evalByAccuracy(recommender = rc.MatrixFactorization(K = 10))
