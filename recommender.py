import numpy as np
import pickle as pk
import tool
import similarity

class Recommender:
    def __init__(self, dataMat, simMeasure):
        self.dataMat = dataMat
        self.simMeasure = simMeasure

    def predict(self, user, book):
        return None

class ItemBased(Recommender):
    def __init__(self, simMeasure):
        Recommender.__init__(self, dataMat=tool.loadBookMat(), simMeasure=simMeasure)

    def predict(self, user, book):
        simTotal= 0.0; ratSimTotal = 0.0
        bookRatings = self.dataMat[book]
        for otherBook, ratings in self.dataMat.items():
            if (user not in ratings) or otherBook == book: continue
            userRating = ratings[user]
            overlapUsers = ratings.keys() & bookRatings.keys()
            if len(overlapUsers) == 0:
                similarity = 0
            else:
                validRatings = { book: [] }
                validRatings[otherBook] = []

                for u in overlapUsers:
                    validRatings[otherBook].append(ratings[u])
                    validRatings[book].append(self.dataMat[book][u])

                similarity = self.simMeasure(np.array(validRatings[book]), np.array(validRatings[otherBook]))

            simTotal += similarity
            ratSimTotal += similarity * userRating
        if simTotal == 0: return 0
        else: return ratSimTotal/simTotal

class UserBased(Recommender):
    def __init__(self, simMeasure):
        Recommender.__init__(self, dataMat=tool.loadUserMat(), simMeasure=simMeasure)

    def predict(self, user, book):
        simTotal= 0.0; ratSimTotal = 0.0
        userRatings = self.dataMat[user]
        for otherUser, ratings in self.dataMat.items():
            if (book not in ratings) or otherUser == user: continue
            userRating = ratings[book]
            overlapBooks = ratings.keys() & userRatings.keys()
            if len(overlapBooks) == 0:
                similarity = 0
            else:
                validRatings = { user: [] }
                validRatings[otherUser] = []

                for b in overlapBooks:
                    validRatings[otherUser].append(ratings[b])
                    validRatings[user].append(self.dataMat[user][b])

                similarity = self.simMeasure(np.array(validRatings[user]), np.array(validRatings[otherUser]))

            simTotal += similarity
            ratSimTotal += similarity * userRating
        if simTotal == 0: return 0
        else: return ratSimTotal/simTotal
