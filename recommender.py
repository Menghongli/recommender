from scipy import linalg as la

import numpy as np
import pickle as pk
import tool
import similarity
import gzip

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


class MatrixFactorization(Recommender):
    def __init__(self, K):
        Recommender.__init__(self, dataMat=None, simMeasure=None)
        self.K = K
        self.users, self.books, self.dataMat = tool.loadRating()
        self.X, self.Q = self.train()


    def train(self, steps=5000, alpha=0.0002, beta=0.02):
        n_users = len(self.dataMat)
        n_books = len(self.dataMat[0])

        U,Sigma, VT = la.svds(self.dataMat, return_singular_vectors='u')
        # Sig2 = Sigma**2
        # totalEng = sum(Sig2) * 0.9
        # dim = 0

        # for u in range(n_users):
        #     if sum(Sig2[:u]) >= totalEng:
        #         dim = u
        #         break
        SigDim = np.mat(np.eye(6)*Sigma[:6])

        transformedUser = (self.dataMat.T * U[:, :6] * SigDim.I).T

        X = np.random.rand(n_users, self.K)
        Q = np.random.rand(n_books, self.K)
        Q = Q.T

        for step in range(steps):
            print("Step: {}".format(step))
            for i in range(dim):
                print('.', end='', flush=True)
                for j in range(n_books):
                    if self.dataMat[i][j] > 0:
                        er = np.dot(X[i,:], Q[:, j]) - self.dataMat[i][j]
                        for k in range(self.K):
                            temp_x = X[i][k] - alpha * (er * Q[k][j] + beta * X[i][k])
                            temp_q = Q[k][j] - alpha * (er * X[i][k] + beta * Q[k][j])

                            X[i][k] = temp_x
                            Q[k][j] = temp_q

            cost = 0
            for i in range(n_users):
                for j in range(n_books):
                    if self.dataMat[i][j] > 0:
                        cost = cost + (self.dataMat[i][j] - np.dot(X[i,:],Q[:,j]))**2
                        for k in range(self.K):
                            cost = cost + beta * (X[i][k]**2 + Q[k][j]**2)

                        cost = cost/2
            if cost < 0.1:
                break

        return X, Q.T

    def predict(self, user, book):
        R = np.dot(self.X, self.Q.T)
        u_index = self.users[user]
        b_index = self.books[book]

        return R[u_index][b_index]
