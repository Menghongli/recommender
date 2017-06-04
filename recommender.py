import numpy as np
import pickle as pk
import tool
import similarity
import gzip
import sys

class Recommender:
    def __init__(self, dataMat, simMeasure):
        self.dataMat = dataMat
        self.simMeasure = simMeasure

    def predict(self, user, book, true_value):
        return None

class ItemBased(Recommender):
    def __init__(self, simMeasure):
        Recommender.__init__(self, dataMat=tool.loadBookMat(), simMeasure=simMeasure)

    def predict(self, user, book, true_value):
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

    def predict(self, user, book, true_value):
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
        Recommender.__init__(self, dataMat=tool.loadUserMat(), simMeasure=None)
        self.K = K
        self.users, self.books = tool.loadUsersBooks()
        self.X, self.Q = self.train()


    def train(self, steps=20, alpha=0.008, beta=0.02):
        n_users = len(self.users.keys())
        n_books = len(self.books.keys())

        # R = sp.csc_matrix(self.dataMat)
        # U, s, Vt = la.svds(R)
        # SigDim = np.mat(np.eye(6)*s)
        # transformedUser = (R.T * U * SigDim.I).T

        X_final = np.random.rand(n_users, self.K)
        Q_final = np.random.rand(self.K, n_books)

        cost = 0

        for step in range(steps):
            X = X_final
            Q = Q_final
            count = 0
            flag = False
            for user, ratings in self.dataMat.items():
                sys.stdout.write('\r')
                sys.stdout.write("[%-80s] %d/%d" % ('=' * int((count/n_users) * 80), count, n_users-1))
                sys.stdout.flush()
                count += 1
                for book, rate in ratings.items():
                    if user in self.users.keys() and book in self.books.keys():
                        i = self.users[user]
                        j = self.books[book]
                        er = np.dot(X[i,:], Q[:, j]) - rate
                        for k in range(self.K):
                            temp_x = X[i][k] - alpha * (er * Q[k][j] + beta * X[i][k])
                            temp_q = Q[k][j] - alpha * (er * X[i][k] + beta * Q[k][j])

                            X[i][k] = temp_x
                            Q[k][j] = temp_q
            print()

            cost_hat = 0
            for user, ratings in self.dataMat.items():
                for book, rate in ratings.items():
                    cost_hat += (rate - np.dot(X[i,:],Q[:,j]))**2
                    for k in range(self.K):
                        cost_hat += beta * (X[i][k]**2 + Q[k][j]**2)

                    cost_hat = cost_hat/2

            print("Step: %d, Error: %f" % (step, cost_hat))
            if cost_hat < 0.1:
                X_final = X
                Q_final = Q
                break

            if cost_hat >= cost and cost != 0:
                if flag: break
                else: flag = True
            else:
                flag = False
                X_final = X
                Q_final = Q

            cost = cost_hat

        return X_final, Q_final

    def predict(self, user, book, true_value):
        if user in self.users.keys() and book in self.books.keys():
            u_index = self.users[user]
            b_index = self.books[book]

            return np.dot(self.X[u_index,:], self.Q[:, b_index])
        else:
            return true_value
