import csv
import pickle as pk
import gzip
import random
import numpy as np

def dumpDataMat():
    userMat = {}
    bookMat = {}
    books   = {}
    users   = {}

    with open('data/BX-Book-Ratings.csv', 'rt', encoding="iso-8859-1") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';', quotechar='"')
        for row in reader:
            if int(row["Book-Rating"]) == 0: continue
            userMat.setdefault(row["User-ID"], {})
            userMat[row["User-ID"]][row["ISBN"]] = int(row["Book-Rating"])

            bookMat.setdefault(row["ISBN"], {})
            bookMat[row["ISBN"]][row["User-ID"]] = int(row["Book-Rating"])

    with open('data/BX-Books.csv', 'rt', encoding="iso-8859-1") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';', quotechar='"')
        index = 0
        for row in reader:
            if row["ISBN"] in bookMat:
                books[row["ISBN"]] = index
                index += 1

    with open('data/BX-Users.csv', 'rt', encoding="iso-8859-1") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';', quotechar='"')
        index = 0
        for row in reader:
            if row["User-ID"] in userMat:
                users[row["User-ID"]] = index
                index += 1

    with gzip.open('Users.pklz', 'wb') as output:
        pk.dump(users, output)

    with gzip.open('Books.pklz', 'wb') as output:
        pk.dump(books, output)

    with gzip.open('UserMat.pklz', 'wb') as output:
        pk.dump(userMat, output)

    with gzip.open('BookMat.pklz', 'wb') as output:
        pk.dump(bookMat, output)

def loadUsersBooks():
    users = None
    books = None

    with gzip.open('Users.pklz', 'rb') as input:
        users = pk.load(input)

    with gzip.open('Books.pklz', 'rb') as input:
        books = pk.load(input)

    return users, books

def loadUserMat():
    dataMat = None
    with gzip.open('UserMat.pklz', 'rb') as input:
        dataMat = pk.load(input)

    return dataMat

def loadBookMat():
    dataMat = None
    with gzip.open('BookMat.pklz', 'rb') as input:
        dataMat = pk.load(input)

    return dataMat

def loadUserTestMat():
    dataMat = loadUserMat();
    testMat = dict(random.sample(dataMat.items(), 1000))
    for user, ratings in testMat.items():
        ratings = ratings.fromkeys(ratings, 0)
        testMat[user] = ratings

    return testMat

def loadBookTestMat():
    dataMat = loadBookMat();
    testMat = dict(random.sample(dataMat.items(), 100))
    for book, ratings in testMat.items():
        ratings = ratings.fromkeys(ratings, 0)
        testMat[book] = ratings

    return testMat
