# Recommender
A recommender system using collaborative filtering (CF) written in python. And compare result between memory-based and low rank matrix factorization approaches.

## Requirements
* Python 3+
* Numpy
* Pickle
* sklearn (Only used to caculat root mean squared error(RMSE))

## Data
The data is from a collection collected by the GroupLens research group.

Book-Crossing [https://grouplens.org/datasets/book-crossing/]

## Preparation
Download the data from GroupLens, uncompress the csv version into `/data` folder and then run the following scripts
```python
from tool import *
dumpDataMat()
```
## Evaluation
```python
import evaluation
evaluation = Evaluation()
evaluation.evalByAccuracy(recommender = rc.ItemBased(simMeasure=cosSim))
evaluation.evalByAccuracy(recommender = rc.ItemBased(simMeasure=euclidSim))
evaluation.evalByAccuracy(recommender = rc.ItemBased(simMeasure=pearsSim))
evaluation.evalByAccuracy(recommender = rc.UserBased(simMeasure=cosSim))
evaluation.evalByAccuracy(recommender = rc.UserBased(simMeasure=euclidSim))
evaluation.evalByAccuracy(recommender = rc.UserBased(simMeasure=pearsSim))
evaluation.evalByAccuracy(recommender = rc.MatrixFactorization(K = 10))
