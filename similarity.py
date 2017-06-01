from numpy import *
from numpy import linalg as la

def euclidSim(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA, inB):
    if len(inA) < 3 : return 1.0
    cor = corrcoef(inA, inB, rowvar = 0)[0][1]
    if isnan(cor): cor = 0
    return 0.5 + 0.5*cor

def cosSim(inA, inB):
    num = float(dot(inA, inB))
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)
