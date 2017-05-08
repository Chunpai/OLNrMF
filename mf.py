import numpy 
from networkx.algorithms import bipartite
import math
import datetime
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import data
import networkx as nx
from networkx.algorithms import bipartite
import plot
import test1
import nrmf

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

###############################################################################



if __name__ == "__main__":
    n = 100
    l = 80
    #RB = bipartite.random_graph(n,l,0.2)
    #X, Y = bipartite.sets(RB)
    
    #print len(X)
    #print len(Y)
    #RB = bipartite.biadjacency_matrix(RB,X)
    #RBM = RB.toarray() 
    
    M, anomaly, injected_M, RBM = test1.synth1(n,l)

    n = len(RBM)
    l = len(RBM[0])
    r = 10
    
    F,G,R = nrmf.AltQP_Inc(injected_M,n,r,l)
    test1.plotResidual(R,anomaly,n,r,l) 


    r = 2

    P = numpy.random.rand(n,r)
    Q = numpy.random.rand(l,r)

    nP, nQ = matrix_factorization(RBM, P, Q, r)
    aR = numpy.dot(nP,nQ.T)
    residual = RBM - aR
    for row in range(len(residual)):
        source_list = []
        target_list = []
        anomaly_source_list = []
        anomaly_target_list = []
        for col in range(len(residual[row])):
            if abs(residual[row][col]) > 0.0001:
                source_list.append(row)
                target_list.append(col)
                if row in anomaly:
                    if col in anomaly[row]:
                        anomaly_source_list.append(row)
                        anomaly_target_list.append(col)
        plt.plot(target_list, source_list, 'b.')
        plt.plot(anomaly_target_list, anomaly_source_list, 'r.')
    plt.savefig('mf.png')
