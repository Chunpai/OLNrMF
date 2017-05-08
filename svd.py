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
import numpy.linalg as LA
import pickle




def nrmf_test():
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
    r = 15
    
    F,G,R = nrmf.AltQP_Inc(injected_M,n,r,l)
    plotResidual(R,anomaly,n,r,l) 
    pickle.dump(RBM,open('synthetic/synth1.pkl','wb'))
    pickle.dump(anomaly,open('synthetic/anomaly1.pkl','wb'))
     

def plotResidual(R,anomaly,n,r,l):
    plt.axis([0,l+1,0,n+1])
    count = 0
    acount = 0
    #outfile = open(data_dir+'residuals2/R'+str(r)+'.csv','w')
    #outfile2 = open(data_dir+'result2.csv','a')
    for source in R:
        source_list = []
        target_list = []
        anomaly_source_list = []
        anomaly_target_list = []
        for target in R[source]:
            if R[source][target] > 0.00001:
                count += 1
                source_list.append(source)
                target_list.append(target)
                #outfile.write(str(source)+';'+str(target)+'\n')
                if source in anomaly:
                    if target in anomaly[source]:
                        acount += 1
                        print acount
                        anomaly_source_list.append(source)
                        anomaly_target_list.append(target)
        #plt.plot(source_list, target_list, 'b.', markersize=0.5)
        plt.plot(target_list, source_list, 'b.')
        #plt.plot(anomaly_source_list, anomaly_target_list, 'r.', markersize=0.5)
        plt.plot(anomaly_target_list, anomaly_source_list, 'r.')
    #outfile.close()
    plt.savefig('nrmf_vs_svd/nrmf.png')
    
    #outfile2.write(str(r)+';'+str(count)+';'+str(acount)+'\n')
    #outfile2.close()
    print r
    print count
    print acount



def svd_test():
    RBM = pickle.load(open('synthetic/synth1.pkl','rb'))
    anomaly = pickle.load(open('synthetic/anomaly1.pkl','rb'))
     
    r = 70 

    #P = numpy.random.rand(n,r)
    #Q = numpy.random.rand(l,r)

    U,s,V = LA.svd(RBM) 
    print U.shape
    print s.shape
    print V.shape

    print U[:,:r].shape
    print numpy.diag(s[:r]).shape
    print V[:r,:].shape
    print U[:,:r].dot(numpy.diag(s[:r])).shape
    print U[:,:r].dot(numpy.diag(s[:r])).dot(V[:r,:]).shape
    aR = U[:,:r].dot(numpy.diag(s[:r])).dot(V[:r,:])

    #nP, nQ = matrix_factorization(RBM, P, Q, r)
    #aR = numpy.dot(nP,nQ.T)
    residual = RBM - aR
    for row in range(len(residual)):
        source_list = []
        target_list = []
        anomaly_source_list = []
        anomaly_target_list = []
        for col in range(len(residual[row])):
            if abs(residual[row][col]) > 0.05 and RBM[row][col] != 0.0:
                source_list.append(row)
                target_list.append(col)
                if row in anomaly:
                    if col in anomaly[row]:
                        anomaly_source_list.append(row)
                        anomaly_target_list.append(col)
        plt.plot(target_list, source_list, 'b.')
        plt.plot(anomaly_target_list, anomaly_source_list, 'r.')
    plt.savefig('nrmf_vs_svd/svd.png')


if __name__ == "__main__":
    #nrmf_test()
    svd_test()
