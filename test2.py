'''
simulate weighted dataset and applied with AltQP_Inc
'''

import numpy as np
import numpy.linalg as LA
import math
import datetime
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import data
import networkx as nx
from networkx.algorithms import bipartite
import plot
import nrmf


def synth2(n,l):
    '''
    sythesize random bipartite graph and anomalies injected
    '''
    RB = bipartite.random_graph(n,l,0.2)
    X, Y = bipartite.sets(RB)
    print 'Y',Y
    print 'degrees',bipartite.degrees(RB,X.union(Y))
    
    #pos = dict()
    #pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
    #pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
    #nx.draw(RB, pos=pos)
    #plt.show()
    print len(X)
    print len(Y)
    RB = bipartite.biadjacency_matrix(RB,X)

    RBM = RB.toarray() 
    print len(RBM)
    print len(RBM[0])
    #M = nx.to_dict_of_dicts(RB)
    
    #print M
    plt.axis([-1,l+1,-1,n+1])
    plt.xlabel('movie')
    plt.ylabel('user')
    count = 0
    source_list = []
    target_list = []
    M = {}
    acount = 0
    for row in range(len(RBM)):
        source_list = []
        target_list = []
        for col in range(len(RBM[row])):
            if RBM[row][col] == 1:
                source_list.append(row)
                target_list.append(col)
                rating = np.random.normal(3,1)
                while rating > 4.0 or rating < 2.0:
                    rating = np.random.normal(3,1)
                if row not in M:
                    M[row] = {}
                    M[row][col] = rating 
                else:
                    M[row][col] = rating 
        plt.plot(target_list, source_list, 'b.')
    
    injected_M = deepcopy(M)
    anomaly = {}
    for i in range(1):
        anomaly_user = random.choice(range(n))
        print 'anomaly_user',anomaly_user
        target_size = l
        for j in range(l):
            anomaly_movie = random.choice(range(l))
            #if anomaly_movie not in injected_M[anomaly_user]:
            #    count += 1
            rating = random.choice([1.0,5.0])
            if anomaly_user not in injected_M:
                injected_M[anomaly_user] = {}
                injected_M[anomaly_user][anomaly_movie] = rating
            else:
                injected_M[anomaly_user][anomaly_movie] = rating

            if anomaly_user not in anomaly:
                acount += 1
                anomaly[anomaly_user] = {}
                anomaly[anomaly_user][anomaly_movie] = rating
            else:
                if anomaly_movie not in anomaly[anomaly_user]:
                    acount += 1
                anomaly[anomaly_user][anomaly_movie] = rating
    
    print 'acount',acount
    for source in anomaly:
        source_list = []
        target_list = []
        for target in anomaly[source]:
            source_list.append(source)
            target_list.append(target)
        #plt.plot(source_list, target_list, 'r.', markersize=1.0)
        plt.plot(target_list, source_list, 'r.')
    print 'source_list',len(source_list)
    plt.show()
    return M, anomaly, injected_M


def plotResidual(R,anomaly,n,r,l):
    plt.axis([-1,l+1,-1,n+1])
    plt.xlabel('movie')
    plt.ylabel('user')
    count = 0
    acount = 0
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
    #plt.savefig(data_dir+'residuals2/R'+str(r)+'.png')
    plt.show()
    #outfile2.write(str(r)+';'+str(count)+';'+str(acount)+'\n')
    #outfile2.close()
    print r
    print count
    print acount
    print R
   

def test2():
    n = 100
    l = 80
    r = 10
    M, anomaly, injected_M = synth2(n,l) 
    F,G,R = nrmf.AltQP_Inc(injected_M,n,r,l)
    plotResidual(R,anomaly,n,r,l) 
    for source in R:
        for target in R[source]:
            if R[source][target] > 0.00001:
                print 'source',source
                print 'target',target
                print 'original', injected_M[source][target]
                print 'residual', R[source][target]
                print '\n'


if __name__ == '__main__':
    test2()
