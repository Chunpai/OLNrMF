'''
apply nrmf to real rating dataset
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
import plot
import nrmf


def readWeightedNetwork(data_dir, filename):
    infile = open(data_dir+filename,'r') 
    M = {}
    source_dict = {}
    target_dict = {}
    count = 0
    for line in infile:
        fields = line.strip().split(';')
        source = int(fields[0])
        target = int(fields[1])
        weight = float(fields[2])
        if source not in source_dict:
            source_dict[source] = 1
        if target not in target_dict:
            target_dict[target] = 1
        if source not in M:
            count += 1
            M[source] = {}
            M[source][target] = weight
            print 'weight',weight
        else:
            count += 1
            M[source][target] = weight
            print 'weight', weight
    infile.close()
    print 'total edges', count
    print 'source', len(source_dict)
    print 'target', len(target_dict)
    return  M, count, source_dict, target_dict

 
def plotResidual(data_dir,R,anomaly,n,r,l):
    plt.axis([-1,l+1,-1,n+1])
    plt.xlabel('residual_movie')
    plt.ylabel('residual_user')
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
        plt.plot(target_list, source_list, 'b.', markersize=0.5)
        #plt.plot(target_list, source_list, 'b.')
        #plt.plot(anomaly_source_list, anomaly_target_list, 'r.', markersize=0.5)
        plt.plot(anomaly_target_list, anomaly_source_list, 'r.', markersize = 0.5)
    #outfile.close()
    #plt.savefig(data_dir+'residuals2/R'+str(r)+'.png')
    #plt.savefig('res_rank30.png')
    plt.show()
    #outfile2.write(str(r)+';'+str(count)+';'+str(acount)+'\n')
    #outfile2.close()
    print r
    print count
    print acount



def test5():
    data_dir = 'datasets/100_ml_ratings/'
    network = '100_ml_ratings.csv'
    
    M, count, user_dict,movie_dict = readWeightedNetwork(data_dir, network)
    n = len(user_dict)
    print n
    l = len(movie_dict)
    print l
    r = 5

    plt.axis([-1,l+1,-1,n+1])
    plt.xlabel('movie')
    plt.ylabel('user')
 
    for user in M:
        user_list = []
        movie_list = []
        for movie in M[user]:
            user_list.append(user)
            movie_list.append(movie)
        plt.plot(movie_list, user_list, 'b.', markersize = 0.5)

    injected_M = deepcopy(M)
    anomaly = {}
    for i in range(1):
        anomaly_user = random.choice(range(n))
        target_size = l
        for j in range(target_size/2):
            anomaly_movie = random.choice(range(l))
            #if anomaly_movie not in injected_M[anomaly_user]:
            #    count += 1
            if anomaly_user not in injected_M:
                injected_M[anomaly_user] = {}
                injected_M[anomaly_user][anomaly_movie] = 5.0
            else:
                injected_M[anomaly_user][anomaly_movie] = 5.0

            if anomaly_user not in anomaly:
                anomaly[anomaly_user] = {}
                anomaly[anomaly_user][anomaly_movie] = 5.0
            else:
                anomaly[anomaly_user][anomaly_movie] = 5.0
                         
    for source in anomaly:
        source_list = []
        target_list = []
        for target in anomaly[source]:
            source_list.append(source)
            target_list.append(target)
        plt.plot(target_list, source_list, 'r.', markersize = 0.5)
    #plt.savefig('full_rank30.png')
    plt.show()

    return M, anomaly, injected_M, n,r,l




if __name__ == '__main__':
    data_dir = '123'
    M, anomaly, injected_M, n, r, l = test5()
    F,G,R = nrmf.AltQP_Inc(injected_M,n,r,l)
    plotResidual(data_dir,R,anomaly,n,r,l) 
