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
from scipy.sparse import lil_matrix
import plot


def initialization(n,r,l):
    """
    initialize F and G as matrix with 0s as all entries 
    the reason to initialize F and G as matrix rather than dictionary is because they are not sparse
    """
    F = np.zeros((n,r))   # input of np.zeros is a tuple if you want to initialize a matrix
    G = np.zeros((r,l))
    #print "F",F[:,r-1]
    #f = np.ones(n)
    #F[:,r-1] = f
    return F,G



def AltQP_Inc_General(M,n,r,l):
    """
    input: original n*l matrix A (here we use A_dict to represent matrix), and rank size r
    output: n*r matrix F, r*l matrix G, and an n*l matrix R
    """ 
    #R_dict =  movie_dict #residual matrix initialized as a dict due to sparsity
    F, G = initialization(n,r,l)
    R = deepcopy(M)
    for k in range(r):
        f, g = RankOneApproximation(R,n,l)
        F[:,k] = f
        G[k,:] = g
        for source in R:
            for target in R[source]:
                if R[source][target] > 0.0:
                    R[source][target] -= f[source]*g[target]
                #elif R[source][target] == 0.0:
                #    print 'GET ZERO'
                #else:
                #    print 'Floating ERROR--------------------------------------------' 
    return F, G, R
   

def RankOneApproximation(R,n,l):
    """
    input: R, the matrix needs to be factorized, which is stored as a dictionary
    output: a colunmn vector f and a row vector g 
    """
    convergent = False
    #f = []
    #g = []
    #for i in range(n):
    #    f.append(random.randint(-5,5)/1.0)
    #for j in range(l):
    #    g.append(random.randint(-5,5)/1.0)
    #f = np.array(f)
    #f = f.T
    #g = np.array(g)
    f = np.random.rand(n)
    print len(f)
    g = np.random.rand(l)
    print len(g)

    #print 'f',f
    #print 'g',g
    iteration = 0
    while convergent == False:
        total_error = 0.0
        iteration += 1
        #print 'ITERATION',iteration
        g = Update_g(R,f,l)
        #print 'g updated'
        f = Update_f(R,g,n)  #note the order of parameters n and l
        #print 'f updated'
        P = outer_prod(f,g)
        for source in R:
            for target in R[source]:
                rating = R[source][target] 
                if rating > 0.0:
                    if source in P and target in P[source]:
                        approx = P[source][target]
                        #print 'approx',approx
                        error = rating - approx
                        #print 'error1',error
                    else:     
                        error = rating
                        #print 'error2',error
                    if error < -0.0001:
                        print 'ERROR------------------------------------2'
                        #print 'error3',error
                        #print 'rating',rating
                        #print 'prod', P[source][target]
                        return
                    #elif error == 0.0:
                    #    print 'Good-------------------------------------'
                    else:
                        total_error += math.pow(error,2)
        total_error = math.sqrt(total_error)
        if total_error <= 1000.0 or iteration >= 100:
            convergent = True
            #print 'convergent-------------------------------------------'
        #else:
        #    print 'total_error',total_error
    return f, g


def Update_g(R,f,l):
    '''
    f is the column vector and g is the row vector
    '''
    g = np.zeros(l)   
    print 'len(g)',len(g)
    for j in range(l): #for element in the row vector
        low = float('-inf')
        up = float('inf')
        t = 0.0
        q = 0.0
        for source in R:   
            print 'source',source
            f_value = f[source]
            if j in R[source]:   
                if R[source][j] > 0.0:
                    q = q + f_value * R[source][j]
                    t = t + math.pow(f_value,2)    #if f_value is 0
                    if f_value > 0:
                        up  = min(up, R[source][j] / f_value) 
                    elif f_value < 0:
                        low = max(low, R[source][j] / f_value)
                    else:
                        continue
            else:
                continue
        if t == 0:
            g[j] = 0.0 
            continue
        q = q / t
        #q = float("{0:.7f}".format(q))
        if q <= up and q >= low:
            g[j] = q
        elif q > up:
            g[j] = up
        else:
            g[j] = low  
    return g    


def Update_f(R,g,n):
    f = np.zeros(n)
    for i in range(n):
        low = float('-inf')
        up = float('inf')
        t = 0.0
        q = 0.0
        if i in R:
            for item_index in R[i]:
                g_value = g[item_index]
                if R[i][item_index] > 0.0:
                    q = q + g_value * R[i][item_index]
                    t = t + math.pow(g_value,2)
                    if g_value > 0:
                        up = min(up, R[i][item_index] / g_value) 
                        #up = float("{0:.7f}".format(up))
                    elif g_value < 0:
                        low = max(low, R[i][item_index] / g_value)
                        #low = float("{0:.7f}".format(low))
                    else:
                        continue
        if t == 0:
            f[i] = 0.0 
            continue
        q = q / t
        if q <= up and q >= low:
            f[i] = q
        elif q > up:
            f[i] = up
        else:
            f[i] = low  
    return f 


def outer_prod(f,g):
    P = {}
    n = f.size 
    l = g.size
    for i in range(n):
        P[i] = {}
        if f[i] != 0.0:
            for j in range(l):
                if g[j] != 0.0:
                    P[i][j] = f[i]*g[j]
    return P 
             



def residualNetowrkAnalysis(data_dir,injected_M, anomaly, n, rank, l, total_edges):
    '''
    analysis of residual network after each rank-one approximation
    '''
    outfile = open(data_dir+'result2.csv','a')
    outfile.write('rank;residual_edges;largest_res_eigval;ano_edges;largest_ano_eigval\n')
    R = injected_M
    for i in range(rank):
        print 'iteration:',i
        res_edges = 0
        ano_edges = 0
        res_nodes = {}
        ano_nodes = {}
        res_matrix = np.zeros((n,l))
        ano_matrix = np.zeros((n,l))
        for source in R:
            for target in R[source]:
                if R[source][target] >= 0.000001:
                    R[source][target] = 1.0
                    res_edges += 1
                    res_matrix[source][target] = R[source][target]
                    #res_matrix[source][target] = 1.0
                    if source in anomaly:
                        if target in anomaly[source]:
                            ano_edges += 1
                            ano_matrix[source][target] = R[source][target]
                            #ano_matrix[source][target] = 1.0
        u1,s1,v1 = LA.svd(res_matrix)
        u2,s2,v2 = LA.svd(ano_matrix)
        print 'residual edges:',res_edges
        print 'anomaly edges:',ano_edges
        print 'residual eigenvalues', s1
        print 'anomaly eigenvalues', s2
        largest_res_eigval = s1[0]
        largest_ano_eigval = s2[0]
        outfile.write('{0};{1};{2};{3};{4}\n'.format(i,res_edges,largest_res_eigval,ano_edges,largest_ano_eigval))
        f,g,R = AltQP_Inc(R,n,1,l)
    outfile.close()


def test1():
    #data_dir = 'datasets/Amazon/top1000/'
    #filename = 'ratings.csv'
    #M1, M2 ,user_dict, item_dict = readData.loadPickle(data_dir)
    data_dir = 'datasets/100_ml_ratings/'
    network = '100_ml_ratings.csv'
    
    M, count, source_dict,target_dict = data.readNetwork(data_dir, network)
    n = len(source_dict)
    l = len(target_dict)
    injected_M, anomaly, acount = data.injectAnomalies(data_dir, M)
    r = 10
    F,G,R = AltQP_Inc(injected_M,n,r,l)
    
    #G, ground_nodes_dict = data.toMatrix('../datasets/','synth5_ground.csv')
    #plotResult(M,n,r,l,'whole.png')  
    plotResidual(data_dir,R,anomaly,n,r,l) 
    #plotResult(G,n,r,l,'ground.png')



def test2():
    data_dir = 'datasets/100_ml_ratings/'
    network = '100_ml_ratings.csv'
    
    M, count, source_dict,target_dict = data.readNetwork(data_dir, network)
    n = len(source_dict)
    l = len(target_dict)
    injected_M, anomaly, acount = data.injectAnomalies(data_dir, M)
    rank = 1
    residualNetowrkAnalysis(data_dir,injected_M,anomaly,n,rank,l, count)
    

def synth1(n,l):
    '''
    sythesize random bipartite graph and anomalies injected
    '''
    RB = bipartite.random_graph(n,l,0.2)
    X, Y = bipartite.sets(RB)
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
    for row in range(len(RBM)):
        source_list = []
        target_list = []
        for col in range(len(RBM[row])):
            if RBM[row][col] == 1:
                source_list.append(row)
                target_list.append(col)
                if row not in M:
                    M[row] = {}
                    M[row][col] = 1.0
                else:
                    M[row][col] = 1.0 
        plt.plot(target_list, source_list, 'b.')
    
    injected_M = deepcopy(M)
    anomaly = {}
    for i in range(1):
        anomaly_user = random.choice(range(n))
        target_size = l
        for j in range(target_size*2):
            anomaly_movie = random.choice(range(l))
            #if anomaly_movie not in injected_M[anomaly_user]:
            #    count += 1
            if anomaly_user not in injected_M:
                injected_M[anomaly_user] = {}
                injected_M[anomaly_user][anomaly_movie] = 1.0
            else:
                injected_M[anomaly_user][anomaly_movie] = 1.0

            if anomaly_user not in anomaly:
                anomaly[anomaly_user] = {}
                anomaly[anomaly_user][anomaly_movie] = 1.0
            else:
                anomaly[anomaly_user][anomaly_movie] = 1.0
                         
    for source in anomaly:
        source_list = []
        target_list = []
        for target in anomaly[source]:
            source_list.append(source)
            target_list.append(target)
        #plt.plot(source_list, target_list, 'r.', markersize=1.0)
        plt.plot(target_list, source_list, 'r.')

    plt.show()
    return M, anomaly, injected_M


def test3():
    n = 100
    l = 80
    r = 17
    M, anomaly, injected_M = synth1(n,l) 
    F,G,R = AltQP_Inc(injected_M,n,r,l)
    data_dir = '123' 
    print anomaly
    plot.plotResidual(data_dir,R,anomaly,n,r,l) 


if __name__ == "__main__": 
    test3()
