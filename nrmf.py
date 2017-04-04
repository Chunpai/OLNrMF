import numpy as np
import numpy.linalg as LA
import math
import datetime
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import data


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



def AltQP_Inc(M,n,r,l):
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
    g = np.random.rand(l)

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
    for j in range(l): #for element in the row vector
        low = float('-inf')
        up = float('inf')
        t = 0.0
        q = 0.0
        for source in R:   
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
    l = f.size
    for i in range(n):
        P[i] = {}
        if f[i] != 0.0:
            for j in range(l):
                if g[j] != 0.0:
                    P[i][j] = f[i]*g[j]
    return P 
             


def plotResidual(data_dir,R,anomaly,n,r,l):
    plt.axis([0,n+1,0,l+1])
    count = 0
    acount = 0
    outfile = open(data_dir+'residuals2/R'+str(r)+'.csv','w')
    outfile2 = open(data_dir+'result2.csv','a')
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
                outfile.write(str(source)+';'+str(target)+'\n')
                if source in anomaly:
                    if target in anomaly[source]:
                        acount += 1
                        anomaly_source_list.append(source)
                        anomaly_target_list.append(target)
        plt.plot(source_list, target_list, 'b.', markersize=0.5)
        plt.plot(anomaly_source_list, anomaly_target_list, 'r.', markersize=0.5)
    outfile.close()
    plt.savefig(data_dir+'residuals2/R'+str(r)+'.png')
    outfile2.write(str(r)+';'+str(count)+';'+str(acount)+'\n')
    outfile2.close()
    print r
    print count
    print acount


def residualNetowrkAnalysis(injected_M,anomaly,n,rank,l):
    '''
    analysis of residual network after each rank-one approximation
    '''
    R = injected_M
    for i in range(rank):
        f,g,R = AltQP_Inc(R,n,1,l)
        print 'iteration:',i
        res_edges = 0
        ano_edges = 0
        res_nodes = {}
        ano_nodes = {}
        for source in R:
            for target in R[source]:
                if R[source][target] >= 0.00001:
                    R[source][target] = 1.0
                    res_edges += 1
                    if source not in res_nodes:
                        res_nodes[source] = 1
                    if target not in res_nodes:
                        res_nodes[target] = 1

                    if source in anomaly:
                        if target in anomaly[source]:
                            ano_edges += 1
                            if source not in ano_nodes:
                                ano_nodes[source] = 1
                            if target not in ano_nodes:
                                ano_nodes[target] = 1
                else:
                    R[source][target] = 0.0
        print 'residual edges:',res_edges
        print 'residual nodes:',len(res_nodes)
        print 'ratio:', res_edges*1.0/len(res_nodes)
        print 'anomaly edges:',ano_edges
        print 'anomaly nodes:',len(ano_nodes)
        #print 'ratio:', ano_edges*1.0/len(ano_nodes)

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
    rank = 20
    residualNetowrkAnalysis(injected_M,anomaly,n,rank,l)
    
   


if __name__ == "__main__": 
   test2() 
