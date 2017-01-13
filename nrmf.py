import numpy as np
import numpy.linalg as LA
import math
import datetime
import matplotlib.pyplot as plt
import random
import readData
from copy import deepcopy



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



def AltQP_Inc(M1,n,r,l):
    """
    input: original n*l matrix A (here we use A_dict to represent matrix), and rank size r
    output: n*r matrix F, r*l matrix G, and an n*l matrix R
    """ 
    #R_dict =  movie_dict #residual matrix initialized as a dict due to sparsity
    F, G = initialization(n,r,l)
    R = deepcopy(M1)
    for k in range(r):
        f, g = RankOneApproximation(R,n,l)
        F[:,k] = f
        G[k,:] = g
        for user_index in R:
            for item_index in R[user_index]:
                if R[user_index][item_index] > 0.0:
                    R[user_index][item_index] -= f[user_index]*g[item_index]
                else:
                    print 'ERROR--------------------------------------------' 
    return F, G, R
   

def RankOneApproximation(R,n,l):
    """
    input: A_dict, the matrix needs to be factorized
    output: a colunmn vector f (movie feature), and a row vector g (user feature)
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
    print 'f',f
    print 'g',g
    iteration = 0
    while convergent == False:
        total_error = 0.0
        iteration += 1
        print 'ITERATION',iteration
        g = Update_g(R,f,l)
        print 'g updated'
        f = Update_f(R,g,n)  #note the order of parameters n and l
        print 'f updated'
        P = outer_prod(f,g)
        for user_index in R:
            for item_index in R[user_index]:
                rating = R[user_index][item_index] 
                if rating > 0.0:
                    if user_index in P and item_index in P[user_index]:
                        approx = P[user_index][item_index]
                        print 'approx',approx
                        error = rating - approx
                        print 'error1',error
                    else:
                        error = rating
                        print 'error2',error
                    if error < 0.0:
                        print 'ERROR------------------------------------2'
                        print 'error3',error
                        print 'rating',rating
                        print 'prod', P[user_index][item_index]
                        return
                    elif error == 0.0:
                        print 'Good-------------------------------------'
                    else:
                        total_error += math.pow(error,2)
        total_error = math.sqrt(total_error)
        if total_error <= 1000.0 or iteration >= 100:
            convergent = True
            print 'convergent-------------------------------------------'
        else:
            print 'total_error',total_error
    return f, g


def Update_g(R,f,l):
    g = np.zeros(l)
    for j in range(l):
        low = float('-inf')
        up = float('inf')
        t = 0.0
        q = 0.0
        for user_index in R:
            f_value = f[user_index]
            if j in R[user_index]:
                if R[user_index][j] > 0.0:
                    q = q + f_value * R[user_index][j]
                    t = t + math.pow(f_value,2)
                    if f_value > 0:
                        up = min(up, R[user_index][j] / f_value) 
                    elif f_value < 0:
                        low = max(low, R[user_index][j] / f_value)
                    else:
                        continue
            else:
                continue
        if t == 0:
            g[j] = 0.0 
            continue
        q = q / t
        q = float("{0:.2f}".format(q))
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
                        up = float("{0:.2f}".format(up))
                    elif g_value < 0:
                        low = max(low, R[i][item_index] / g_value)
                        low = float("{0:.2f}".format(low))
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
             


def plotResult(R,n,l):
    plt.axis([0,l+1,0,n+1])
    count = 0
    for user_index in R:
        user_list = []
        item_list = []
        for item_index in R[user_index]:
            if R[user_index][item_index] > 0.0000:
                count += 1
                user_list.append(user_index)
                item_list.append(item_index)
        plt.plot(user_list, item_list, 'b.')
    plt.savefig('anomaly.png')
    print count




if __name__ == "__main__": 
    data_dir = 'datasets/Amazon/top1000/'
    filename = 'ratings.csv'

    M1, M2 ,user_dict, item_dict = readData.loadPickle(data_dir)
    n = len(user_dict)
    l = len(item_dict)
    r = 1
    AltQP_Inc(M1,n,r,l)
    



