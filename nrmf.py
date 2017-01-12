import numpy as np
import numpy.linalg as LA
import math
import datetime
import matplotlib.pyplot as plt
import random
import readData




def initialization(n,l,r):
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



def AltQPInc(movie_dict,user_dict, F,G,n,l,r):
    """
    input: original n*l matrix A (here we use A_dict to represent matrix), and rank size r
    output: n*r matrix F, r*l matrix G, and an n*l matrix R
    """ 
    #R_dict =  movie_dict #residual matrix initialized as a dict due to sparsity
    for k in range(r):
        f, g = RankOneApproximation(movie_dict, user_dict,n,l,r)
        F[:,k] = f
        G[k,:] = g
        for movie in movie_dict:
            for user in movie_dict[movie]:
                if movie_dict[movie][user] > 0.0:
                    movie_dict[movie][user] = movie_dict[movie][user] - f[movie-1]*g[user-1]
                    user_dict[user][movie] = user_dict[user][movie] - f[movie-1]*g[user-1]
    return F, G, movie_dict, user_dict
   

def RankOneApproximation(movie_dict,user_dict,n,l,r):
    """
    input: A_dict, the matrix needs to be factorized
    output: a colunmn vector f (movie feature), and a row vector g (user feature)
    """
    convergent = False
    f = []
    g = []
    for i in range(n):
        f.append(random.randint(-5,5)/1.0)
    for j in range(l):
        g.append(random.randint(-5,5)/1.0)
    f = np.array(f)
    f = f.T
    g = np.array(g)
    #print "f",f
    #print "g",g
    product = np.outer(f,g)
    #print "product",product
    while convergent == False:
        total_error = 0.0
        g = Update_g(movie_dict,f,g,n,l,r)
        #print "update-g",g
        f_hat = Update_g(user_dict,g.T,f.T,l,n,r)  #note the order of parameters n and l
        f = f_hat.T
        #print "update-f",f
        product_next = np.outer(f,g)
        """
        error = LA.norm(product_next - product)
        norm = LA.norm(product_next)
        """
        for movie in movie_dict:
            for user in movie_dict[movie]:
                rating = movie_dict[movie][user]
                if rating > 0.0:
                    error = product_next[movie-1][user-1] - rating 
                    if error > 0.0:
                        abc= "OMGGGGGGGGGGGGGGG, SOMTHING WRONG"
                    elif error == 0.0:
                        abc= "gooooooooooooooooooooooood"
                    else:
                        total_error += math.pow(error,2)
        #print "norm",norm
        total_error = math.sqrt(total_error)
        #print "total_error",total_error
        #print "product_next",product_next
        if total_error > 1000.0:
            product = product_next
        else:
            convergent = True
            #print "convergent------------------------------------------------------------"
    return f, g


def Update_g(A_dict,f,g,n,l,r):
    """
    input: the origin matrix (movie_dict) needs to be factorized, and a column vector
    output: a row vector
    """
    for j in range(l):
        low = float("-inf")
        up = float("inf")
        t = 0.0
        q = 0.0
        for movie in A_dict:
            user = j+1
            if user in A_dict[movie]:
                if A_dict[movie][user] > 0.0:
                    q = q + f[movie-1] * A_dict[movie][user]
                    t = t + math.pow(f[movie-1],2)
                    #print f[movie-1]
                    if f[movie-1] > 0:
                        up = min(up, A_dict[movie][user]/f[movie-1]) 
                    elif f[movie-1] < 0:
                        low = max(low, A_dict[movie][user]/f[movie-1])
                    else:
                        continue
        if t == 0:
            g[j] = 0.0 
            continue
        q = q / t
        if q <= up and q >= low:
            g[j] = q
        elif q > up:
            g[j] = up
        else:
            g[j] = low  
    return g    


def plotResult(R_dict,n,l):

    plt.axis([0,l+1,0,n+1])
    count = 0
    for movie in R_dict:
        user_list = []
        movie_list = []
        for user in R_dict[movie]:
            if R_dict[movie][user] > 0.00001:
                count += 1
                user_list.append(user)
                movie_list.append(movie)
        plt.plot(user_list, movie_list, "b.")
    plt.savefig("origin.png")
    print count




if __name__ == "__main__": 
    data_dir = 'datasets/Amazon/'
    filename = 'ratings_Movies_and_TV.csv'
    M1, M2 = readData.preprocess(data_dir, filename)
    print M1.shape, M2.shape
