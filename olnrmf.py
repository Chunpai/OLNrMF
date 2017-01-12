import numpy as np
import numpy.linalg as LA
import math
import datetime
import matplotlib.pyplot as plt
import random



def readData():
    """
    read data into two dictionaries: movie_dict and user_dict 
    movie_dict: key is the movie_id, and value is the user_id
    user_dict: key is the user_id, and value is the movie id
    """
    infile = open("ml-100k/u.data","r")
    user_dict = {}
    movie_dict = {}
    user_list = []
    movie_list = []
    for line in infile:
        fields = line.strip().split()
        user = int(fields[0])
        movie = int(fields[1])
        rating = float(fields[2])
        time_stamp = int(fields[3])
        if user not in user_dict:
            user_dict[user] = {}
            user_dict[user][movie] = rating
            #user_dict[user][movie] = 1
            user_list.append(user)
        else:
            #user_dict[user][movie] = 1
            user_dict[user][movie] = rating

        if movie not in movie_dict:
            movie_dict[movie] = {}
            #movie_dict[movie][user] = 1
            movie_dict[movie][user] = rating
            movie_list.append(movie)
        else:
            movie_dict[movie][user] = rating
            #movie_dict[movie][user] = 1
    infile.close() 

    infile2 = open("test.data","r")
    for line in infile2:
        fields = line.strip().split()
        user = int(fields[0])
        movie = int(fields[1])
        rating = float(fields[2])
        if user not in user_dict:
            user_dict[user] = {}
            user_dict[user][movie] = rating
            #user_dict[user][movie] = 1
            user_list.append(user)
        else:
            user_dict[user][movie] = rating
            #user_dict[user][movie] = 1
        if movie not in movie_dict:
            movie_dict[movie] = {}
            movie_dict[movie][user] = rating
            #movie_dict[movie][user] = 1
            movie_list.append(movie)
        else:
            movie_dict[movie][user] = rating  
            #movie_dict[movie][user] = 1
    infile2.close() 

    l = len(user_dict)
    n = len(movie_dict)
    user_list.sort()
    movie_list.sort()
    print user_list
    print movie_list
    #print user_list[-1], l
    #print movie_list[-1], n
    return  movie_dict, user_dict, n, l



def readData2():
    infile = open("citation/Cit-HepTh.txt")
    for i in range(4):
        infile.readline()
    from_node_dict = {}
    to_node_dict = {}
    node_dict = {}
    for line in infile:
        fields = line.strip().split()
        from_node = int(fields[0])
        to_node = int(fields[1])
        if from_node not in from_node_dict:
            from_node_dict[from_node] = {}
            from_node_dict[from_node][to_node] = 1
        else:
            from_node_dict[from_node][to_node] = 1

        ##to_node_dict = movie_dict
        if to_node not in to_node_dict:
            to_node_dict[to_node] = {}
            to_node_dict[to_node][from_node] = 1
        else:
            to_node_dict[to_node][from_node] = 1
        
        id = 0
        if from_node not in node_dict:
            id += 1
            node_dict[from_node] = id
        if to_node not in node_dict:
            id += 1
            node_dict[to_node] = id

    print len(from_node_dict)
    print len(to_node_dict)
    n =  len(node_dict)
    l =  len(node_dict)

    infile.close() 
    return to_node_dict, from_node_dict, n, l


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
    movie_dict, user_dict, n, l = readData2() 
    """
    r = 10
    F, G = initialization(n,l,r)
    F, G, R_dict, user_dict = AltQPInc(movie_dict, user_dict, F,G, n, l, r)
    print "F",F
    print "G",G
    print "R_dict", R_dict    
    """
    plotResult(movie_dict, n,l)
