import numpy as np
import networkx as nx
import random

def randomGraph(n,p):
    G = nx.Graph()
    outfile = open('datasets/synthetic1.csv','w')
    for i in range(n):
        for j in range(n):
            if i != j:
                val = random.uniform(0,1)
                if val <= p:
                    outfile.write(str(i)+' '+str(j)+'\n')
                    G.add_edge(i,j) 
    outfile.close()
    return G 

def dataFormat1():
    infile = open('datasets/cit-HepPh.txt','r')
    outfile = open('datasets/hepPhCit.csv','w')
    for line in infile:
        fields = line.strip().split('\t')
        print fields
        outfile.write(fields[0]+';'+fields[1]+'\n')
    infile.close()
    outfile.close()


def dataFormat2():
    infile = open('datasets/ml-latest-small/ratings_100.csv','r')
    outfile = open('datasets/ml_ratings_100.csv','w')
    infile.readline()
    uindex = 0
    mindex = 0
    user_dict = {}
    movie_dict = {}
    movie_list = []
    lines = infile.readlines()
    for line in lines:
        fields = line.strip().split(',')
        movieId = fields[1]
        if movieId not in movie_list:
            movie_list.append(movieId)
    movie_new_list = range(len(movie_list))
    for movieId in sorted(movie_list):
        movie_dict[movieId] = movie_new_list.pop(0)
    for line in lines:
        fields = line.strip().split(',')
        userId = fields[0]
        movieId = fields[1]
        if userId not in user_dict:
            userNewId = uindex
            user_dict[userId] = uindex
            uindex += 1
        else:
            userNewId = user_dict[userId] 
        movieNewId = movie_dict[movieId]
        outfile.write(str(userNewId)+';'+str(movieNewId)+'\n')
    infile.close()
    outfile.close()


def scaleFreeNetwork(n,m,size,name):
    G = nx.barabasi_albert_graph(n,m)
    G = injectAnomalies(G,size,name) 
    outfile = open('datasets/'+name+'.csv','w')
    for (source,target) in G.edges():
        outfile.write(str(source)+';'+str(target)+'\n')
        if source > target:
            print 'hahahahahahahahahahahaha'
    outfile.close()


def injectAnomalies(G,size,name):
    n = G.number_of_nodes()
    ground_truth_file = open('datasets/'+name+'_ground.csv','w')
    while(size>=0):
        source = random.randint(0,n)
        target = random.randint(0,n)
        if source != target and (source,target) not in G.edges():
            G.add_edge(source,target) 
            ground_truth_file.write(str(source)+';'+str(target)+'\n')
            size -= 1
    ground_truth_file.close()
    return G 


def toMatrix(data_dir, filename):
    '''
    read data, build dictionary, store into numpy ndarray, and save as pickle format for data serialization
    '''
    infile = open(data_dir+filename,'r') 
    M = {}
    source_dict = {}
    target_dict = {}
    for line in infile:
        fields = line.strip().split(';')
        source = int(fields[0])
        target = int(fields[1])
        if source not in source_dict:
            source_dict[source] = 1
        if target not in target_dict:
            target_dict[target] = 1
        if source not in M:
            M[source] = {}
            M[source][target] = 1.0
        else:
            M[source][target] = 1.0
        '''    
        if target not in M:
            M[target] = {}
            M[target][source] = 1.0
        else:
            M[target][source] = 1.0
        '''
    infile.close()
    print 'source',len(source_dict)
    print 'target',len(target_dict)
    return  M, source_dict, target_dict



if __name__ == '__main__':
    #n = 1000
    #m = 20
    #size = 100
    #scaleFreeNetwork(n,m,size,'synthetic_100')
    dataFormat2()
