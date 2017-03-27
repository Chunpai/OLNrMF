import numpy as np
import networkx as nx
import random

def randomGraph(n,p):
    G = nx.Graph()
    outfile = open('../datasets/synthetic1.csv','w')
    for i in range(n):
        for j in range(n):
            if i != j:
                val = random.uniform(0,1)
                if val <= p:
                    outfile.write(str(i)+' '+str(j)+'\n')
                    G.add_edge(i,j) 
    outfile.close()
    return G 

def dataFormat():
    infile = open('../datasets/cit-HepPh.txt','r')
    outfile = open('../datasets/hepPhCit.csv','w')
    for line in infile:
        fields = line.strip().split('\t')
        print fields
        outfile.write(fields[0]+';'+fields[1]+'\n')
    infile.close()
    outfile.close()


def scaleFreeNetwork(n,m,size):
    G = nx.barabasi_albert_graph(n,m)
    G = injectAnomalies(G,size) 
    outfile = open('../datasets/synthetic20.csv','w')
    for (source,target) in G.edges():
        outfile.write(str(source)+';'+str(target)+'\n')
    outfile.close()

def injectAnomalies(G,size):
    n = G.number_of_nodes()
    ground_truth_file = open('../datasets/synth20_ground.csv','w')
    while(size>0):
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
    nodes_dict = {}
    for line in infile:
        fields = line.strip().split(';')
        [source,target] = fields
        if source not in nodes_dict:
            nodes_dict[source] = 1
        if target not in nodes_dict:
            nodes_dict[target] = 1
        if source not in M:
            M[source] = {}
            M[source][target] = 1.0
        else:
            M[source][target] = 1.0
    infile.close()
    return  M, nodes_dict





if __name__ == '__main__':
    n = 1000
    m = 20
    size = 1000
    scaleFreeNetwork(n,m,size)
    #dataFormat()
