import numpy as np
import networkx as nx
import random

from copy import deepcopy

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
    '''
    extract first 100 user data from  movie_len dataset, and store the data
    in preferred format 
    '''
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


def generateAnomalies_1(M,source_dict,target_dict,data_dir):
    outfile = open(data_dir+'ground_truth1.csv','w')
    count = 0
    for i in range(2):
        anomaly_source = random.choice(source_dict.keys())
        target_size = len(target_dict)
        for j in range(target_size/2): 
            anomaly_target = random.choice(target_dict.keys())
            if anomaly_target not in M[anomaly_source]:
                count += 1
                outfile.write(str(anomaly_source)+';'+str(anomaly_target)+'\n')
    outfile.close()
    print 'anomaly edges',count
    return count


def generateAnomalies_2(M,source_dict,target_dict,data_dir):
    outfile = open(data_dir+'ground_truth2.csv','w')
    count = 0
    for i in range(10):
        anomaly_target = random.choice(target_dict.keys())
        source_size = len(source_dict)
        for j in range(source_size*3): 
            anomaly_source = random.choice(source_dict.keys())
            if anomaly_target not in M[anomaly_source]:
                count += 1
                outfile.write(str(anomaly_source)+';'+str(anomaly_target)+'\n')
    outfile.close()
    print 'anomaly edges',count
    return count


def readNetwork(data_dir, filename):
    infile = open(data_dir+filename,'r') 
    M = {}
    source_dict = {}
    target_dict = {}
    count = 0
    for line in infile:
        fields = line.strip().split(';')
        source = int(fields[0])
        target = int(fields[1])
        if source not in source_dict:
            source_dict[source] = 1
        if target not in target_dict:
            target_dict[target] = 1
        if source not in M:
            count += 1
            M[source] = {}
            M[source][target] = 1.0
        else:
            count += 1
            M[source][target] = 1.0
    infile.close()
    print 'total edges', count
    print 'source', len(source_dict)
    print 'target', len(target_dict)
    return  M, count, source_dict, target_dict


def injectAnomalies(data_dir, M):
    infile = open(data_dir + 'ground_truth2.csv','r')
    injected_M = deepcopy(M)
    anomaly = {}
    count = 0
    for line in infile:
        fields = line.strip().split(';')
        source = int(fields[0])
        target = int(fields[1])
        if source not in anomaly:
            anomaly[source] = {}
            anomaly[source][target] = 1.0
        else:
            anomaly[source][target] = 1.0
        if source not in injected_M:
            count += 1
            injected_M[source] = {}
            injected_M[source][target] = 1.0
        else:
            if target not in injected_M[source]:
                count += 1
                injected_M[source][target] = 1.0
    infile.close()
    print 'anomaly edges:',count
    return  injected_M, anomaly, count




if __name__ == '__main__':
    data_dir = 'datasets/100_ml_ratings/'
    filename = '100_ml_ratings.csv' 
    M, count, source_dict, target_dict = readNetwork(data_dir,filename)
    n = len(source_dict)
    l = len(target_dict)
    #generateAnomalies_2(M,source_dict,target_dict,data_dir)
    injected_M, anomaly, acount = injectAnomalies(data_dir, M)
    plotNetwork(data_dir,M,anomaly,n,l)
    #outfile = open(data_dir+'result2.csv','w')
    #outfile.write(str(count)+';'+str(acount)+'\n')
    #outfile.close()
