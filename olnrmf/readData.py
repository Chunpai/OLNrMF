import numpy as np
import numpy.linalg as LA
import networkx as nx
import datetime
import pickle 
import scipy as sp
import scipy.sparse.linalg as sLA


def preprocess(data_dir, filename):
    '''
    read data, build dictionary, store into numpy ndarray, and save as pickle format for data serialization
    '''
    infile = open(data_dir+filename,'r') 
    infile.readline()
    user_index = -1
    item_index = -1
    user_dict = {}
    item_dict = {}
    M1 = {}
    M2 = {}
    for line in infile:
        fields = line.strip().split(',')
        #print fields
        [user,item,rating,timestamp] = fields
        if user not in user_dict:
            user_index += 1
            user_dict[user_index] = user
        if item not in item_dict:
            item_index += 1
            item_dict[item_index] = item
        year = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y')#'%Y-%m-%d %H:%M:%S')
        if int(year) <= 2010:
            if user_index not in M1:
                M1[user_index] = {}
                M1[user_index][item_index] = float(rating)
            else:
                M1[user_index][item_index] = float(rating)
        else:
            if user_index not in M2:
                M2[user_index] = {}
                M2[user_index][item_index] = float(rating)
            else:
                M2[user_index][item_index] = float(rating)
    infile.close()
    print len(M1)
    print len(M2)
    dumpPickle(data_dir, M1, M2, user_dict, item_dict)
    #return  M1, M2, user_dict, item_dict


def dumpPickle(data_dir, M1, M2, user_dict, item_dict):
    pickle.dump(M1, open(data_dir+'M1.pkl','wb'))
    pickle.dump(M2, open(data_dir+'M2.pkl','wb'))  
    pickle.dump(user_dict, open(data_dir+'user_dict.pkl','wb'))
    pickle.dump(item_dict, open(data_dir+'item_dict.pkl','wb'))
 

def loadPickle(data_dir):
    M1 = pickle.load(open(data_dir+'M1.pkl','rb'))
    M2 = pickle.load(open(data_dir+'M2.pkl','rb'))
    user_dict = pickle.load(open(data_dir+'user_dict.pkl','rb'))
    item_dict = pickle.load(open(data_dir+'item_dict.pkl','rb'))
    return M1, M2, user_dict, item_dict


def extractTop1000(data_dir,infilename,outfilename):
    '''
    extract rating records of first 1000 users, and save as /top1000/ratings.csv
    '''
    infile = open(data_dir+infilename,'r') 
    first = infile.readline()
    outfile = open(data_dir+outfilename,'w')
    outfile.write(first)
    user_index = -1
    item_index = -1
    user_dict = {}
    item_dict = {}
    M1 = {}
    M2 = {}
    for line in infile:
        fields = line.strip().split(',')
        #print fields
        [user,item,rating,timestamp] = fields
        if user not in user_dict and user_index < 999:
            user_index += 1
            user_dict[user] = user_index
            outfile.write(line)
        elif user in user_dict:
            outfile.write(line)
    print len(user_dict)
    infile.close()
    outfile.close() 



if __name__ == '__main__':
    #data_dir = 'datasets/Amazon/'
    #infilename = 'ratings_Movies_and_TV.csv'
    
    data_dir = 'datasets/Amazon/top1000/'
    infilename = 'ratings.csv'
    preprocess(data_dir,infilename)
    #extractTop1000('datasets/Amazon/','ratings_Movies_and_TV.csv','top1000/ratings.csv')
