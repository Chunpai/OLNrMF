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
    G1 = nx.Graph()   #before 2010
    G2 = nx.Graph()   #after 2010
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
            G1.add_edge(user_index,item_index,weight=float(rating))  
            G2.add_edge(user_index,item_index,weight=0.0)
        else:
            G2.add_edge(user_index,item_index,weight=float(rating))  
            G1.add_edge(user_index,item_index,weight=0.0)
    infile.close()
    M1 = np.array(nx.to_numpy_matrix(G1)) 
    M2 = np.array(nx.to_numpy_matrix(G2)) 
    #dumpPickle(data_dir, M1, M2)
    return M1, M2


def dumpPickle(data_dir, M1, M2):
    pickle.dump(M1, open('M1.pkl','wb'))
    pickle.dump(M2, open('M2.pkl','wb'))  
 

def loadPickle(data_dir):
    M1 = pickle.load(open('M1.pkl','rb'))
    M2 = pickle.load(open('M2.pkl','rb'))
    return M1, M2



if __name__ == '__main__':
    data_dir = 'datasets/Amazon/'
    filename = 'ratings_Movies_and_TV.csv'
    preprocess(data_dir,filename)
