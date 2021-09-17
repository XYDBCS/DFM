import numpy as np
import random
def read_txt1(path):
    with open(path, 'r', newline='') as txt_file:
        md_data = []
        reader = txt_file.readlines()
        for row in reader:
            line = row.split( )
            row = []
            for k in line:
                row.append(float(k))
            md_data.append(row)
        md_data = np.array(md_data)
        m,n = md_data.shape
        md_data_new = np.zeros([m,n])
        for i in range(m):
            for j in range(n):
                md_data_new[i,j] = int(md_data[i, j])
        return md_data

def get_all_the_samples(A): # return the same number of negative samples and postive samples, and all the negative samples
    m,n = A.shape
    pos = []
    neg = []
    for i in range(m):
        for j in range(n):
            if A[i,j] ==1:
                pos.append([i,j,1])
            else:
                neg.append([i,j,0])
    n = len(pos)
    neg_new = random.sample(neg, n)
    tep_samples = pos + neg_new
    samples = random.sample(tep_samples, len(tep_samples))
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    neg = np.array(neg)
    pos = np.array(pos)
    return samples, neg, pos

def get_balance_samples(A): # return the same number of negative samples and postive samples, and all the negative samples
    m,n = A.shape
    pos = []
    neg = []
    for i in range(m):
        temp_neg_row = []
        pos_row_n = 0
        for j in range(n):
            if A[i,j] ==1:
                pos_row_n = pos_row_n+1
                pos.append([i,j,1])
            else:
                temp_neg_row.append([i,j,0])
        neg = neg + temp_neg_row
    n = len(pos)
    neg_new = random.sample(neg, n)
    tep_samples = pos + neg_new
    samples = random.sample(tep_samples, len(tep_samples))
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    neg = np.array(neg)
    return samples, neg


def update_Adjacency_matrix (A, test_samples):
    m = test_samples.shape[0]
    A_tep = A.copy()
    for i in range(m):
        if test_samples[i,2] ==1:
            #print("index", test_samples[i,0], test_samples[i,1] )
            A_tep [int(test_samples[i,0]), int(test_samples[i,1])] = 0
    return A_tep

def array_list(arry):
    li = arry.tolist()
    return li

def get_lapl_matrix(sim):
    m,n = sim.shape
    lap_matrix_tep = np.zeros([m,m])
    for i in range(m):
        lap_matrix_tep[i,i] = np.sum(sim[i,:])
    lap_matrix = lap_matrix_tep - sim
    return lap_matrix


def data_transform2 (A, neg,  k):#k as the same meaning of the upper function
    n = neg.shape[0]
    h, v = A.shape
    test_fea = np.zeros([n, h+v])
    test_y = np.zeros([n])
    A_T = A.transpose()
    for i in range(n):
        test_fea[i,:] = np.hstack((A[neg[i,0], :], A_T[neg[i, 1],:]))
        test_y[i] = neg[i,2]
    m,n =test_fea.shape
    if k ==1:
        train_xi = np.zeros([m,n])
        train_xv = np.ones([m,n])
        #test_xi = np.zeros([int(m1/10),n1])
        #test_xv = np.ones([int(m1/10),n1])
        for i in range(m):
            for j in range(n):
                train_xi[i,j] = 2*j+test_fea[i,j]
        # for i in range(int(m1/10)):
        #     for j in range(n1):
        #         test_xi[i,j] = 2*j+test_fea[i,j]
    return train_xi, train_xv

def get_feature_label(new_matrix, train_samples, k):#k is the choose of which kind of model do we use, when k =1, it use 0,1 for each connection between a pair of miRNA-disease
    m= train_samples.shape[0]
    h, v = new_matrix.shape
    new_matrix_T = new_matrix.transpose()
    if k==1:
        train_fea = np.zeros([m, h+v])
        train_y = np.zeros([m])
        for i in range(m):
            train_fea[i,:] =np.hstack((new_matrix[train_samples[i,0], :], new_matrix_T[train_samples[i, 1], :]))
            train_y[i] = train_samples[i,2]
    y_train_ = array_list(train_y)
    return train_fea, y_train_

def data_transform (train_fea,  k):#k as the same meaning of the upper function
    m,n = train_fea.shape
    if k ==1:
        train_xi = np.zeros([m,n])
        train_xv = np.ones([m,n])
        for i in range(m):
            for j in range(n):
                train_xi[i,j] = 2*j+train_fea[i,j]

    Xi_train_ = array_list(train_xi)
    Xv_train_ = array_list(train_xv)
    return Xi_train_, Xv_train_
