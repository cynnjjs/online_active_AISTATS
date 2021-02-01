import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import matplotlib.pylab as pylab
from utils import process_amazon_data

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

np.random.seed(0)

def adapt_loss(X, y, alpha, thr, mode):
    X_Q = []
    Y_Q = []
    tot_loss = 0
    
    T = len(y)
    C = max(np.linalg.norm(x) for x in X)
    M_inv = np.eye(X[0].shape[1],dtype=float)/C

    delta_rec=[]
    cum_delta = 0
    last_query = -1
    query_ind = []

    for i in range(T):
        delta_r = np.matmul(M_inv, X[i].T)
        delta = np.matmul(X[i], delta_r)[0][0]
        delta_rec.append(delta)
        cum_delta += delta

        if len(Y_Q)>0:
            yhat = np.matmul(y_pre, delta_r)
        else:
            yhat = 3 # No query so far, predict score 3 as default
        y_pred = yhat

        if (mode=='greedy'):
            query = (i < thr*T)
        elif (mode=='uniform'):
            query = (np.random.rand() < thr) # thr is probability
        else:
            query = (np.random.rand() < delta*alpha) # qufur

        if query:
            cum_delta = 0
            last_query = i
            xM_1 = np.matmul(X[i], M_inv)
            M_inv = M_inv - np.outer(xM_1, xM_1.T)/(1+np.inner(xM_1, X[i])[0][0])

            X_Q.append(X[i])
            Y_Q.append(y[i])
            y_pre = np.matmul(np.asarray(Y_Q).T, np.concatenate(X_Q, axis=0))
            query_ind.append(1)
        else:
            query_ind.append(0)
        tot_loss += (y[i]-y_pred)**2
    print('Online total loss', tot_loss[0])
    print('Query frequency', len(Y_Q)/T)

    return len(Y_Q), tot_loss[0]

if __name__ == "__main__":
    # Datapoints for Games, Grocery, Auto, respectively
    n1 = 1200
    n2 = 600
    n3 = 300

    X1 = pickle.load(open(f'Bert_X_words.pkl', 'rb'))
    Y1 = pickle.load(open(f'Bert_Y_words.pkl', 'rb'))

    config = sys.argv[1]
    if config==0: # 300 Auto + 600 Grocery + 1200 Games
        X = X1[n1+n2:]+X1[n1:n1+n2]+X1[:n1]
        Y = Y1[n1+n2:]+Y1[n1:n1+n2]+Y1[:n1]
    elif config==1: # 1200 Games + 600 Grocery + 300 Auto
        X = X1
        Y = Y1
    else: # Uniform
        T = n1+n2+n3
        idx = np.random.permutation(T)
        X = np.asarray(X1)[idx]
        Y = np.asarray(Y1)[idx]

    num_exp = 5

    # Sweep parameter range for plotting
    alpha_list = [0.0625, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 8]
    rand_prob = [0.05, 0.075, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    det_thr= [0.001, 0.004, 0.016, 0.064, 0.256, 0.512, 1, 2, 4, 8, 32, 128, 256]

    rand_q = np.zeros((num_exp, len(rand_prob)))
    rand_r = np.zeros((num_exp, len(rand_prob)))
    loss_greedy_q = np.zeros(len(rand_prob))
    loss_greedy_r = np.zeros(len(rand_prob))
    delta_q = np.zeros((num_exp, len(alpha_list)))
    delta_r = np.zeros((num_exp, len(alpha_list)))

    start = time.time()
    end = start

    idx = np.zeros(len(Y))

    for exp in range(num_exp):
        print('Experiment', exp)

        # Greedy
        if (exp==0):
            x1 = []
            y1 = []
            for prob in rand_prob:
                x, yy = adapt_loss(X, Y, alpha=0, thr=prob, mode='greedy')
                x1.append(x)
                y1.append(yy)
            loss_greedy_q = loss_greedy_q + np.array(x1)
            loss_greedy_r = loss_greedy_r + np.array(y1)
            print('Greedy done')
            end = time.time()

        # Uniform querying
        x1 = []
        y1 = []
        for prob in rand_prob:
            x, yy = adapt_loss(X, Y, alpha=0, thr=prob, mode='uniform')
            x1.append(x)
            y1.append(yy)
        rand_q[exp] = np.array(x1)
        rand_r[exp] = np.array(y1)
        print('Uniform done')
        end = time.time()

        # Our algorithm
        x1 = []
        y1 = []
        for s in alpha_list:
            x, yy = adapt_loss(X, Y, alpha=s, thr=0, mode='qufur')
            x1.append(x)
            y1.append(yy)

        delta_q[exp] = np.array(x1)
        delta_r[exp] = np.array(y1)
        print('Qufur done')
        end = time.time()

    end = time.time()

    std_rand_r = np.std(rand_r, axis=0)
    std_delta_r = np.std(delta_r, axis=0)
    mean_rand_r = np.mean(rand_r, axis=0)
    mean_delta_r = np.mean(delta_r, axis=0)
    std_rand_q = np.std(rand_q, axis=0)
    std_delta_q = np.std(delta_q, axis=0)
    mean_rand_q = np.mean(rand_q, axis=0)
    mean_delta_q = np.mean(delta_q, axis=0)

    plt.figure()
    plt.plot(loss_greedy_q, loss_greedy_r, marker='o', label='Greedy')
    plt.errorbar(mean_rand_q, mean_rand_r, xerr=std_rand_q, yerr=std_rand_r, fmt='-o', label='Uniform queries')
    plt.errorbar(mean_delta_q, mean_delta_r, xerr=std_delta_q, yerr=std_delta_r, fmt='-o', label='QuFUR')
    plt.xlabel('Total # Queries')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.title('Amazon Review Dataset')
    plt.tight_layout()
    plt.savefig('Amazon_tradeoff.png')
