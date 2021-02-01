import numpy as np
import matplotlib.pyplot as plt
from data import libsvm_dataset
import matplotlib.pylab as pylab
import sys
import time

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

np.random.seed(0)

# Algorithmic parameter sweep
base = 1
power = 20
delta_fixed = []
delta_alpha = []

for i in range(power-1):
    delta_alpha.append(1.0/((power-i)*base)**2)
    delta_fixed.append(1.0/((power-i)*base)**2)

for i in range(power-1):
    if (i>0):
        delta_fixed.append(1-1.0/((i+1)*base)**2)

for i in range(power):
    delta_alpha.append(((i+1)*base)**2)

rand_prob = [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def synthetic_dataset():
    # Parameters for synthetic setup
    task_num = 20
    single_task_t_max = 100
    single_task_d_max = 6

    t = np.zeros(task_num, dtype=int)
    d = np.zeros(task_num, dtype=int)
    ds = []
    c_t = 0
    for i in range(task_num):
        tcoin = np.random.randint(2)+1

        t[i] = 50*tcoin
        d[i] = 3*(3-tcoin)
        ds.append(np.arange(c_t, c_t+d[i]))
        c_t += d[i]
    tot_d = c_t+1
    print('Total dimension', tot_d)

    # Construct X
    task_seq = np.arange(task_num)
    X = []
    tot_t = 0
    for task in task_seq:
        for i in range(t[task]):
            xi = np.random.normal(0, 1, d[task])
            xi = xi / np.linalg.norm(xi, ord=2) # Normalize to l2 norm 1
            xj = np.zeros(tot_d)
            j = 0
            for dim in ds[task]:
                xj[dim] = xi[j]
                j += 1
            X.append(xj)
        tot_t += t[task]
    X = np.asarray(X)

    # Ground truth theta
    theta = np.random.normal(0, 1, tot_d)
    theta = theta / np.linalg.norm(theta) # Normalize to l2 norm 1

    # Generate Y
    Y = np.matmul(X, theta)
    Y = Y + np.random.normal(0, 0.1, tot_t)

    return X, Y, theta

# Choose mode from 'uniform', 'delta_random', 'greedy'
# Assume X, theta, Y are all normalized to 1
def adapt_loss(X, Y, theta, alpha, thr, mode):
    use_theta = False
    w = 0.0
    tot_loss = 0.0
    tot_d = X.shape[1]
    M_inv = np.eye(tot_d,dtype=float)
    X_Q = []
    Y_Q = []
    t_prime = -1
    delta_rec = []
    for i in range(X.shape[0]):
        delta_r = np.matmul(M_inv, X[i].T)
        delta = np.matmul(X[i], delta_r)
        w += delta
        delta_rec.append(delta)
        Y_Q_array = np.asarray(Y_Q)
        X_Q_array = np.asarray(X_Q)
        if len(Y_Q_array)>0:
            yhat = np.matmul(Y_Q_array.T, X_Q_array)
            yhat = np.matmul(yhat, delta_r)
        else:
            yhat = 0
        yhat = np.clip(yhat, -1, 1)

        if use_theta:
            tot_loss += (np.dot(X[i], theta)-yhat)**2
        else:
            tot_loss += (Y[i]-yhat)**2

        if (mode=='delta_fixed'):
            query = (delta > thr)
        elif (mode=='uniform'):
            query = (np.random.rand() < thr) # thr is probability
        elif (mode=='greedy'):
            query = (i < thr*X.shape[0])
        else:
            query = (np.random.rand() < delta*alpha) # delta is probability

        if query:
            xM_1 = np.matmul(X[i], M_inv)
            M_inv = M_inv - np.outer(xM_1, xM_1.T)/(1+np.dot(xM_1, X[i]))
            w = 0
            t_prime = i
            X_Q.append(X[i])
            Y_Q.append(Y[i])

    return len(X_Q), tot_loss

if __name__ == "__main__":
    # Choose from 'synthetic', 'cpusmall', 'abalone'
    dataset = sys.argv[1]
    # Hyperparameters for each dataset
    if (dataset=='synthetic'):
        rand_prob = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        delta_alpha = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
        delta_len = len(delta_alpha)
        s_thr = 0.016
    elif (dataset=='cpusmall'):
        delta_len = 21
        s_thr = 0.12
        X, Y = libsvm_dataset(13, False, 'cpusmall_scale')
        theta = 0
    else:
        delta_len = 22
        s_thr = 0.063
        X, Y = libsvm_dataset(9, False, 'abalone_scale')
        theta = 0

    num_exp = 5
    rand_q = np.zeros((num_exp, len(rand_prob)))
    rand_r = np.zeros((num_exp, len(rand_prob)))
    loss_greedy_q = np.zeros(len(rand_prob))
    loss_greedy_r = np.zeros(len(rand_prob))
    delta_q = np.zeros((num_exp, delta_len))
    delta_r = np.zeros((num_exp, delta_len))

    start = time.time()
    end = start

    for exp in range(num_exp):
        print('Experiment', exp)
        if dataset=='synthetic':
            X, Y, theta = synthetic_dataset()

        # Greedy
        if (exp==0):
            x1 = []
            y1 = []
            for prob in rand_prob:
                x, y = adapt_loss(X, Y, theta, alpha=0, thr=prob, mode='greedy')
                x1.append(x)
                y1.append(y)
            loss_greedy_q = loss_greedy_q + np.array(x1)
            loss_greedy_r = loss_greedy_r + np.array(y1)
            print('Greedy done')
            end = time.time()

        # Uniform querying
        x1 = []
        y1 = []
        for prob in rand_prob:
            x, y = adapt_loss(X, Y, theta, alpha=0, thr=prob, mode='uniform')
            x1.append(x)
            y1.append(y)
        rand_q[exp] = np.array(x1)
        rand_r[exp] = np.array(y1)
        print('Uniform done')
        end = time.time()

        # Our algorithm
        x1 = []
        y1 = []
        for s in delta_alpha:
            if s < s_thr:
                continue
            x, y = adapt_loss(X, Y, theta, alpha=s, thr=0, mode='delta_random')
            if (x > 0.01*X.shape[0]):
                x1.append(x)
                y1.append(y)
            else:
                print('Error: Number of queries < 0.01T.')
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
    plt.title(dataset+' dataset')
    plt.tight_layout()
    plt.savefig(dataset+'.png')
