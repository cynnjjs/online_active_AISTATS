import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import nn, optim
from utils import load_mnist_data, load_portraits_data
import time
import sys

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

np.random.seed(0)

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        x = F.log_softmax(self.fc1(x), dim=1)
        return x

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def adapt_loss(X, Y, alpha, thr, mode, plot_query=False):
    T = Y.size()[0]
    tot_loss = 0
    tot_acc = 0
    batch_size = 1
    input_size = X[0].size()[-1]
    n_way = 10
    num_batches = (T + batch_size - 1) // batch_size
    lmd = torch.norm(X, p=None, dim=1, keepdim=True)
    C = max(lmd.numpy())

    model = LinearModel(input_size, n_way)
    M_inv = np.eye(input_size,dtype=float)/C

    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=0.001)

    X_Q = []
    Y_Q = []
    delta_rec=[]
    num_query = 0
    last_query = -1
    cum_delta = 0.0
    exp_avg_mom = 0.9
    delta_avg = 0
    for b in range(num_batches):
        end_idx = min((b+1)*batch_size, T)
        images = X[b*batch_size : end_idx]
        labels = Y[b*batch_size : end_idx]
        with torch.no_grad():
            logits = model(images)
            loss = F.nll_loss(logits, labels)
            tot_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            acc = (predicted == labels).sum().float()
            tot_acc += acc

        # sum over Uncertainty of whole batch
        delta = 0
        features = images
        for z in features:
            delta_r = np.matmul(M_inv, z.T)
            delta += np.matmul(z, delta_r)

        if b==0:
            delta_avg = delta
        delta_avg = exp_avg_mom*delta_avg + (1-exp_avg_mom)*delta
        delta_rec.append(delta_avg)
        cum_delta += delta

        if (mode=='greedy'):
            query = (b < thr*num_batches)
        elif (mode=='uniform'):
            query = (np.random.rand() < thr) # thr is probability
        elif (mode=='qufur'):
            query = (np.random.rand() < delta*alpha) # qufur
        else:
            query = cum_delta*delta > thr/(b-last_query)

        if query:
            cum_delta = 0
            last_query = b

            for (x, z, y) in zip(images, features, labels):
                xM_1 = np.array(np.matmul(z, M_inv))
                M_inv = M_inv - np.outer(xM_1, xM_1.T)/(1+np.dot(xM_1, z))
                X_Q.append(x)
                Y_Q.append(y)
            num_query += batch_size

            num_epoch = 3
            batch_size1 = 64
            num_batches1 = (num_query + batch_size1 - 1) // batch_size1

            # Retrain on new dataset
            for e in range(num_epoch):
                for b1 in range(num_batches1):
                    end_idx = min((b1+1)*batch_size1, num_query)
                    images = torch.stack(X_Q[b1*batch_size1 : end_idx])
                    labels = torch.stack(Y_Q[b1*batch_size1 : end_idx])

                    optimizer.zero_grad()
                    logits = model(images)
                    loss = F.nll_loss(logits, labels)
                    loss.backward()
                    optimizer.step()

            # Only train on new example:
            if (num_epoch==0):
                optimizer.zero_grad()
                logits = model(images)
                loss = F.nll_loss(logits, labels)
                loss.backward()
                optimizer.step()

    print('Online average accuracy', tot_acc.item()/T)
    print('Query frequency', float(num_query)/T)

    if plot_query:
        plt.figure()
        plt.plot([min(1,x*alpha) for x  in delta_rec])
        plt.axvline(x=125, linestyle='--', color='orange', linewidth=5)
        plt.axvline(x=125+250, linestyle='--', color='orange', linewidth=5)
        plt.xlabel('t')
        plt.ylabel('Query probability')
        plt.title("Query budget {0:.0%}".format(float(num_query)/T))
        plt.tight_layout()
        plt.savefig("Query budget {0:.0%}.png".format(float(num_query)/T))

    return num_query, 1-tot_acc/T

if __name__ == "__main__":
    # Choose from 'mnist', 'portraits'
    dataset = sys.argv[1]
    # Choose from '0', '1', '2'
    config = sys.argv[2]

    if dataset=='mnist':
        # For Rotated MNIST dataset
        # config = 0: 60-degree duration 500, 30-degree duration 250, 0-degree duration 125
        # config = 1: 60-degree duration 125, 30-degree duration 250, 0-degree duration 500
        # config = 2: 60-degree duration 250, 30-degree duration 250, 60-degree duration 125, 0-degree duration 125, 60-degree duration 125
        if config=='1':
            t_u = [125, 250, 500]
        else:
            t_u = [500, 250, 125]
        X, Y = load_mnist_data('mnist_file', t_u)
        if config=='2':
            X1 = [X[250:-125], X[:125], X[-125:], X[125:250]]
            Y1 = [Y[250:-125], Y[:125], Y[-125:], Y[125:250]]
            X = torch.cat(X1, dim=0)
            Y = torch.cat(Y1, dim=0)
        alpha_list = [0.1, 0.2, 0.4, 0.6, 0.8, 1.2, 1.5, 2, 3]
    else:
        # For Portraits dataset
        # config = 0: durations 512, 256, 128, 64, 32
        # config = 1: durations 32, 64, 128, 256, 512
        # config = 2: durations 200, 200, 200, 200, 200
        if config=='0':
            t_u = [512, 256, 128, 64, 32]
        elif config=='1':
            t_u = [32, 64, 128, 256, 512]
        else:
            t_u = [200, 200, 200, 200, 200]
        X, Y = load_portraits_data('../gradual_domain_adaptation/dataset_32x32.mat', t_u)
        alpha_list = [0.15, 0.2, 0.4, 0.6, 0.8, 1.2, 1.5, 2, 3, 5]

    if len(sys.argv)>3:
        alpha_list = [0.25, 0.5, 1]
        for s in alpha_list:
            _, _ = adapt_loss(X, Y, alpha=s, thr=0, mode='qufur', plot_query=True)
        sys.exit()

    num_exp = 5

    rand_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    det_thr= [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 32, 64]

    rand_q = np.zeros((num_exp, len(rand_prob)))
    rand_r = np.zeros((num_exp, len(rand_prob)))
    loss_greedy_q = np.zeros(len(rand_prob))
    loss_greedy_r = np.zeros(len(rand_prob))
    delta_q = np.zeros((num_exp, len(alpha_list)))
    delta_r = np.zeros((num_exp, len(alpha_list)))

    start = time.time()
    end = start

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

        # Qufur
        x1 = []
        y1 = []
        for s in alpha_list:
            x, yy = adapt_loss(X, Y, alpha=s, thr=0, mode='qufur')
            if (x > 0.01*len(Y)):
                x1.append(x)
                y1.append(yy)
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
    plt.ylabel('Average 0-1 Loss')
    plt.legend()
    if dataset=='mnist':
        plt.title('Rotated MNIST Dataset')
    else:
        plt.title('Portraits Dataset')
    plt.tight_layout()
    if dataset=='mnist':
        plt.savefig('MNIST_tradeoff.png')
    else:
        plt.savefig('Portraits_tradeoff.png')
