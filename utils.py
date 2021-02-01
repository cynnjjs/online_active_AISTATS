import torch
from torchvision import datasets, transforms
import numpy as np
from scipy import ndimage, io
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from transformers import AutoTokenizer, AutoModel

transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,)),
                                  ])
def shuffle(xs, ys):
    indices = list(range(len(xs)))
    np.random.shuffle(indices)
    return xs[indices], ys[indices]

def load_mnist_data(filename, t_u):
    trainset = datasets.MNIST(filename, download=True, train=True, transform=transform)
    num = sum(t_u)
    images = trainset.data[:num].float() / 255.0
    labels = trainset.targets[:num]

    images, labels = shuffle(images, labels)

    # Perform rotations, 30 degrees at a time
    num_domains = len(t_u)
    idx = 0
    X = []
    for i in range(num_domains):
        for j in range(t_u[i]):
            rand = np.random.rand()*60
            angle = (2-i)*30
            X.append(ndimage.rotate(images[idx], angle, reshape=False))
            idx += 1

    return torch.from_numpy(np.array(X)).view(num, -1), labels

def load_portraits_data(filename, t_u):
    interval = 8000
    data = io.loadmat(filename)
    xs = np.squeeze(data['Xs'])
    ys = data['Ys'][0]
    num_domains = len(t_u)
    num = sum(t_u)
    idx = 0
    X = []
    Y = []
    for i in range(num_domains):
        shuffled = shuffle(xs[idx:idx+t_u[i]], ys[idx:idx+t_u[i]])
        X = X + [shuffled[0]]
        Y = Y + [shuffled[1]]
        print('Domain', i, 'image indices range from', idx, 'to', idx+t_u[i]-1)
        idx += interval
    X = np.concatenate(X)
    Y = np.concatenate(Y)

    return torch.reshape(torch.from_numpy(X), (num, -1)), torch.from_numpy(Y).long()

def process_text_data(myfile, n):
    df = pd.read_json(myfile, lines=True)
    X = []
    y = []
    num_0 = 0
    num_1 = 0
    for index, row in df.iterrows():
        rev = row['reviewText']
        score = row['overall']
        X.append(rev)
        y.append(score)
        if (len(y)>=n):
            break
    return X, y

def process_amazon_data(base_dir):
    n1 = 1200
    n2 = 600
    n3 = 300
    model_str = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    model = AutoModel.from_pretrained(model_str)
    for param in model.parameters():
        param.requires_grad = False

    X1, Y1 = process_text_data(os.path.join(base_dir, "reviews_Video_Games_5.json.gz"), n1)
    X2, Y2 = process_text_data(os.path.join(base_dir, "reviews_Grocery_and_Gourmet_Food_5.json.gz"), n2)
    X3, Y3 = process_text_data(os.path.join(base_dir, "reviews_Automotive_5.json.gz"), n3)
    raw_X = X1+X2+X3
    Y = Y1+Y2+Y3

    X = []
    for x in raw_X:
        num_chunks = int((len(x)-1)/512)+1
        avg_we = np.zeros((1, 768))
        for b in range(num_chunks):
            x1 = x[b*512:(b+1)*512]
            encoded_input = tokenizer.encode_plus(x1, add_special_tokens=True, \
                max_length=None, return_tensors = 'pt')
            model_output = model(encoded_input["input_ids"],  \
                token_type_ids=encoded_input["token_type_ids"])
            avg_we += torch.mean(model_output[0], dim=1).numpy()
        X.append(avg_we/num_chunks)

    pickle.dump(X, open(f'Bert_X_words.pkl', 'wb'))
    pickle.dump(Y, open(f'Bert_Y_words.pkl', 'wb'))
    print('Amazon dataset BERT embedding saved.')

if __name__ == "__main__":
    process_amazon_data(base_dir='.')
