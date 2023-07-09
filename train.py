import json
import string
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from nltk_utils import tokenize, stem, create_bag_of_words 

# Hyperparameters
BATCH_SIZE = 8

with open('intents.json', 'r') as f:
    intents = json.load(f)
all_words = []
tags = []
xy = []    # will later hold all our oatterns and tags

for i in intents['intents']:
    tag = i['intent']
    tags.append(tag)
    for pattern in i['text']:
        tokens = tokenize(pattern)
        all_words.extend(tokens)
        # using append would create an array of arrays as tokens would be a list of words itself
        xy.append((tokens, tag))  # Packing in a tuple 
        
ignore_words = list(string.punctuation)
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (tokens, tag) in xy :   # Unpacking the tuple
    bag = create_bag_of_words(tokens,all_words)
    X_train.append(bag)
    
    tag = tags.index(tag)
    y_train.append(tag)
    
X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
        
    # dataset[idx]
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
dataset = Dataset()
data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=2)
    
    

