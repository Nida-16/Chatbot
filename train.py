import json
import string
import numpy as np
import torch
import torch.nn as nn
from model import NeuralNet
from torch.utils.data import Dataset,DataLoader
from nltk_utils import tokenize, stem, create_bag_of_words 


with open('intents.json', 'r') as f:
    intents = json.load(f)
all_words = []
tags = []
xy = []    # will later hold all our oatterns and tags

for i in intents['intents']:
    tag = i['tag']
    tags.append(tag)
    for pattern in i['patterns']:
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
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

# Hyperparameters
BATCH_SIZE = 8
INPUT_SIZE = len(X_train[0])
HIDDEN_SIZE = 8
OUTPUT_SIZE = len(tags)
LEARNING_RATE = 0.0001
EPOCHS = 1000
# print(INPUT_SIZE, len(all_words), len(bag))
# print(OUTPUT_SIZE, tags)

dataset = ChatDataset()
data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
# print(device)
# loss and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=EPOCHS)

for epoch in range(EPOCHS):
    for (words, labels) in data_loader:
        words = words.to(device)
        labels = labels.to(device)
        labels = labels.long()
        # Feed Forward Network
        predicted_ouputs = model(words)
        
        loss = criterion(predicted_ouputs, labels)
        
        # Back Propogation and Optimizers
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch %100 ==0):
        print(f"Epochs : {epoch}/{EPOCHS}, Loss : {loss.item():.4f}")
        
print(f"Final loss : {loss.item():.4f}") 

data = {
    "model_state" : model.state_dict(),
    "input_size" : INPUT_SIZE,
    "output_size" : OUTPUT_SIZE,
    "hidden_size" : HIDDEN_SIZE,
    "all_words" : all_words,
    "tags" : tags
}

FILE_NAME = "data.pth"
torch.save(data, FILE_NAME)

print("Training completed")
print(f"File saved to {FILE_NAME}")

