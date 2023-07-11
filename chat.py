import random
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize,create_bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as f:
    intents = json.load(f)
    
FILE_NAME = "data.pth"
data = torch.load(FILE_NAME)
model_state = data["model_state" ]
INPUT_SIZE = data["input_size" ]
OUTPUT_SIZE = data["output_size" ]
HIDDEN_SIZE = data["hidden_size" ]
all_words = data["all_words" ]
tags = data["tags"] 

model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "Athena"
print("Lets chat! Type 'quit' to exit")
while (True):
    sentence = input("You: ")
    if(sentence=="quit"):
        break
    
    sentence = tokenize(sentence)
    X = create_bag_of_words(sentence,all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    
    output = model(X)
    _,predicted = torch.max(output,dim=1)
    
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if(prob.item() > 0):
        for i in intents['intents']:
            if (tag==i['tag']):
                print(f"{bot_name}: {random.choice(i['responses'])}")
    else :
        print(f"{bot_name}: Sorry couldn't interpret")


