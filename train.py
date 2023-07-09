import json
import string
import numpy as np
from nltk_utils import tokenize, stem, create_bag_of_words 

with open('intents.json', 'r') as f:
    intents = json.load(f)
    
# print(intents)

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





