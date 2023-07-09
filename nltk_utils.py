import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

stemmer = PorterStemmer()
def stem(word):
    return stemmer.stem(word.lower())

def create_bag_of_words(sentence_tokens,all_words):
    sentence_tokens = [stem(w) for w in sentence_tokens]
    bag_of_words = np.zeros(len(all_words),dtype=np.int32)
    for idx,word in enumerate(all_words):
        if word in sentence_tokens:
            bag_of_words[idx] = 1
    return bag_of_words




