import random
import json
import torch
import numpy as np
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import torch.nn as nn

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Portfolio Chatbot ---
with open('intents.json', 'r') as json_data:
    intents_portfolio = json.load(json_data)
FILE_PORTFOLIO = "portfolio.pth"
data_portfolio = torch.load(FILE_PORTFOLIO, map_location=device)
input_size_p = data_portfolio["input_size"]
hidden_size_p = data_portfolio["hidden_size"]
output_size_p = data_portfolio["output_size"]
all_words_p = data_portfolio['all_words']
categories_p = data_portfolio['tags']
model_state_p = data_portfolio["model_state"]
model_portfolio = NeuralNet(input_size_p, hidden_size_p, output_size_p).to(device)
model_portfolio.load_state_dict(model_state_p)
model_portfolio.eval()

def get_response_portfolio(sentence):
    sentence_tokens = tokenize(sentence)
    X = bag_of_words(sentence_tokens, all_words_p)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model_portfolio(X)
    _, predicted = torch.max(output, dim=1)
    category = categories_p[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents_portfolio['intents']:
            if category == intent["category"]:
                return random.choice(intent['responses'])
    return "I do not understand. Can you please provide an appropriate response?"

# --- CodeArenas Chatbot ---
with open('chatdata.json', 'r') as json_data:
    intents_codearenas = json.load(json_data)
FILE_CODEARENAS = "codearenas.pth"
data_codearenas = torch.load(FILE_CODEARENAS, map_location=device)
input_size_c = data_codearenas["input_size"]
hidden_size_c = data_codearenas["hidden_size"]
output_size_c = data_codearenas["output_size"]
all_words_c = data_codearenas['all_words']
categories_c = data_codearenas['tags']
model_state_c = data_codearenas["model_state"]
model_codearenas = NeuralNet(input_size_c, hidden_size_c, output_size_c).to(device)
model_codearenas.load_state_dict(model_state_c)
model_codearenas.eval()

def get_response_codearenas(sentence):
    sentence_tokens = tokenize(sentence)
    X = bag_of_words(sentence_tokens, all_words_c)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model_codearenas(X)
    _, predicted = torch.max(output, dim=1)
    category = categories_c[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents_codearenas['intents']:
            if category == intent["category"]:
                return random.choice(intent['responses'])
    return "I do not understand. Can you please provide an appropriate response?"

# Backward compatibility for CLI (portfolio)
def get_response(sentence):
    return get_response_portfolio(sentence)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        message = sys.argv[1]
    else:
        message = sys.stdin.readline().strip()
    response = get_response(message)
    print(f"Mahi: {response}")
