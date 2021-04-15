from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

import os
import torch
import nltk
import numpy as np
import string

from collections import Counter
from torchsummary import summary
from matplotlib import pyplot as plt

from tqdm import tqdm_notebook

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score

import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

class_map = {
    0: 'not onion (true)',
    1: 'onion',
    2: 'fake'
}

# load vocabulary
with open('vocab2index.pickle', 'wb') as handle:
    vocab2index = pickle.load(handle)
# vocab2index = {"": 0, "UNK": 1}
# words = ["", "UNK"]
# for word in tqdm_notebook(counts):
#     vocab2index[word] = len(words)
#     words.append(word)


#tokenization & lemmatization
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokenized = word_tokenize(text)
    return [lemmatizer.lemmatize(token) for token in tokenized]


def encode_sentence(text, vocab2index, N=79):
    tokenized = preprocess_text(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"])
                     for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length


class LSTM_variable_input(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(0.3)
        self.embeddings = torch.nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, 5)

    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(
            x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out


def evaluate_text(text, model):
    encoded_headline = encode_sentence(text, vocab2index)
    encoded_input = np.array([encoded_headline[0]]).astype(np.int32)
    sample_tensor = torch.from_numpy(encoded_input).long().cuda()
    l = encoded_headline[1]

    model.eval()
    y_hat = model(sample_tensor, torch.tensor([l]))
    pred = torch.max(y_hat, 1)[1].cpu().item()
    return class_map[pred]


model = LSTM_variable_input(vocab_size=34366, embedding_dim=79, hidden_dim=79)
model = model.cuda().float()
model.load_state_dict(torch.load('RNN.pth'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict_fun():

    if request.method == 'POST':
        message = request.form['message']
        my_prediction = evaluate_text(model, message)

    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run()
