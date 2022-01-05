# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 10:18:51 2021

@author: WEI-CHENG HSU
Purdue University - M.S. in Aeronautics and Astronautics

Thesis: Lightweight Cyberattack Intrusion Detection System for Unmanned
        Aerial Vehicles using Recurrent Neural Networks.

Toolbox and function:

Pytorch:
    git clone https://github.com/pytorch/pytorch.git

Long short-term memory:
    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

Gated recurrent units:
    https://pytorch.org/docs/stable/generated/torch.nn.GRU.html        

Simple recurrent units:
@inproceedings{lei2018sru,
  title={Simple Recurrent Units for Highly Parallelizable Recurrence},
  author={Tao Lei and Yu Zhang and Sida I. Wang and Hui Dai and Yoav Artzi},
  booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
  year={2018}
}

git clone https://github.com/asappresearch/sru.git

Simple recurrent neural network:
    https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
"""

from torch.utils.data import DataLoader
import torch 
import torch.nn as nn

import numpy as np
from sru import *
import time
import matplotlib.pyplot as plt
from models import GRUModel,SRUModel,LSTMModel, RNNModel
import dataset
from utils import print_size_of_model



file_dir = 'Data/'
# Get dataset from the directory
dtrainset,dtestset = dataset.get_dataset(file_dir)


batch_size = 32
n_iters = 30000
data_size = len(dtrainset)+ len(dtestset)

num_epochs = n_iters / (data_size / batch_size)
num_epochs = int(num_epochs)+1

train_loader = DataLoader(dataset=dtrainset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

test_loader = DataLoader(dataset=dtestset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


input_dim = 32
hidden_dim = 16
layer_dim = 1 
output_dim = 3
seq_dim = 8


'''
LSTM
'''
model_lstm = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
model_lstm = model_lstm.cuda()
model_lstm = model_lstm.float()

print(model_lstm)
print(len(list(model_lstm.parameters())))
for i in range(len(list(model_lstm.parameters()))):
    print(list(model_lstm.parameters())[i].size())

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model_lstm.parameters(), lr = 0.00001)

start = time.time()  

hist_lstm = []
hist_lstm_loss = []
hist_lstm_test_loss = []
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, seq_dim, input_dim)
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model_lstm(images.float())
        
        loss = criterion(outputs, labels)
        hist_lstm_loss.append(loss.item())
        loss.backward()

        optimizer.step()
        
        iter += 1
        
        if iter % 100 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.view(-1, seq_dim, input_dim)
                images = images.cuda()
                labels = labels.cuda()

                outputs = model_lstm(images.float())
                loss = criterion(outputs, labels)
                hist_lstm_test_loss.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            hist_lstm.append(accuracy)

        if iter%1000 == 0:
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
            
end = time.time()
print('Time elapsed (lstm):', round(end-start,2), 's')    
total_params_lstm = sum(p.numel() for p in model_lstm.parameters() if p.requires_grad)
print('Total trainable parameters in model (lstm):', total_params_lstm)
total_params_lstm = sum(p.numel() for p in model_lstm.parameters())
print('Total parameters in model (lstm):', total_params_lstm)
model_lstm.to('cpu')
print_size_of_model(model_lstm)


'''
GRU
'''

model_gru = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)
model_gru = model_gru.cuda()
model_gru = model_gru.float()
print(model_gru)
print(len(list(model_gru.parameters())))
for i in range(len(list(model_gru.parameters()))):
    print(list(model_gru.parameters())[i].size())

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model_gru.parameters(), lr = 0.00001)  

start = time.time()

hist_gru = []
hist_gru_loss = []
hist_gru_test_loss = []
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, seq_dim, input_dim)
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model_gru(images.float())
        
        loss = criterion(outputs, labels)
        hist_gru_loss.append(loss.item())
        loss.backward()

        optimizer.step()
        
        iter += 1
        
        if iter % 100 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.view(-1, seq_dim, input_dim)
                images = images.cuda()
                labels = labels.cuda()

                outputs = model_gru(images.float())
                loss = criterion(outputs, labels)
                hist_gru_test_loss.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            hist_gru.append(accuracy)

        if iter%1000 == 0:
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
            
end = time.time()
print('Time elapsed (gru):', round(end-start,2), 's')    
total_params_gru = sum(p.numel() for p in model_gru.parameters() if p.requires_grad)
print('Total trainable parameters in model (gru):', total_params_gru)
total_params_gru = sum(p.numel() for p in model_gru.parameters())
print('Total parameters in model (gru):', total_params_gru)
model_gru.to('cpu')
print_size_of_model(model_gru)
        
'''
SRU
'''

model_sru = SRUModel(input_dim, hidden_dim, layer_dim, output_dim)
model_sru = model_sru.cuda()
model_sru = model_sru.float()

print(model_sru)
print(len(list(model_sru.parameters())))
for i in range(len(list(model_sru.parameters()))):
    print(list(model_sru.parameters())[i].size())

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model_sru.parameters(), lr = 0.0001)  

start = time.time()

hist_sru = []
hist_sru_loss = []
hist_sru_test_loss = []
iter = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, seq_dim, input_dim)
        images = images.permute(1,0,2)
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        outputs = model_sru(images.float())
        loss = criterion(outputs, labels)
        hist_sru_loss.append(loss.item())
        loss.backward()

        optimizer.step()
        iter += 1
        
        if iter % 100 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.view(-1, seq_dim, input_dim)
                images = images.permute(1,0,2)
                images = images.cuda()
                labels = labels.cuda()

                outputs = model_sru(images.float())
                loss = criterion(outputs, labels)
                hist_sru_test_loss.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            hist_sru.append(accuracy)

        if iter%1000 == 0:
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
            
end = time.time()
print('Time elapsed:', round(end-start,2), 's')    
total_params_sru = sum(p.numel() for p in model_sru.parameters() if p.requires_grad)
print('Total trainable parameters in model:', total_params_sru)
total_params_sru = sum(p.numel() for p in model_sru.parameters())
print('Total parameters in model:', total_params_sru)
model_sru.to('cpu')
print_size_of_model(model_sru)


'''
RNN
'''

model_rnn = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
model_rnn = model_rnn.cuda()
model_rnn = model_rnn.float()

print(model_rnn)
print(len(list(model_rnn.parameters())))
for i in range(len(list(model_rnn.parameters()))):
    print(list(model_rnn.parameters())[i].size())

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model_rnn.parameters(), lr = 0.0001)  

hist_rnn = []
hist_rnn_loss = []
hist_rnn_test_loss = []
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, seq_dim, input_dim)
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        outputs = model_rnn(images.float())
        
        loss = criterion(outputs, labels)
        hist_rnn_loss.append(loss.item())

        loss.backward()

        optimizer.step()
        
        iter += 1

        if iter % 100 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.view(-1, seq_dim, input_dim)
                images = images.cuda()
                labels = labels.cuda()

                outputs = model_rnn(images.float())
                loss = criterion(outputs, labels)
                hist_rnn_test_loss.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / total
            hist_rnn.append(accuracy)

        if iter%1000 == 0:
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
            
end = time.time()
print('Time elapsed (rnn):', round(end-start,2), 's')    
total_params_rnn = sum(p.numel() for p in model_rnn.parameters() if p.requires_grad)
print('Total trainable parameters in model (rnn):', total_params_rnn)
total_params_rnn = sum(p.numel() for p in model_rnn.parameters())
print('Total parameters in model (rnn):', total_params_rnn)
model_rnn.to('cpu')
print_size_of_model(model_rnn)

plt.figure()
plt.plot(100*np.linspace(1,len(hist_sru), num =len(hist_sru)), hist_sru, label='SRU')
plt.plot(100*np.linspace(1,len(hist_lstm), num =len(hist_lstm)), hist_lstm, label='LSTM')
plt.plot(100*np.linspace(1,len(hist_gru), num =len(hist_gru)), hist_gru, label='GRU')
plt.plot(100*np.linspace(1,len(hist_rnn), num =len(hist_rnn)), hist_rnn, label='RNN')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Accuracy (%)')
        
print('Max accuracy for SRU:', max(hist_sru), '%')
print('Max accuracy for LSTM:', max(hist_lstm), '%')
print('Max accuracy for GRU:', max(hist_gru), '%')
print('Max accuracy for RNN:', max(hist_rnn), '%')

