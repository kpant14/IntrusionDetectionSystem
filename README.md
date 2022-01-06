# Cybersecurity

data_process.m: Constructs data set from px4 log file. Uses cpuload message timestamp as base event time. Each data point consists of current reading and past 7 instances. For each cpuload message comes in, find 7 past readings and current reading of cpuload message and 8 most recent reading of other messages. 

Training.py: Reads the data set from data_process, and spilts data set to 80% training and 20% testing. Constructs and trains PyTorch long short-term memory (LSTM), gated recurrent units (GRUs), simple recurrent units (SRUs) and Simple recurrent neural network (RNN) models.


Dataset can be downloaded from the following link: https://drive.google.com/file/d/1QtQndC00yOFlM56NnifqtKkCzDwxUTQE/view?usp=sharing
