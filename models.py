import torch 
import torch.nn as nn
from sru import SRU

class LSTMModel(nn.Module):
   
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
       
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        h0 = h0.cuda()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = c0.cuda()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:,-1,:])
        return out
    
class GRUModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        h0 = h0.cuda()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = c0.cuda()
        out, hn = self.gru(x, h0)
        out = self.fc(out[:,-1,:])
        return out

class SRUModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        
        super(SRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.sru = SRU(input_dim, hidden_dim, layer_dim, has_skip_term = False, amp_recurrence_fp16=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        h0 = h0.cuda()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = c0.cuda()
        out, c = self.sru(x)
        out = self.fc(out[-1,:,:])
        return out

class RNNModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        
        hidden = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        hidden = hidden.cuda()
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:,-1,:])
        return out
    
    def init_hidden(self, batch_size):
        
        hidden = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
        return hidden