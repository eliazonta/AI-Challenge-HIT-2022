import torch
from torch import nn

class SingleLSTMCell(nn.Module):
    def __init__(self, input_size, output_size, batch_size, hidden_dim,device):
        super(SingleLSTMCell, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim

        # Define the torch device
        self.device = device

        # LSTM Cell
        self.lstm1 = nn.LSTMCell(input_size, hidden_dim)
        # self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        # Initializing hidden state only once
        self.hidden,self.cell = self.init_hidden_and_cell(batch_size)

    def forward(self, x, pred_fut=0):
        outputs, batch_size = [], x.size(0)

        # Initializing hidden state for first input using method defined below
        # hidden,cell = self.init_hidden_and_cell(batch_size)
        
        # Initializing hidden state only once
        hidden,cell = self.hidden, self.cell
        
        # x = torch.unsqueeze(x, 2)
        for time_step in x.split(1, dim=1):
            hidden,cell = self.lstm1(time_step, (hidden,cell))
            output = self.fc(hidden) # output from the last FC layer
            outputs.append(output)
        
        if(pred_fut):
            for i in range(pred_fut):
                # this only generates future predictions if we pass in future_preds>0
                # mirrors the code above, using last output/prediction as input
                hidden,cell = self.lstm1(output, (hidden,cell))
                output = self.fc(hidden)
                outputs.append(output) 
            
        # transform list to tensor    
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
    def init_hidden_and_cell(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        cell = torch.zeros(batch_size, self.hidden_dim).to(self.device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden,cell

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, num_layers, device):
        super(LSTM, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim

        # Define number of layers
        self.num_layers = num_layers

        # Define the torch device
        self.device = device

        # LSTM Cell
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_dim, num_layers = self.num_layers, 
                            batch_first=True, dropout=0.3)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        # Initializing hidden state only once
        # self.hidden,self.cell = self.init_hidden_and_cell(batch_size)

    def forward(self, x, pred_fut=0):
        outputs, batch_size, sequence_length = [], x.size(0), x.size(1)

        # initialize hidden and cells
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        x = x.unsqueeze(2)
        # the first part of the output is the already known curve
        outputs = x

        for i in range(pred_fut):
            # this generates future predictions, aslo based on subsequent predictions
            
            # selection of the window
            outputs_window = outputs[:,i:sequence_length+i,:]
            output, (h0, c0) = self.lstm(outputs_window, (h0, c0))
            output = self.fc(h0[0])
            output = nn.LeakyReLU(negative_slope=0.3)(output)
            # append the predicted point to the output
            outputs = torch.cat((outputs, output.unsqueeze(2)), dim=1)
            
        return outputs.squeeze(2).to(self.device)
    
    def init_hidden_and_cell(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        cell = torch.zeros(batch_size, self.hidden_dim).to(self.device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden,cell
