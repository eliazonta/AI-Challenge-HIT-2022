import torch
from torch import nn

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

class LSTM_multiple_stations(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, num_layers, device):
        super(LSTM_multiple_stations, self).__init__()

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

    def forward(self, x, pred_fut=0):
        outputs, batch_size, sequence_length = [], x.size(0), x.size(1)

        # initialize hidden and cells
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        
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
            outputs = torch.cat((outputs, output.unsqueeze(1)), dim=1)
            
        return outputs.to(self.device)
