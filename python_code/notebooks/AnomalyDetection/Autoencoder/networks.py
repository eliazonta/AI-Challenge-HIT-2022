""" Network architectures.
"""

import torch
from torch import nn
import numpy as np

class AE(nn.Module):
    ''' Class for the AE using Fully Connected 
    '''
    def __init__(self,opt):
        super().__init__()
        self.c1 = nn.Conv1d(1,32,7,padding=2,padding_mode='reflect',stride=2)
        self.c2 = nn.Conv1d(32,16,5,padding=2,padding_mode='reflect',stride=2)
        self.dropout = nn.Dropout(p=0.2)
        self.d1 = nn.ConvTranspose1d(16,32,2,stride=2)
        self.d2 = nn.ConvTranspose1d(32,1,2,stride=2)

    def forward(self, x): 
        x1 = self.c1(torch.unsqueeze(x,1))
        # print('1st CONV:',x1.size())
        x1 = nn.LeakyReLU(0.2)(x1)
        x1 = self.dropout(x1)
        x2 = self.c2(x1)
        # print('2nd CONV:',x2.size())
        encoded = nn.LeakyReLU(0.2)(x2)
        # print('encoded:',encoded.size())
        
        y = self.d1(encoded)
        y = nn.LeakyReLU(0.2)(y)
        # print('y_deconvolve:',y.size())

        y = self.dropout(y)
        decoded = self.d2(y) 
        decoded = nn.LeakyReLU(0.2)(decoded)
        # print('decoded:',decoded.size())

        return torch.squeeze(decoded,1)




